#!/usr/bin/env python
"""1-D logistic-regression probing (glp_plm.eval.probing_1d, the same protocol as
run_probing.py's 104-concept headline benchmark) applied to a synthetic UNION
label instead of a native Swiss-Prot column -- run_probing.py only iterates real
eval-store concept columns, so a union needs its own small driver. Reuses
build_union_labels from compare_feature_methods.py (label union, not a feature
union) and probing_1d.probe_concept's exact selection protocol (top-512 shortlist
by train mean-diff, per-candidate LogisticRegressionCV, select by val AUC, report
test AUC) so its result is directly comparable to the other probing.json rows.

Usage:
  python scripts/probe_union.py --config configs/configs_lucia/esm8m_500m_4096_d24.yaml \\
    --eval-dir /gpfs/scratch/bsc70/bsc070332/glp-prot-data/uniref50_100m_4096/swissprot/processed \\
    --ckpt .../glp_final.pt --scaler .../rep_statistics.pt --methods sae,glp
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from compare_feature_methods import ZINC_NUMBERED, ZINC_UNION_NAME, build_union_labels
from glp_plm.config import load_config
from glp_plm.data import swissprot_labels as sl
from glp_plm.eval import baselines as bl
from glp_plm.eval import probing_1d as pr
from glp_plm.eval.meta_neurons import meta_neuron_bank
from glp_plm.model import GLPDenoiser
from glp_plm.utils import get_device, resolve_config_data_dir, resolve_storage_dir, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval-dir")
    ap.add_argument("--ckpt")
    ap.add_argument("--scaler")
    ap.add_argument("--methods", default="sae,glp")
    ap.add_argument("--max-pos", type=int, default=3000)
    ap.add_argument("--neg-per-pos", type=float, default=1.0)
    ap.add_argument("--n-shards-per-split", type=int)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    out = Path(cfg.output_dir)
    ckpt_dir = resolve_storage_dir(out, "checkpoints", "GLP_RUNS_ROOT", symlink=False)
    processed = Path(args.eval_dir) if args.eval_dir else resolve_config_data_dir(cfg.data.eval_dir)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    ns = args.n_shards_per_split

    A_tr, Y_tr, names = sl.load_split(processed, "valid", cfg.data.layer, n_shards=ns, binarize=False)
    A_te, Y_te, _ = sl.load_split(processed, "test", cfg.data.layer, n_shards=ns, binarize=False)
    A_tr, A_te = A_tr.half(), A_te.half()
    print(f"[probe-union] valid {tuple(A_tr.shape)}  test {tuple(A_te.shape)}", flush=True)

    protein_id_tr = sl.residue_protein_ids(processed, "valid", ns)
    protein_id_te = sl.residue_protein_ids(processed, "test", ns)

    u_val, ov_val = build_union_labels(Y_tr, ZINC_NUMBERED)
    u_test, ov_test = build_union_labels(Y_te, ZINC_NUMBERED)
    print(f"[probe-union] {ZINC_UNION_NAME}: overlap val={ov_val} test={ov_test} "
         f"(0 expected)", flush=True)
    y_tr_full = (u_val > 0).astype(int)
    y_te_full = (u_test > 0).astype(int)

    i_tr = pr.balanced_indices(y_tr_full, args.max_pos, args.neg_per_pos, seed=cfg.seed,
                               protein_id=protein_id_tr)
    i_te = pr.balanced_indices(y_te_full, args.max_pos, args.neg_per_pos, seed=cfg.seed + 1,
                               protein_id=protein_id_te)
    y_tr, y_te = y_tr_full[i_tr], y_te_full[i_te]
    print(f"[probe-union] subsample n_train={len(y_tr)} pos={y_tr.sum()}  "
         f"n_test={len(y_te)} pos={y_te.sum()}", flush=True)

    scaler = None
    model = None
    if "glp" in methods:
        from glp_plm.diffusion.scaler import Scaler
        scaler = Scaler.load(args.scaler or ckpt_dir / "rep_statistics.pt")
        model = GLPDenoiser(cfg.model).to(device).eval()
        model.load_state_dict(torch.load(args.ckpt or ckpt_dir / "glp_final.pt", map_location=device))
    sae = None
    if "sae" in methods:
        from interplm.sae.inference import load_sae_from_hf
        plm = "esm2-8m" if "8M" in cfg.data.model_name else "esm2-650m"
        sae = load_sae_from_hf(plm, cfg.data.layer)

    a_tr, a_te = A_tr[i_tr].float(), A_te[i_te].float()
    u_grid = list(cfg.probing.u_grid)
    results = []
    for method in methods:
        if method == "glp":
            z_tr, z_te = scaler.standardize(a_tr), scaler.standardize(a_te)
            btr = torch.cat([meta_neuron_bank(model, z_tr, u, device=device) for u in u_grid], dim=1)
            bte = torch.cat([meta_neuron_bank(model, z_te, u, device=device) for u in u_grid], dim=1)
            Ftr, Fte = btr.numpy(), bte.numpy()
        elif method == "sae":
            Ftr = bl.sae_bank(sae, a_tr).numpy()
            Fte = bl.sae_bank(sae, a_te).numpy()
        else:
            raise ValueError(method)

        res = pr.probe_concept(Ftr, y_tr, Fte, y_te, ZINC_UNION_NAME, method,
                               topk=cfg.probing.topk, seed=cfg.probing.seed, n_jobs=args.n_jobs)
        if method == "glp":
            width = Ftr.shape[1] // len(u_grid)
            res.u = u_grid[res.feature_index // width]
            res.feature_index = int(res.feature_index % width)
        print(f"[probe-union] {method}: val_auc={res.val_auc:.4f} test_auc={res.test_auc:.4f} "
             f"feature_index={res.feature_index} u={res.u}", flush=True)
        results.append(res.to_dict())

    out_path = Path("/scratch/tmp/claude-4311/-gpfs-home-bsc-bsc070332-projects-glp-prot/"
                    "02e12189-4dc4-4824-8353-39eabd94fd74/scratchpad/probe_union_result.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[probe-union] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
