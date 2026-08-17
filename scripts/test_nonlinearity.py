#!/usr/bin/env python
"""Nonlinearity falsification test (GLProt probing spec Part 6; PLAN §6.5.1, listed
as MVP priority (3) — "cheap, needs a new script, settles the repackaging-vs-real-
structure question before write-up"). Run AFTER run_probing.py (`--metric auc`): reads
its `probing.json`, finds every concept where a GLP meta-neuron beat the SAE, and for
each one regresses that meta-neuron's activation on the full SAE dictionary
(glp_plm.eval.falsification.regress_on_sae_dict). High R^2 on a winning concept means
the SAE dictionary already spans that direction — repackaging, not new structure.

Regression rows are a fresh random RESIDUE subsample (not the concept's own balanced
probing subsample) — this asks a representation-level question ("is this meta-neuron
in general a linear function of the SAE dictionary?"), not a concept-conditioned one.
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from glp_plm.config import load_config
from glp_plm.data import swissprot_labels as sl
from glp_plm.eval import baselines as bl
from glp_plm.eval import falsification as fz
from glp_plm.eval.meta_neurons import meta_neuron_bank
from glp_plm.experiments import RunLogger
from glp_plm.model import GLPDenoiser
from glp_plm.utils import get_device, resolve_config_data_dir, resolve_storage_dir, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval-dir")
    ap.add_argument("--ckpt")
    ap.add_argument("--scaler")
    ap.add_argument("--probing-json", help="default <output_dir>/metrics/probing.json "
                                           "(run_probing.py --metric auc)")
    ap.add_argument("--n-train", type=int, default=20000, help="residues sampled from "
                                                                "'valid' to fit the ridge regression")
    ap.add_argument("--n-test", type=int, default=20000, help="residues sampled from "
                                                               "'test' to report R^2 on")
    ap.add_argument("--n-shards-per-split", type=int)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    out = Path(cfg.output_dir)
    ckpt_dir = resolve_storage_dir(out, "checkpoints", "GLP_RUNS_ROOT", symlink=False)
    if args.eval_dir:
        processed = Path(args.eval_dir)
    elif cfg.data.eval_dir:
        processed = resolve_config_data_dir(cfg.data.eval_dir)
    else:
        processed = resolve_storage_dir(out, "swissprot", "GLP_DATA_ROOT", symlink=False) / "processed"
    ns = args.n_shards_per_split

    run = RunLogger(out)
    probing_path = Path(args.probing_json) if args.probing_json else run.metrics_dir / "probing.json"
    data = json.loads(probing_path.read_text())
    winners = fz.glp_wins(data["results"])
    n_concepts = len({r["concept"] for r in data["results"]})
    print(f"[test_nonlinearity] {len(winners)}/{n_concepts} concepts: GLP beats SAE — "
         f"regressing each winning meta-neuron on the full SAE dictionary")
    if not winners:
        print("[test_nonlinearity] no GLP wins over SAE — nothing to falsify-check")
        (run.metrics_dir / "falsification.json").write_text(json.dumps(
            {"results": [], "r2_distribution": {"n": 0}}, indent=2))
        return

    # ---- shared random residue sample (representation-level question, not per-concept) ----
    A_val, _, _ = sl.load_split(processed, "valid", cfg.data.layer, n_shards=ns)
    A_te, _, _ = sl.load_split(processed, "test", cfg.data.layer, n_shards=ns)
    rng = np.random.default_rng(cfg.seed)
    i_tr = rng.choice(A_val.shape[0], min(args.n_train, A_val.shape[0]), replace=False)
    i_te = rng.choice(A_te.shape[0], min(args.n_test, A_te.shape[0]), replace=False)
    a_tr, a_te = A_val[i_tr].float(), A_te[i_te].float()

    from glp_plm.diffusion.scaler import Scaler
    scaler = Scaler.load(args.scaler or ckpt_dir / "rep_statistics.pt")
    z_tr, z_te = scaler.standardize(a_tr), scaler.standardize(a_te)

    model = GLPDenoiser(cfg.model).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt or ckpt_dir / "glp_final.pt", map_location=device))
    from interplm.sae.inference import load_sae_from_hf
    plm = "esm2-8m" if "8M" in cfg.data.model_name else "esm2-650m"
    sae = load_sae_from_hf(plm, cfg.data.layer)

    Fsae_tr = bl.sae_bank(sae, a_tr).numpy()
    Fsae_te = bl.sae_bank(sae, a_te).numpy()

    mn_cache = {}   # u -> (meta_neuron_bank(train), meta_neuron_bank(test)) — one forward pass per u
    results = []
    for w in winners:
        u, fi, concept = w["u"], w["feature_index"], w["concept"]
        if u not in mn_cache:
            mn_tr = meta_neuron_bank(model, z_tr, u, device=device).numpy()
            mn_te = meta_neuron_bank(model, z_te, u, device=device).numpy()
            mn_cache[u] = (mn_tr, mn_te)
        mn_tr, mn_te = mn_cache[u]
        r2 = fz.regress_on_sae_dict(Fsae_tr, mn_tr[:, fi], Fsae_te, mn_te[:, fi])
        res = fz.FalsificationResult(concept=concept, u=u, feature_index=fi, r2=r2,
                                     n_train=len(i_tr), n_test=len(i_te))
        results.append(res.to_dict())
        print(f"[test_nonlinearity] {concept[:44]:<44} u={u} unit={fi:<6} R2={r2:.3f}", flush=True)

    dist = fz.r2_distribution(results)
    print(f"[test_nonlinearity] R^2 distribution: mean {dist['mean']:.3f}  median {dist['median']:.3f}  "
         f"[{dist['p25']:.3f}, {dist['p75']:.3f}]  {dist['frac_above_0.8']*100:.0f}% above 0.8")

    (run.metrics_dir / "falsification.json").write_text(json.dumps(
        {"results": results, "r2_distribution": dist, "n_glp_wins": len(winners),
         "n_concepts": n_concepts}, indent=2))

    import csv
    csv_path = run.metrics_dir / "probing_falsification.csv"
    with open(csv_path, "w", newline="") as f:
        w_ = csv.writer(f)
        w_.writerow(["concept", "winning_meta_neuron", "r2_from_sae"])
        for r in results:
            w_.writerow([r["concept"], f"u={r['u']}:block_unit={r['feature_index']}",
                        f"{r['r2']:.6f}"])
    print(f"[test_nonlinearity] wrote {csv_path}")


if __name__ == "__main__":
    main()
