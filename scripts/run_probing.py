#!/usr/bin/env python
"""1-D probing: GLP meta-neurons vs SAE vs raw-layer vs raw-MLP (PLAN §6.5, W3).

Deliverable 2. Trains on the eval store's *valid* split (5-fold CV inside gives the
val AUC used for selection) and reports TEST AUC on the disjoint *test* split.

GLP candidate pool spans the `u` grid, exactly as the released probe pools over
(layer, u) and selects by val AUC — so `u` is never selected on test (D9).
Baselines get the identical candidate budget (top-k) and identical subsampling.
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
from glp_plm.eval import probing_1d as pr
from glp_plm.eval.meta_neurons import meta_neuron_bank
from glp_plm.experiments import RunLogger, make_report
from glp_plm.model import GLPDenoiser
from glp_plm.utils import get_device, set_seed


def _raw_mlp_cached(processed: Path, split_shards, layer, model_name, device, cache: Path,
                    d_ff: int = 1280):
    """Compute (once) and cache the ESM FFN-intermediate bank for a split, on disk."""
    if cache.exists():
        n = cache.stat().st_size // (d_ff * np.dtype("float16").itemsize)
        return np.memmap(cache, dtype="float16", mode="r", shape=(n, d_ff))
    import pandas as pd
    rows = []
    for i in split_shards:
        df = pd.read_csv(processed / f"shard_{i}" / "protein_data.tsv", sep="\t")
        seq_col = next(c for c in df.columns if c.lower() == "sequence")
        rows.extend(str(s) for s in df[seq_col].tolist())
    print(f"[probing] building raw_mlp cache for {len(rows):,} proteins -> {cache.name}", flush=True)
    return bl.raw_mlp_to_memmap(rows, cache, layer=layer, model_name=model_name, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval-dir", required=True, help="…/swissprot/processed (labeled eval store)")
    ap.add_argument("--ckpt", help="GLP checkpoint (default <output_dir>/glp_final.pt)")
    ap.add_argument("--scaler", help="rep_statistics.pt (default <output_dir>/rep_statistics.pt)")
    ap.add_argument("--concepts", help="frozen_concepts.json; default = all qualifying")
    ap.add_argument("--max-pos", type=int, default=3000, help="cap positives per concept")
    ap.add_argument("--neg-per-pos", type=float, default=1.0)
    ap.add_argument("--min-pos", type=int, default=100, help="skip concepts below this in either split")
    ap.add_argument("--methods", default=",".join(bl.METHODS))
    ap.add_argument("--max-concepts", type=int, help="cap concepts (smoke tests)")
    ap.add_argument("--n-shards-per-split", type=int,
                    help="use only the first N shards of each split (bounds the raw_mlp cache)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    out = Path(cfg.output_dir)
    processed = Path(args.eval_dir)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    # ---- data ----
    ns = args.n_shards_per_split
    A_tr, Y_tr, names = sl.load_split(processed, "valid", cfg.data.layer, n_shards=ns)
    A_te, Y_te, _ = sl.load_split(processed, "test", cfg.data.layer, n_shards=ns)
    A_tr, A_te = A_tr.float(), A_te.float()
    print(f"[probing] valid {tuple(A_tr.shape)}  test {tuple(A_te.shape)}  concepts {len(names)}")

    from glp_plm.diffusion.scaler import Scaler
    scaler = Scaler.load(args.scaler or out / "rep_statistics.pt")
    Z_tr, Z_te = scaler.standardize(A_tr), scaler.standardize(A_te)

    model = None
    if "glp" in methods:
        model = GLPDenoiser(cfg.model).to(device).eval()
        model.load_state_dict(torch.load(args.ckpt or out / "glp_final.pt", map_location=device))
    sae = None
    if "sae" in methods:
        from interplm.sae.inference import load_sae_from_hf
        plm = "esm2-8m" if "8M" in cfg.data.model_name else "esm2-650m"
        sae = load_sae_from_hf(plm, cfg.data.layer)
    mlp_tr = mlp_te = None
    if "raw_mlp" in methods:
        tag = f"s{ns}" if ns else "all"
        for split, holder in (("valid", "tr"), ("test", "te")):
            shards = sl.split_shards(processed, split, ns)
            bank = _raw_mlp_cached(processed, shards, cfg.data.layer, cfg.data.model_name,
                                   device, processed / "cache" / f"raw_mlp_{split}_{tag}.npy")
            if holder == "tr":
                mlp_tr = bank
            else:
                mlp_te = bank
        assert mlp_tr.shape[0] == A_tr.shape[0], "raw_mlp/activation row mismatch (valid)"
        assert mlp_te.shape[0] == A_te.shape[0], "raw_mlp/activation row mismatch (test)"

    # ---- concept list ----
    if args.concepts:
        sel = json.loads(Path(args.concepts).read_text())
        concepts = sel["all_qualifying"]
        headline = {h["concept"] for h in sel["headline"]}
    else:
        concepts, headline = list(names), set()
    concepts = [c for c in concepts
                if Y_tr[c].sum() >= args.min_pos and Y_te[c].sum() >= args.min_pos]
    if args.max_concepts:
        concepts = concepts[:args.max_concepts]
    print(f"[probing] probing {len(concepts)} concepts x {len(methods)} methods")

    u_grid = list(cfg.probing.u_grid)
    results = []
    for ci, concept in enumerate(concepts, 1):
        y_tr_full = Y_tr[concept].to_numpy().astype(int)
        y_te_full = Y_te[concept].to_numpy().astype(int)
        i_tr = pr.balanced_indices(y_tr_full, args.max_pos, args.neg_per_pos, seed=cfg.seed)
        i_te = pr.balanced_indices(y_te_full, args.max_pos, args.neg_per_pos, seed=cfg.seed + 1)
        y_tr, y_te = y_tr_full[i_tr], y_te_full[i_te]

        for method in methods:
            if method == "glp":
                # pool candidates across the u grid; selection happens on val
                btr = torch.cat([meta_neuron_bank(model, Z_tr[i_tr], u, device=device) for u in u_grid], dim=1)
                bte = torch.cat([meta_neuron_bank(model, Z_te[i_te], u, device=device) for u in u_grid], dim=1)
                Ftr, Fte = btr.numpy(), bte.numpy()
            elif method == "sae":
                Ftr = bl.sae_bank(sae, A_tr[i_tr]).numpy()
                Fte = bl.sae_bank(sae, A_te[i_te]).numpy()
            elif method == "raw_layer":
                Ftr, Fte = A_tr[i_tr].numpy(), A_te[i_te].numpy()
            elif method == "raw_mlp":
                Ftr = np.asarray(mlp_tr[i_tr], dtype=np.float32)
                Fte = np.asarray(mlp_te[i_te], dtype=np.float32)
            else:
                raise ValueError(method)

            res = pr.probe_concept(Ftr, y_tr, Fte, y_te, concept, method,
                                   topk=cfg.probing.topk, seed=cfg.probing.seed, n_jobs=args.n_jobs)
            if method == "glp":
                width = Ftr.shape[1] // len(u_grid)
                res.u = u_grid[res.feature_index // width]
                res.feature_index = int(res.feature_index % width)
            res_d = res.to_dict(); res_d["headline"] = concept in headline
            results.append(res_d)
        best = {r["method"]: round(r["test_auc"], 3) for r in results if r["concept"] == concept}
        print(f"[probing] ({ci}/{len(concepts)}) {concept[:44]:<44} {best}")

    # ---- aggregate + persist ----
    run = RunLogger(out)
    agg = {}
    for m in methods:
        vals = [r["test_auc"] for r in results if r["method"] == m]
        mean, lo, hi = pr.bootstrap_ci(vals)
        agg[m] = {"mean_test_auc": mean, "ci_low": lo, "ci_high": hi, "n_concepts": len(vals)}
        print(f"[probing] {m:<10} mean test AUC {mean:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  (n={len(vals)})")
    (run.metrics_dir / "probing.json").write_text(json.dumps(
        {"results": results, "aggregate": agg, "u_grid": u_grid, "topk": cfg.probing.topk}, indent=2))
    run.log_summary(probing_mean_test_auc={m: agg[m]["mean_test_auc"] for m in agg})
    make_report(out)


if __name__ == "__main__":
    main()
