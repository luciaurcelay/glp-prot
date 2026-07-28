#!/usr/bin/env python
"""1-D probing: GLP meta-neurons vs SAE vs raw-layer vs raw-MLP (PLAN §6.5, W3).

Deliverable 2. Trains on the eval store's *valid* split (5-fold CV inside gives the
val AUC used for selection) and reports TEST AUC on the disjoint *test* split.

GLP candidate pool spans the `u` grid, exactly as the released probe pools over
(layer, u) and selects by val AUC — so `u` is never selected on test (D9).
Baselines get the identical candidate budget (top-k) and identical subsampling.
"""
import argparse
import gc
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from glp_plm.config import load_config
from glp_plm.data import swissprot_labels as sl
from glp_plm.eval import baselines as bl
from glp_plm.eval import probing_1d as pr
from glp_plm.eval import probing_f1 as pf
from glp_plm.eval.meta_neurons import meta_neuron_bank
from glp_plm.experiments import RunLogger, make_report
from glp_plm.model import GLPDenoiser
from glp_plm.utils import get_device, resolve_storage_dir, set_seed


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
    ap.add_argument("--ckpt", help="GLP checkpoint (default <output_dir>/checkpoints/glp_final.pt)")
    ap.add_argument("--scaler", help="rep_statistics.pt (default <output_dir>/checkpoints/rep_statistics.pt)")
    ap.add_argument("--concepts", help="frozen_concepts.json; default = all qualifying")
    ap.add_argument("--max-pos", type=int, default=3000, help="cap positives per concept")
    ap.add_argument("--neg-per-pos", type=float, default=1.0)
    ap.add_argument("--min-pos", type=int, default=100, help="skip concepts below this in either split")
    ap.add_argument("--metric", choices=["auc", "f1"], default="auc",
                    help="auc = GLP protocol (PLAN §6.5); f1 = InterPLM domain-adjusted F1 (D5)")
    ap.add_argument("--methods", default=",".join(bl.METHODS))
    ap.add_argument("--max-concepts", type=int, help="cap concepts (smoke tests)")
    ap.add_argument("--n-shards-per-split", type=int,
                    help="use only the first N shards of each split (bounds the raw_mlp cache)")
    ap.add_argument("--restart", action="store_true", help="ignore any saved partial results")
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    out = Path(cfg.output_dir)
    ckpt_dir = resolve_storage_dir(out, "checkpoints", "GLP_RUNS_ROOT", symlink=False)
    processed = Path(args.eval_dir)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    # ---- data ----
    ns = args.n_shards_per_split
    binz = args.metric == "auc"          # F1 needs domain-instance indices
    A_tr, Y_tr, names = sl.load_split(processed, "valid", cfg.data.layer, n_shards=ns, binarize=binz)
    A_te, Y_te, _ = sl.load_split(processed, "test", cfg.data.layer, n_shards=ns, binarize=binz)
    # Keep the resident copies in fp16 and standardize only the per-concept subsample.
    # Holding raw+standardized float32 for both splits is ~3.2 GB of steady state,
    # which OOMs alongside everything else on a 16 GB machine.
    A_tr, A_te = A_tr.half(), A_te.half()
    gc.collect()
    print(f"[probing] valid {tuple(A_tr.shape)}  test {tuple(A_te.shape)}  concepts {len(names)}",
          flush=True)

    from glp_plm.diffusion.scaler import Scaler
    scaler = Scaler.load(args.scaler or ckpt_dir / "rep_statistics.pt")

    model = None
    if "glp" in methods:
        model = GLPDenoiser(cfg.model).to(device).eval()
        model.load_state_dict(torch.load(args.ckpt or ckpt_dir / "glp_final.pt", map_location=device))
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
                if (Y_tr[c] > 0).sum() >= args.min_pos and (Y_te[c] > 0).sum() >= args.min_pos]
    if args.max_concepts:
        concepts = concepts[:args.max_concepts]
    print(f"[probing] probing {len(concepts)} concepts x {len(methods)} methods")

    u_grid = list(cfg.probing.u_grid)
    # F1 fixes the probing time (conservative for GLP: no selection over u, unlike AUC).
    # u=0.9 is GLP's released default and won 88/104 concepts under AUC.
    u_fixed = max(u_grid)
    DIMS = {"glp": cfg.model.n_blocks * cfg.model.d_mlp, "sae": 10240,
            "raw_layer": cfg.model.activation_dim, "raw_mlp": 1280}

    def bank(method, split, rows, cols):
        """(len(rows), len(cols)) float32 features for arbitrary row indices."""
        A = A_tr if split == "valid" else A_te
        if method == "glp":
            z = scaler.standardize(A[rows].float())
            f = meta_neuron_bank(model, z, u_fixed, device=device).numpy()
        elif method == "sae":
            f = bl.sae_bank(sae, A[rows].float()).numpy()
        elif method == "raw_layer":
            f = A[rows].float().numpy()
        elif method == "raw_mlp":
            M = mlp_tr if split == "valid" else mlp_te
            f = np.asarray(M[rows], dtype=np.float32)
        else:
            raise ValueError(method)
        return f[:, cols]


    # Resumable: each (concept, method) is appended as it finishes, so a crash or
    # interruption costs one probe, not the whole sweep.
    run = RunLogger(out)
    suffix = "" if args.metric == "auc" else f"_{args.metric}"
    partial = run.metrics_dir / f"probing_partial{suffix}.jsonl"
    results = []
    if partial.exists() and not args.restart:
        results = [json.loads(l) for l in partial.read_text().splitlines() if l.strip()]
        print(f"[probing] resuming: {len(results)} (concept, method) probes already done")
    done = {(r["concept"], r["method"]) for r in results}

    for ci, concept in enumerate(concepts, 1):
        todo = [m for m in methods if (concept, m) not in done]
        if not todo:
            continue
        y_tr_full = Y_tr[concept].to_numpy().astype(int)
        y_te_full = Y_te[concept].to_numpy().astype(int)
        i_tr = pr.balanced_indices(y_tr_full, args.max_pos, args.neg_per_pos, seed=cfg.seed)
        i_te = pr.balanced_indices(y_te_full, args.max_pos, args.neg_per_pos, seed=cfg.seed + 1)
        y_tr, y_te = y_tr_full[i_tr], y_te_full[i_te]

        if args.metric == "f1":
            # F1 is prevalence-sensitive -> evaluate on the FULL split, streamed.
            # Candidates are still picked with the same top-k budget on the subsample.
            for method in todo:
                allc = np.arange(DIMS[method])
                sub = bank(method, "valid", i_tr, allc)
                cand = pr.topk_filter(sub, (y_tr > 0).astype(int), cfg.probing.topk)
                del sub; gc.collect()
                res = pf.run_concept_f1(
                    lambda sp, r, c, _m=method: bank(_m, sp, r, c),
                    A_tr.shape[0], A_te.shape[0], y_tr_full, y_te_full,
                    cand, concept, method, u=u_fixed if method == "glp" else None)
                res_d = res.to_dict(); res_d["headline"] = concept in headline
                results.append(res_d)
                with open(partial, "a") as f:
                    f.write(json.dumps(res_d) + "\n")
                gc.collect()
                if device == "mps" and hasattr(torch, "mps"):
                    torch.mps.empty_cache()
            best = {r["method"]: round(r["test_f1"], 3) for r in results if r["concept"] == concept}
            print(f"[probing] ({ci}/{len(concepts)}) {concept[:44]:<44} {best}", flush=True)
            continue

        a_tr, a_te = A_tr[i_tr].float(), A_te[i_te].float()   # subsample -> float32
        for method in todo:
            if method == "glp":
                # pool candidates across the u grid; selection happens on val
                z_tr, z_te = scaler.standardize(a_tr), scaler.standardize(a_te)
                btr = torch.cat([meta_neuron_bank(model, z_tr, u, device=device) for u in u_grid], dim=1)
                bte = torch.cat([meta_neuron_bank(model, z_te, u, device=device) for u in u_grid], dim=1)
                Ftr, Fte = btr.numpy(), bte.numpy()
                del btr, bte, z_tr, z_te
            elif method == "sae":
                Ftr = bl.sae_bank(sae, a_tr).numpy()
                Fte = bl.sae_bank(sae, a_te).numpy()
            elif method == "raw_layer":
                Ftr, Fte = a_tr.numpy(), a_te.numpy()
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
            with open(partial, "a") as f:        # checkpoint immediately
                f.write(json.dumps(res_d) + "\n")
            del Ftr, Fte                          # release the bank before the next method
            gc.collect()
        del a_tr, a_te
        gc.collect()
        # PyTorch's MPS allocator caches freed blocks; over 100+ concepts that cache
        # grows until the process is killed. Release it each concept.
        if device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        best = {r["method"]: round(r["test_auc"], 3) for r in results if r["concept"] == concept}
        print(f"[probing] ({ci}/{len(concepts)}) {concept[:44]:<44} {best}", flush=True)

    # ---- aggregate + persist ----
    agg = {}
    for m in methods:
        key = "test_f1" if args.metric == "f1" else "test_auc"
        vals = [r[key] for r in results if r["method"] == m]
        mean, lo, hi = pr.bootstrap_ci(vals)
        agg[m] = {f"mean_{key}": mean, "ci_low": lo, "ci_high": hi, "n_concepts": len(vals)}
        print(f"[probing] {m:<10} mean {key} {mean:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  (n={len(vals)})")
    (run.metrics_dir / f"probing{suffix}.json").write_text(json.dumps(
        {"results": results, "aggregate": agg, "u_grid": u_grid, "topk": cfg.probing.topk}, indent=2))
    run.log_summary(**{f"probing_mean_{key}": {m: agg[m][f"mean_{key}"] for m in agg}})
    make_report(out)


if __name__ == "__main__":
    main()
