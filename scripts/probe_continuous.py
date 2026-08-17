#!/usr/bin/env python
"""Continuous-property 1-D probing (GLProt probing spec Part 1/4, continuous axis):
GLP meta-neurons vs SAE vs raw-layer vs raw-MLP against BioPython/modlAMP sequence
properties (GRAVY, MW, pI, aromaticity, instability, charge@pH7, aliphatic index,
helix/turn/sheet fraction). Mirrors run_probing.py's structure and CLI, but:

  * labels are per-PROTEIN continuous scores, not per-residue binary concepts, so
    features are per-protein POOLED activations (mean-over-residues — spec Part 4
    open item 2; same choice applied identically to all four representations);
  * a subsample of proteins per split bounds memory (GLP meta-neurons are wide:
    24*1280 = 30,720 dims at the 24-block/500M checkpoint) — pooling happens in small
    contiguous per-protein batches so peak memory is bounded by --protein-batch-size
    proteins' residues, not the whole split;
  * reported statistic is Spearman rho (glp_plm/eval/probing_continuous.py), not AUC.

GRAVY is the property that matters most: it's the same target as the DiffMean
steering experiment (build_gravy_contrast.py) — the winning meta-neuron here is the
bridge to that experiment (spec Part 7, GRAVY panel figure).
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
from glp_plm.eval import continuous_properties as cp
from glp_plm.eval import probing_continuous as prc
from glp_plm.eval.meta_neurons import meta_neuron_bank
from glp_plm.experiments import RunLogger, make_report
from glp_plm.model import GLPDenoiser
from glp_plm.utils import get_device, resolve_config_data_dir, resolve_storage_dir, set_seed


def _sample_proteins(spans, max_n: int, seed: int):
    """Random protein-level subsample (not residue-level — pooling needs each
    sampled protein's FULL residue set to stay honest)."""
    if len(spans) <= max_n:
        return spans.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(spans), max_n, replace=False))
    return spans.iloc[idx].reset_index(drop=True)


def _pooled_bank(residue_fn, A: torch.Tensor, spans, mode: str = "mean",
                 batch_proteins: int = 16) -> np.ndarray:
    """Pool per-residue features (from `residue_fn(residues) -> (n, d)`) to
    per-protein (n_protein, d), `batch_proteins` proteins' residues at a time so
    peak memory never holds more than a small batch's worth of the (possibly wide,
    e.g. 30,720-dim GLP) per-residue feature matrix."""
    agg = (lambda t: t.max(dim=0).values) if mode == "max" else (lambda t: t.mean(dim=0))
    pooled = []
    n = len(spans)
    for i in range(0, n, batch_proteins):
        chunk = spans.iloc[i:i + batch_proteins]
        pieces = [A[int(r.start):int(r.end)] for r in chunk.itertuples()]
        lens = [p.shape[0] for p in pieces]
        cat = torch.cat(pieces, dim=0)
        feats = residue_fn(cat)                        # (sum(lens), d)
        off = 0
        for L in lens:
            pooled.append(agg(feats[off:off + L]).numpy())
            off += L
        del cat, feats
    gc.collect()
    return np.stack(pooled)


def _pooled_raw_mlp(mlp_memmap, spans, mode: str = "mean") -> np.ndarray:
    """Same pooling, but reading a precomputed on-disk FFN-activation cache (no
    forward pass needed) so no batching is required."""
    agg = np.max if mode == "max" else np.mean
    return np.stack([agg(np.asarray(mlp_memmap[int(r.start):int(r.end)], dtype=np.float32),
                         axis=0)
                     for r in spans.itertuples()])


def build_bank(method: str, split_name: str, spans, A: torch.Tensor, *, model, scaler,
              sae, mlp_bank, u_grid, device, pool_mode: str, batch_proteins: int) -> np.ndarray:
    """(n_protein, dim) pooled feature bank for one representation; for GLP this pools
    the u grid separately then concatenates (spec: sweep u on validation, select per
    property), matching run_probing.py's candidate-pooling-across-u convention."""
    if method == "glp":
        per_u = []
        for u in u_grid:
            def residue_fn(cat, _u=u):
                z = scaler.standardize(cat.float())
                return meta_neuron_bank(model, z, _u, device=device)
            per_u.append(_pooled_bank(residue_fn, A, spans, pool_mode, batch_proteins))
        return np.concatenate(per_u, axis=1)
    elif method == "sae":
        return _pooled_bank(lambda cat: bl.sae_bank(sae, cat.float()), A, spans,
                            pool_mode, batch_proteins)
    elif method == "raw_layer":
        return _pooled_bank(lambda cat: cat.float(), A, spans, pool_mode, batch_proteins)
    elif method == "raw_mlp":
        return _pooled_raw_mlp(mlp_bank, spans, pool_mode)
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval-dir", help="…/swissprot/processed. Default: config data.eval_dir, "
                                        "else <output_dir>/swissprot/processed")
    ap.add_argument("--ckpt", help="GLP checkpoint (default <output_dir>/checkpoints/glp_final.pt)")
    ap.add_argument("--scaler", help="rep_statistics.pt (default <output_dir>/checkpoints/rep_statistics.pt)")
    ap.add_argument("--methods", default=",".join(bl.METHODS))
    ap.add_argument("--properties", help="comma-separated subset of "
                                         "glp_plm.eval.continuous_properties.PROPERTY_NAMES "
                                         "(default: all)")
    ap.add_argument("--max-proteins", type=int, default=2000,
                    help="protein subsample per split (bounds pooled-bank compute at "
                         "wide GLP widths)")
    ap.add_argument("--pool", choices=["mean", "max"], default="mean",
                    help="per-protein residue aggregation (spec Part 4 open item 2 — "
                         "applied identically to all four representations)")
    ap.add_argument("--protein-batch-size", type=int, default=16,
                    help="proteins per forward-pass batch during pooling")
    ap.add_argument("--n-shards-per-split", type=int)
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--train-manifest")
    ap.add_argument("--skip-leakage-check", action="store_true")
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
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    properties = ([p.strip() for p in args.properties.split(",") if p.strip()]
                 if args.properties else list(cp.PROPERTY_NAMES))
    ns = args.n_shards_per_split

    sl.leakage_gate(cfg, out, processed, args.train_manifest, args.skip_leakage_check)

    # ---- per-residue activations + protein spans (same source as run_probing.py) ----
    A_val, _, _ = sl.load_split(processed, "valid", cfg.data.layer, n_shards=ns)
    A_te, _, _ = sl.load_split(processed, "test", cfg.data.layer, n_shards=ns)
    A_val, A_te = A_val.half(), A_te.half()

    spans_val_full = sl.protein_spans(processed, "valid", ns)
    spans_te_full = sl.protein_spans(processed, "test", ns)
    assert int(spans_val_full["end"].iloc[-1]) == A_val.shape[0], "valid span/activation row mismatch"
    assert int(spans_te_full["end"].iloc[-1]) == A_te.shape[0], "test span/activation row mismatch"

    spans_val = _sample_proteins(spans_val_full, args.max_proteins, cfg.seed)
    spans_te = _sample_proteins(spans_te_full, args.max_proteins, cfg.seed + 1)
    print(f"[probe_continuous] valid {len(spans_val)}/{len(spans_val_full)} proteins  "
         f"test {len(spans_te)}/{len(spans_te_full)} proteins", flush=True)

    # ---- continuous labels (drops sequences with non-canonical residues) ----
    labels_val = cp.properties_for_proteins(spans_val["accession"].tolist(), spans_val["sequence"].tolist())
    labels_te = cp.properties_for_proteins(spans_te["accession"].tolist(), spans_te["sequence"].tolist())
    spans_val = spans_val[spans_val["accession"].isin(labels_val.index)].reset_index(drop=True)
    spans_te = spans_te[spans_te["accession"].isin(labels_te.index)].reset_index(drop=True)
    labels_val = labels_val.loc[spans_val["accession"]].reset_index(drop=True)
    labels_te = labels_te.loc[spans_te["accession"]].reset_index(drop=True)

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
    mlp_val = mlp_te = None
    if "raw_mlp" in methods:
        tag = f"s{ns}" if ns else "all"
        for split, holder, spans_full in (("valid", "val", spans_val_full), ("test", "te", spans_te_full)):
            shards = sl.split_shards(processed, split, ns)
            bank = bl.raw_mlp_cached_bank(processed, shards, cfg.data.layer, cfg.data.model_name,
                                          device, processed / "cache" / f"raw_mlp_{split}_{tag}.npy")
            assert bank.shape[0] == int(spans_full["end"].iloc[-1]), \
                f"raw_mlp/activation row mismatch ({split})"
            if holder == "val":
                mlp_val = bank
            else:
                mlp_te = bank

    u_grid = list(cfg.probing.u_grid)
    DIMS = {"glp": cfg.model.n_blocks * cfg.model.d_mlp * len(u_grid), "sae": 10240,
            "raw_layer": cfg.model.activation_dim, "raw_mlp": 1280}
    n_blocks_str = f"{cfg.model.n_blocks}-block"

    run = RunLogger(out)
    run.log_config(cfg)
    partial = run.metrics_dir / "probing_continuous_partial.jsonl"
    results = []
    if partial.exists() and not args.restart:
        results = [json.loads(l) for l in partial.read_text().splitlines() if l.strip()]
        print(f"[probe_continuous] resuming: {len(results)} (property, method) probes already done")
    done = {(r["property"], r["method"]) for r in results}

    # Methods outer, properties inner: the pooled feature bank doesn't depend on the
    # property, only on the representation — building it (a forward pass over the
    # sampled proteins, up to 30,720-dim for GLP) is the expensive step, so it must
    # happen once per method, not once per (property, method) pair.
    for method in methods:
        todo_props = [p for p in properties if (p, method) not in done]
        if not todo_props:
            continue
        width = DIMS[method] // (len(u_grid) if method == "glp" else 1)
        F_val = build_bank(method, "valid", spans_val, A_val, model=model, scaler=scaler,
                           sae=sae, mlp_bank=mlp_val, u_grid=u_grid, device=device,
                           pool_mode=args.pool, batch_proteins=args.protein_batch_size)
        F_te = build_bank(method, "test", spans_te, A_te, model=model, scaler=scaler,
                          sae=sae, mlp_bank=mlp_te, u_grid=u_grid, device=device,
                          pool_mode=args.pool, batch_proteins=args.protein_batch_size)
        for prop in todo_props:
            y_val = labels_val[prop].to_numpy(dtype=np.float64)
            y_te = labels_te[prop].to_numpy(dtype=np.float64)
            res = prc.probe_property(F_val, y_val, F_te, y_te, prop, method,
                                     topk=cfg.probing.topk, seed=cfg.probing.seed, n_jobs=args.n_jobs)
            if prop == "gravy" and method == "glp":
                # Winning meta-neuron's raw TEST activation vs GRAVY (spec Part 7 fig
                # 4 — "the bridge to the steering experiment") — saved here because
                # it needs the actual pooled activation column, not just the summary
                # stat; figures.py's fig_gravy_panel reads this back, no recompute.
                np.savez(run.metrics_dir / "gravy_panel.npz",
                        activation=F_te[:, res.feature_index], gravy=y_te,
                        u=u_grid[res.feature_index // width])
            if method == "glp":
                res.u = u_grid[res.feature_index // width]
                res.feature_index = int(res.feature_index % width)
            res_d = res.to_dict()
            res_d["pool"] = args.pool
            res_d["glp_depth"] = n_blocks_str
            results.append(res_d)
            with open(partial, "a") as f:
                f.write(json.dumps(res_d) + "\n")
            print(f"[probe_continuous] {method:<10} {prop:<20} "
                 f"test_rho={res.test_spearman:+.3f}", flush=True)
        del F_val, F_te
        gc.collect()

    # ---- aggregate + persist ----
    from glp_plm.eval.probing_1d import bootstrap_ci
    agg = {}
    for m in methods:
        vals = [r["test_spearman"] for r in results if r["method"] == m]
        mean, lo, hi = bootstrap_ci(vals)
        agg[m] = {"mean_test_spearman": mean, "ci_low": lo, "ci_high": hi, "n_properties": len(vals)}
        print(f"[probe_continuous] {m:<10} mean test rho {mean:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    (run.metrics_dir / "probing_continuous.json").write_text(json.dumps(
        {"results": results, "aggregate": agg, "u_grid": u_grid, "topk": cfg.probing.topk,
         "pool": args.pool, "glp_depth": n_blocks_str, "max_proteins": args.max_proteins}, indent=2))

    # ---- probing_continuous.csv (spec Part 7) ----
    import csv
    csv_path = run.metrics_dir / "probing_continuous.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["property", "representation", "selected_unit", "noise_level_u",
                   "val_spearman", "test_spearman"])
        for r in results:
            w.writerow([r["property"], r["method"], r["feature_index"], r.get("u", ""),
                       f"{r['val_spearman']:.6f}", f"{r['test_spearman']:.6f}"])
    print(f"[probe_continuous] wrote {csv_path}")

    run.log_summary(**{"probing_continuous_mean_test_spearman":
                       {m: agg[m]["mean_test_spearman"] for m in agg}})
    make_report(out)


if __name__ == "__main__":
    main()
