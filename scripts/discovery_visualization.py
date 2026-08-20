#!/usr/bin/env python
"""Discovery §4 — visualisation. A t-SNE projection is ILLUSTRATIVE, not evidence
(non-linear projections are hyperparameter-sensitive and can manufacture apparent
clusters) — the quantitative companion (unsupervised clustering vs. category labels,
ARI + purity) is the actual evidence; report the projection's hyperparameters in
the caption and never as a standalone finding (Discovery spec Part 4).

Uses sklearn's built-in TSNE + KMeans/HDBSCAN (no umap-learn install — decided this
session) on the SAME per-protein winning-unit pooling machinery
`discovery_hierarchy.py`/`probe_continuous.py` already use, but over ALL concepts a
GLP or SAE unit won (not just the 24 family-restricted ones from Part 2).

Run AFTER run_probing.py (--metric auc) — needs its probing.json.
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_continuous as pc  # noqa: E402

from glp_plm.config import load_config
from glp_plm.data import swissprot_labels as sl
from glp_plm.eval import baselines as bl
from glp_plm.eval.concept_selection import categorize
from glp_plm.experiments import RunLogger
from glp_plm.model import GLPDenoiser
from glp_plm.utils import get_device, resolve_config_data_dir, resolve_storage_dir, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval-dir")
    ap.add_argument("--ckpt")
    ap.add_argument("--scaler")
    ap.add_argument("--probing-json")
    ap.add_argument("--max-proteins", type=int, default=500)
    ap.add_argument("--protein-batch-size", type=int, default=16)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--n-shards-per-split", type=int)
    ap.add_argument("--out-dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    out = Path(cfg.output_dir)
    out_dir = Path(args.out_dir) if args.out_dir else out
    run = RunLogger(out_dir)
    ckpt_dir = resolve_storage_dir(out, "checkpoints", "GLP_RUNS_ROOT", symlink=False)

    if args.eval_dir:
        processed = Path(args.eval_dir)
    elif cfg.data.eval_dir:
        processed = resolve_config_data_dir(cfg.data.eval_dir)
    else:
        processed = resolve_storage_dir(out, "swissprot", "GLP_DATA_ROOT", symlink=False) / "processed"

    probing_path = Path(args.probing_json) if args.probing_json else run.metrics_dir / "probing.json"
    probing = json.loads(probing_path.read_text())
    winners = {"glp": {}, "sae": {}}
    for r in probing["results"]:
        if r["method"] in winners:
            winners[r["method"]][r["concept"]] = (r.get("u"), r["feature_index"])
    print(f"[discovery_viz] glp: {len(winners['glp'])} concepts, "
         f"sae: {len(winners['sae'])} concepts", flush=True)

    ns = args.n_shards_per_split
    A_te, Y_te, names = sl.load_split(processed, "test", cfg.data.layer, n_shards=ns, binarize=True)
    spans_full = sl.protein_spans(processed, "test", ns)
    spans = pc._sample_proteins(spans_full, args.max_proteins, seed=cfg.seed)
    print(f"[discovery_viz] pooling {len(spans)}/{len(spans_full)} test proteins", flush=True)

    from glp_plm.diffusion.scaler import Scaler
    scaler = Scaler.load(args.scaler or ckpt_dir / "rep_statistics.pt")
    model = GLPDenoiser(cfg.model).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt or ckpt_dir / "glp_final.pt", map_location=device))
    from interplm.sae.inference import load_sae_from_hf
    plm = "esm2-8m" if "8M" in cfg.data.model_name else "esm2-650m"
    sae = load_sae_from_hf(plm, cfg.data.layer)
    u_grid = list(cfg.probing.u_grid)
    width = cfg.model.n_blocks * cfg.model.d_mlp

    projections = {}
    clustering = []
    for method in ("glp", "sae"):
        concepts = sorted(winners[method])
        bank = pc.build_bank(method, "test", spans, A_te, model=model, scaler=scaler, sae=sae,
                             mlp_bank=None, u_grid=u_grid, device=device,
                             pool_mode="mean", batch_proteins=args.protein_batch_size)
        cols, cats = [], []
        for c in concepts:
            u, local = winners[method][c]
            col = u_grid.index(u) * width + local if method == "glp" else local
            cols.append(col)
            cats.append(categorize(c) or "other")
        X = bank[:, cols].T                      # (n_concepts, n_proteins) -> one row per WINNING UNIT
        del bank; gc.collect()

        from sklearn.manifold import TSNE
        Xs = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
        emb = TSNE(n_components=2, perplexity=min(args.perplexity, (len(concepts) - 1) / 3),
                  random_state=cfg.seed, init="pca").fit_transform(Xs)
        projections[method] = {"concepts": concepts, "categories": cats,
                               "xy": emb.tolist()}

        from sklearn.cluster import HDBSCAN, KMeans
        from sklearn.metrics import adjusted_rand_score
        cat_labels = np.array(cats)
        n_cat = len(set(cats))
        km = KMeans(n_clusters=n_cat, random_state=cfg.seed, n_init=10).fit_predict(Xs)
        hdb = HDBSCAN(min_cluster_size=3).fit_predict(Xs)

        def purity(pred, true):
            p = 0
            for cl in set(pred):
                idx = pred == cl
                if idx.sum() == 0:
                    continue
                vals, counts = np.unique(true[idx], return_counts=True)
                p += counts.max()
            return p / len(true)

        for cluster_method, pred in (("kmeans", km), ("hdbscan", hdb)):
            ari = float(adjusted_rand_score(cat_labels, pred))
            pur = float(purity(pred, cat_labels))
            clustering.append({"representation": method, "method": cluster_method,
                              "n_clusters": int(len(set(pred[pred >= 0]))), "ari": ari, "purity": pur})
            print(f"[discovery_viz] {method}/{cluster_method}: ARI={ari:.3f} purity={pur:.3f}",
                 flush=True)

    proj_path = run.metrics_dir / "discovery_projections.json"
    proj_path.write_text(json.dumps({
        "perplexity": args.perplexity, "n_proteins": len(spans), "seed": cfg.seed,
        "projections": projections}, indent=2))
    print(f"[discovery_viz] wrote {proj_path}")

    clust_path = run.metrics_dir / "discovery_clustering.json"
    clust_path.write_text(json.dumps(clustering, indent=2))
    print(f"[discovery_viz] wrote {clust_path}")


if __name__ == "__main__":
    main()
