#!/usr/bin/env python
"""Build the labeled Swiss-Prot eval store (PLAN §11, W1 Step 3).

Fetch InterPLM's 1k dense-annotated subset -> extract_annotations -> prepare_eval_set
-> embed each shard at the hooked layer, aligned to per-AA labels (docs/INTERPLM_MAP.md).
Prints a sanity report (rows aligned, d, a concept's positive prevalence).
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401  (puts repo root + InterPLM on sys.path)
from glp_plm.config import load_config
from glp_plm.data import swissprot_labels as sl
from glp_plm.utils import resolve_shared_dir, resolve_storage_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--min-required-instances", type=int, default=10,
                    help="subcategory-inclusion threshold for extract_annotations (README walkthrough uses 10)")
    ap.add_argument("--n-prot", type=int, default=2000,
                    help="densely-annotated proteins to pull for the eval set")
    args = ap.parse_args()

    cfg = load_config(args.config)
    # The raw TSV is identical for any run pulling the same --n-prot — shared across
    # runs (like UniRef50/the training FASTA), keyed by n_prot so different pulls
    # don't collide. `processed/` embeds it at this run's model+layer, so THAT stays
    # per-run (redirected under $GLP_DATA_ROOT/<run>/swissprot/processed) — data-sized
    # (per-AA labels aligned to activations), so NOT symlinked back under runs/.
    subset = resolve_shared_dir("swissprot", "GLP_DATA_ROOT") / f"swissprot_annotated_eval_{args.n_prot}.tsv"
    processed = resolve_storage_dir(Path(cfg.output_dir), "swissprot", "GLP_DATA_ROOT", symlink=False) / "processed"

    # The shipped 1k subset is not committed upstream (README-only), so pull a fresh
    # densely-annotated Swiss-Prot slice via the same REST query family (docs/INTERPLM_MAP.md).
    sl.fetch_uniprot_annotated_tsv(subset, n_prot=args.n_prot)
    sl.build_annotations(subset, processed, n_shards=cfg.data.n_shards,
                         min_required_instances=args.min_required_instances)
    sl.build_eval_subsets(
        processed,
        valid_range=tuple(cfg.data.valid_shard_range),
        test_range=tuple(cfg.data.test_shard_range),
        min_aa=cfg.data.min_aa_per_concept,
        min_domains=cfg.data.min_domains_per_concept,
    )
    all_shards = list(range(cfg.data.n_shards))
    sl.embed_shards(processed, all_shards, cfg.data.model_name, cfg.data.layer, device=cfg.device)

    # sanity report
    acts, labels_df, concepts = sl.load_split(processed, "valid", cfg.data.layer)
    print(f"[build_swissprot_eval] valid: acts={tuple(acts.shape)} concepts={len(concepts)}")
    assert acts.shape[1] == cfg.model.activation_dim, "activation dim mismatch"
    # spot-check a structural concept if present
    for probe in ("Disulfide bond", "Helix", "Beta strand"):
        col = next((c for c in concepts if probe in c), None)
        if col is not None:
            prev = float(labels_df[col].mean())
            print(f"[build_swissprot_eval] concept '{col}' positive prevalence = {prev:.4f}")
            break
    print("[build_swissprot_eval] done.")


if __name__ == "__main__":
    main()
