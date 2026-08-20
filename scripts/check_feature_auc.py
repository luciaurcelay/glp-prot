#!/usr/bin/env python
"""Quick, targeted AUC check for specific (concept, feature) pairs.

For when an `own`/`interplm_exact` winner was migrated forward from an earlier run
(compare_feature_methods.py) before AUC-of-the-winner tracking existed, and its own
AUC was never computed/saved -- rather than re-running the whole concept group, just
score the specific candidate(s) in question, same balanced-subsample AUC protocol
`compare_feature_methods.py`'s "own" method uses.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from glp_plm.config import load_config
from glp_plm.data import swissprot_labels as sl
from glp_plm.eval import baselines as bl
from glp_plm.eval import probing_1d as pr
from glp_plm.utils import resolve_config_data_dir, resolve_storage_dir, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval-dir")
    ap.add_argument("--concept", required=True)
    ap.add_argument("--features", required=True, help="comma-separated SAE feature indices")
    ap.add_argument("--max-pos", type=int, default=3000)
    ap.add_argument("--neg-per-pos", type=float, default=1.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    out = Path(cfg.output_dir)
    processed = (Path(args.eval_dir) if args.eval_dir else
                resolve_config_data_dir(cfg.data.eval_dir) if cfg.data.eval_dir else
                resolve_storage_dir(out, "swissprot", "GLP_DATA_ROOT", symlink=False) / "processed")

    A_val, labels_val, names = sl.load_split(processed, "valid", cfg.data.layer, binarize=False)
    assert args.concept in names
    y_val = labels_val[args.concept].to_numpy().astype(int)
    protein_id_val = sl.residue_protein_ids(processed, "valid")

    from interplm.sae.inference import load_sae_from_hf
    plm = "esm2-8m" if "8M" in cfg.data.model_name else "esm2-650m"
    sae = load_sae_from_hf(plm, cfg.data.layer)

    sub = pr.balanced_indices((y_val > 0).astype(int), args.max_pos, args.neg_per_pos,
                              seed=cfg.seed, protein_id=protein_id_val)
    feats = [int(f) for f in args.features.split(",")]
    sub_feats = bl.sae_bank(sae, A_val[sub].float()).numpy()[:, feats]
    aucs = pr.fast_auc_matrix(sub_feats, (y_val[sub] > 0).astype(int)[:, None])[0]
    for f, a in zip(feats, aucs):
        print(f"[check_feature_auc] {args.concept!r} feature {f}: AUC={a:.4f} "
             f"(direction as-is; {1-a:.4f} if the feature encodes it negatively)")


if __name__ == "__main__":
    main()
