#!/usr/bin/env python
"""Freeze the pre-registered headline concept set (PLAN §7) -> frozen_concepts.json.

`⚠ FAIRNESS`: run this BEFORE any probing. Selection uses only InterPLM's frequency
filter, split statistics, and the SAE's own tractability — never GLP output. Commit
the resulting frozen_concepts.json.
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from glp_plm.config import load_config
from glp_plm.data import swissprot_labels as sl
from glp_plm.eval import baselines as bl
from glp_plm.eval import concept_selection as cs
from glp_plm.eval.probing_1d import fast_auc_matrix
from glp_plm.utils import resolve_config_data_dir, resolve_storage_dir, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval-dir", help="…/swissprot/processed. Default: config data.eval_dir, "
                                        "else <output_dir>/swissprot/processed")
    ap.add_argument("--out", help="default <eval-dir>/frozen_concepts.json")
    ap.add_argument("--min-pos-test", type=int, default=500)
    ap.add_argument("--n-per-category", type=int, default=3)
    ap.add_argument("--tractability-sample", type=int, default=20000,
                    help="rows of the VALID split used to score SAE tractability")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    out = Path(cfg.output_dir)
    # Eval store precedence: CLI --eval-dir > config data.eval_dir > per-run default.
    if args.eval_dir:
        processed = Path(args.eval_dir)
    elif cfg.data.eval_dir:
        processed = resolve_config_data_dir(cfg.data.eval_dir)
    else:
        processed = resolve_storage_dir(out, "swissprot", "GLP_DATA_ROOT", symlink=False) / "processed"

    A_tr, Y_tr, names = sl.load_split(processed, "valid", cfg.data.layer)
    _, Y_te, _ = sl.load_split(processed, "test", cfg.data.layer)
    n_pos_test = np.array([int(Y_te[c].sum()) for c in names])

    # ---- SAE tractability: best single-feature val AUC (SAE-only; no GLP) ----
    from interplm.sae.inference import load_sae_from_hf
    plm = "esm2-8m" if "8M" in cfg.data.model_name else "esm2-650m"
    sae = load_sae_from_hf(plm, cfg.data.layer)

    rng = np.random.default_rng(cfg.seed)
    n = min(args.tractability_sample, A_tr.shape[0])
    idx = rng.choice(A_tr.shape[0], n, replace=False)
    F = bl.sae_bank(sae, A_tr[idx].float()).numpy()
    Y = Y_tr.iloc[idx].to_numpy().astype(np.float64)
    keep = Y.sum(axis=0) >= 10          # AUC too noisy below this
    auc = fast_auc_matrix(F, Y[:, keep])                      # (C_keep, D)
    best = np.abs(auc - 0.5).max(axis=1) + 0.5                # direction-agnostic
    tract = {c: float(b) for c, b in zip(np.array(names)[keep], best)}
    print(f"[freeze] scored SAE tractability for {len(tract)} concepts on {n:,} rows")

    sel = cs.select_concepts(names, n_pos_test, sae_tractability=tract,
                             min_pos_test=args.min_pos_test,
                             n_per_category=args.n_per_category, seed=cfg.seed)
    path = cs.save(sel, args.out or processed / "frozen_concepts.json")

    print(f"[freeze] pool: {len(sel['all_qualifying'])} qualifying concepts")
    for cat, items in sel["by_category"].items():
        print(f"          {cat:<22} {len(items)}")
    print(f"[freeze] headline set ({len(sel['headline'])}):")
    for h in sel["headline"]:
        t = h["sae_tractability"]
        tstr = f"SAE-AUC {t:.3f}" if t is not None else "(abundance fallback)"
        print(f"          [{h['category']:<22}] {h['concept'][:44]:<44} {tstr}")
    print(f"[freeze] wrote {path}")


if __name__ == "__main__":
    main()
