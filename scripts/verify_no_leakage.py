#!/usr/bin/env python
"""Verify zero overlap between GLP's training proteins and every evaluation protein
(GLProt probing spec Part 2.2), by exact sequence content. Not a duplicate of
`--exclude-from` (extract_activations.py) — that PREVENTS leakage at fetch time; this
VERIFIES the outcome independently, so a forgotten flag or an ID-space mismatch
(UniRef cluster ID vs UniProt accession) can't silently pass (see CLAUDE.md "Never
train on eval proteins" — this exact failure mode cost a W1.5 re-run).

Run once after `extract_activations.py` (which now always writes `train_manifest.json`
next to the store) and before `run_probing.py` — which also runs this automatically
unless `--skip-leakage-check` is passed.
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from glp_plm.config import load_config
from glp_plm.data import swissprot_labels as sl
from glp_plm.experiments import RunLogger
from glp_plm.utils import resolve_config_data_dir, resolve_storage_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--store", default="train", help="training store name (subdir under "
                                                       "output_dir) whose manifest to check")
    ap.add_argument("--train-manifest", help="override: path to train_manifest.json "
                                             "(default <store>_store/train_manifest.json, "
                                             "resolved like extract_activations.py)")
    ap.add_argument("--eval-dir", help="…/swissprot/processed. Default: config data.eval_dir, "
                                        "else <output_dir>/swissprot/processed")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(cfg.output_dir)

    if args.train_manifest:
        manifest = Path(args.train_manifest)
    elif cfg.data.store_dir:
        manifest = resolve_config_data_dir(cfg.data.store_dir) / "train_manifest.json"
    else:
        manifest = resolve_storage_dir(out, f"{args.store}_store", "GLP_DATA_ROOT",
                                       symlink=False) / "train_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(
            f"{manifest} not found — re-run extract_activations.py (it now always writes "
            f"a manifest) or pass --train-manifest explicitly."
        )

    if args.eval_dir:
        processed = Path(args.eval_dir)
    elif cfg.data.eval_dir:
        processed = resolve_config_data_dir(cfg.data.eval_dir)
    else:
        processed = resolve_storage_dir(out, "swissprot", "GLP_DATA_ROOT", symlink=False) / "processed"

    report = sl.assert_no_leakage(manifest, processed)

    run = RunLogger(out)
    path = run.metrics_dir / "leakage_report.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"[verify_no_leakage] wrote {path}")


if __name__ == "__main__":
    main()
