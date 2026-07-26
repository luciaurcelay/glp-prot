#!/usr/bin/env python
"""Render all applicable figures for a run (PLAN §11).

Idempotent: renders every figure whose metric artifacts exist under the run dir.
train_glp and eval_generation already call this automatically; use this to re-render
(e.g. after restyling) or to build figures for a run produced elsewhere.
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from glp_plm.config import load_config
from glp_plm.experiments import make_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="config whose output_dir is the run")
    ap.add_argument("--run-dir", help="run dir directly (overrides --config)")
    args = ap.parse_args()
    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.config:
        run_dir = Path(load_config(args.config).output_dir)
    else:
        raise SystemExit("provide --run-dir or --config")
    make_report(run_dir)


if __name__ == "__main__":
    main()
