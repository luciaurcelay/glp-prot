#!/usr/bin/env python
"""Compare >=2 already-finished runs against each other (PLAN §9 W2+ "scaling_curve").

Standalone, deliberately NOT part of submit_pipeline.sh — run by hand once you have
finished runs to compare (e.g. does probing AUC / generation FD improve as
n_train_activations scales up?). Renders whatever comparison figures have enough
data (skips the rest — see glp_plm.experiments.compare's module docstring) into
runs/comparisons/<name>/.

Example (the 100M vs 500M UniRef50 runs):
  python scripts/compare_runs.py runs/uniref50_100m runs/uniref50_500m
"""
import argparse

import _bootstrap  # noqa: F401
from glp_plm.experiments.compare import compare_runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", help="two or more finished run directories")
    ap.add_argument("--labels", nargs="+",
                    help="one label per run, same order as run_dirs (default: derived "
                         "from the training-store's activation count, e.g. '100M')")
    ap.add_argument("--name", help="comparison name (default: <label>_vs_<label>...)")
    ap.add_argument("--out-root", default="runs/comparisons",
                    help="where <name>/ is written (default: runs/comparisons)")
    args = ap.parse_args()

    if len(args.run_dirs) < 2:
        raise SystemExit("need at least 2 run dirs to compare")
    if args.labels and len(args.labels) != len(args.run_dirs):
        raise SystemExit("--labels must have exactly one label per run dir")

    compare_runs(args.run_dirs, labels=args.labels, name=args.name, out_root=args.out_root)


if __name__ == "__main__":
    main()
