#!/usr/bin/env python
"""Depth × data-scale generation-quality comparison across finished runs (the paper's
scaling section — PLAN §9). Standalone, NOT part of submit_pipeline.sh: run by hand
once the grid runs exist. Renders FLOPs-vs-loss, FD-vs-steps, and (if a run has a
pca_proj_vs_steps.npz from scripts/pca_vs_steps.py) the sampler-convergence PCA, plus a
params/FD table, into runs/comparisons/<name>/. Generation only — no probing.

Example (the 100M + 500M depth grids):
  python scripts/compare_runs.py --name depth_scaling \
      runs/uniref50_500m_4096_d3 runs/uniref50_500m_4096 runs/uniref50_500m_4096_d12 \
      runs/uniref50_500m_4096_d24 runs/uniref50_100m_4096_b3 runs/uniref50_100m_4096_b6 \
      runs/uniref50_100m_4096_b12 runs/uniref50_100m_4096_b24
"""
import argparse

import _bootstrap  # noqa: F401
from glp_plm.experiments.compare import compare_runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", help="two or more finished run directories")
    ap.add_argument("--labels", nargs="+",
                    help="one label per run, same order as run_dirs (default: run dir name)")
    ap.add_argument("--name", help="comparison name (default: depth_scaling)")
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
