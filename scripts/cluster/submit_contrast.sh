#!/bin/bash
#SBATCH --job-name=glp-contrast
#SBATCH --account=bsc70
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Build the Exp B diff-of-means contrast set(s) — scripts/build_property_contrast.py.
#
# submit_steering.sh already builds ONE property's contrast inline on its first run, so
# this launcher exists for the multi-property case: several properties built together share
# a single UniRef50 scan (the pool does not depend on the property, only the quartile split
# does), which saves streaming 12 GB per extra property AND makes their natural-correlation
# reference literally identical rather than three independent estimates.
#
# Run this BEFORE the steering jobs when comparing properties, because two of its outputs
# are decisions, not just data:
#   * the length gap between each contrast's two sets, in pool-SD units. If it is large the
#     diff-of-means vector partly encodes LENGTH — rebuild with --match-length --restart.
#   * each property's alpha scale (mean||h|| / ||v||). --alpha-mode relative divides by it,
#     so the one shared --alphas grid the comparison needs is picked from these numbers.
#
# Submit from the repo root:
#   sbatch --account=ehpc394 scripts/cluster/submit_contrast.sh [property] [config] [extra args...]
# `property` defaults to `all` (gravy, charge, instability); pass a single name or a
# comma-separated list to build a subset. Existing contrast files are skipped unless
# --restart is passed through.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env_mn5.sh

PROPERTY="${1:-all}"
CONFIG="${2:-configs/configs_lucia/esm8m_500m_4096_d24.yaml}"
OUT_DIR="$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output_dir'])")"
RUN_NAME="$(basename "$OUT_DIR")"
shift $(( $# < 2 ? $# : 2 ))

record_launch "$RUN_NAME" "$CONFIG" "contrast" "$PROPERTY" "" "$@"

python scripts/build_property_contrast.py --config "$CONFIG" --property "$PROPERTY" "$@"
