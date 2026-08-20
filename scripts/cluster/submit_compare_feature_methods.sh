#!/bin/bash
#SBATCH --job-name=glp-compare-features
#SBATCH --account=bsc70
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Three feature-selection methods (own AUC-shortlist, InterPLM-exact literal grid,
# Villegas-Garcia associated-latent rule) x SAE/GLP representations, for one concept
# group (scripts/compare_feature_methods.py). Rebuild of the ad-hoc scripts that
# produced docs/steering_history.md §15-18's numbers — those were never committed and
# are gone; this is the tracked, reproducible replacement.
#
# Budget: SAE side scans all 10240 features per concept for own+interplm_exact
# (~150-220s each) + a cheap villegas pass (~3s) -- ~6-7min/concept. "ss" (4 concepts)
# ~25-30min; "zinc" (7 concepts incl. the UNION) ~45-50min. GLP's own-only pass is
# fast (smaller dim, no 10240-feature grid). 2h leaves ample margin for both groups
# in one job if ever combined; run separately by default (one job per group below).
#
# Submit from the repo root:
#   sbatch scripts/cluster/submit_compare_feature_methods.sh ss
#   sbatch scripts/cluster/submit_compare_feature_methods.sh zinc
#   sbatch scripts/cluster/submit_compare_feature_methods.sh zinc configs/esm8m_h100.yaml sae,glp 2 64 <eval_dir>
#
# EVAL_DIR (arg 6) matters for block-depth-suffixed configs (_b3/_b6/_b12/_b24, e.g.
# the canonical configs/esm8m_h100.yaml itself, output_dir runs/uniref50_100m_4096_b12):
# they share ONE Swiss-Prot eval store under the un-suffixed base name --
# $GLP_DATA_ROOT/uniref50_100m_4096/swissprot/processed -- their own
# <output_dir>/swissprot/processed does not exist (same gotcha as
# submit_probe_sae_topk.sh's arg 8; found the hard way 2026-08-19, job 44821047).
# Defaults to that shared store for the canonical config; override for anything else.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env_mn5.sh

GROUP="${1:?usage: submit_compare_feature_methods.sh <ss|zinc> [config] [representations] [top_n] [shortlist_k] [eval_dir] [ckpt] [scaler] [out_dir]}"
CONFIG="${2:-configs/esm8m_h100.yaml}"
REPRESENTATIONS="${3:-sae,glp}"
TOP_N="${4:-2}"
SHORTLIST_K="${5:-64}"
EVAL_DIR="${6:-$GLP_DATA_ROOT/uniref50_100m_4096/swissprot/processed}"
CKPT="${7:-}"
SCALER="${8:-}"
OUT_DIR="${9:-}"
echo "[submit_compare_feature_methods] group=$GROUP config=$CONFIG eval_dir=$EVAL_DIR ckpt=${CKPT:-<default>} scaler=${SCALER:-<default>} out_dir=${OUT_DIR:-<default>}" >&2

RUN_OUT_DIR="$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output_dir'])")"
RUN_NAME="$(basename "$RUN_OUT_DIR")"
record_launch "$RUN_NAME" "$@"

ARGS=(--config "$CONFIG" --concept-group "$GROUP" --representations "$REPRESENTATIONS" \
      --top-n "$TOP_N" --shortlist-k "$SHORTLIST_K" --eval-dir "$EVAL_DIR")
[ -n "$CKPT" ] && ARGS+=(--ckpt "$CKPT")
[ -n "$SCALER" ] && ARGS+=(--scaler "$SCALER")
[ -n "$OUT_DIR" ] && ARGS+=(--out-dir "$OUT_DIR")

python scripts/compare_feature_methods.py "${ARGS[@]}"
