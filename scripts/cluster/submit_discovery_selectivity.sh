#!/bin/bash
#SBATCH --job-name=glp-discovery-selectivity
#SBATCH --account=bsc70
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Discovery section Part 3 (scripts/discovery_selectivity.py): builds the full
# winning-unit x concept AUC matrix (probe_concept never keeps this — only each
# concept's winner survives run_probing.py) for GLP/SAE/raw_layer/raw_mlp, then
# computes each winning unit's selectivity ratio and concept-count-above-0.8.
# ~300 winning units x 104 concepts single-feature fits, same _fit_one/joblib
# machinery run_probing.py already uses at a bigger scale (512 candidates x 104
# concepts x 4 methods) — 2h is a conservative ceiling, resumable via
# discovery_selectivity_partial.jsonl if it doesn't finish (resubmit to continue).
#
# Run AFTER run_probing.py (--metric auc) — needs its probing.json for the winner list.
#
# Submit from the repo root:
#   sbatch scripts/cluster/submit_discovery_selectivity.sh
#   sbatch scripts/cluster/submit_discovery_selectivity.sh \
#       runs/uniref50_500m_4096_d24_steering/config.yaml \
#       runs/uniref50_500m_4096_d24_steering \
#       "" /gpfs/scratch/bsc70/bsc070332/glp-prot-data/uniref50_500m/swissprot/processed \
#       runs/uniref50_500m_4096_d24/checkpoints/glp_final.pt \
#       runs/uniref50_500m_4096_d24/checkpoints/rep_statistics.pt
#     Args 5-6 (ckpt/scaler) matter for *_steering run dirs: their own checkpoints/
#     is empty (the trained model lives under the un-suffixed sibling run).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env_mn5.sh

CONFIG="${1:-configs/esm8m_h100.yaml}"
OUT_DIR="${2:-}"
N_SHARDS_PER_SPLIT="${3:-}"
EVAL_DIR="${4:-}"
CKPT="${5:-}"
SCALER="${6:-}"

RUN_OUT_DIR="$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output_dir'])")"
RUN_NAME="$(basename "$RUN_OUT_DIR")"
record_launch "$RUN_NAME" "$@"

ARGS=(--config "$CONFIG")
[ -n "$OUT_DIR" ] && ARGS+=(--out-dir "$OUT_DIR")
[ -n "$N_SHARDS_PER_SPLIT" ] && ARGS+=(--n-shards-per-split "$N_SHARDS_PER_SPLIT")
[ -n "$EVAL_DIR" ] && ARGS+=(--eval-dir "$EVAL_DIR")
[ -n "$CKPT" ] && ARGS+=(--ckpt "$CKPT")
[ -n "$SCALER" ] && ARGS+=(--scaler "$SCALER")

python scripts/discovery_selectivity.py "${ARGS[@]}"
