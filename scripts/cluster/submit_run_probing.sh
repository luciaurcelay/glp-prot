#!/bin/bash
#SBATCH --job-name=glp-probing
#SBATCH --account=bsc70
#SBATCH --qos=acc_bsccs
#SBATCH --partition=acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 1-D probing: GLP meta-neurons vs SAE vs raw-layer vs raw-MLP (PLAN §6.5, Deliverable 2).
# Run scripts/cluster/submit_freeze_concepts.sh FIRST (§7 fairness rule — freeze before probing).
# Resumable: safe to resubmit after a timeout/crash, it picks up metrics/probing_partial*.jsonl.
# Submit from the repo root:
#   sbatch scripts/cluster/submit_run_probing.sh                                # metric=auc
#   sbatch scripts/cluster/submit_run_probing.sh configs/esm8m_h100.yaml f1     # metric=f1

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env_mn5.sh

CONFIG="${1:-configs/esm8m_h100.yaml}"
METRIC="${2:-auc}"
OUT_DIR="$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output_dir'])")"
RUN_NAME="$(basename "$OUT_DIR")"
# Data-sized, not symlinked under runs/ — reached directly at $GLP_DATA_ROOT/<run>/.
EVAL_DIR="$GLP_DATA_ROOT/$RUN_NAME/swissprot/processed"
CONCEPTS="$EVAL_DIR/frozen_concepts.json"

python scripts/run_probing.py --config "$CONFIG" --eval-dir "$EVAL_DIR" \
  --concepts "$CONCEPTS" --metric "$METRIC"
