#!/bin/bash
#SBATCH --job-name=glp-extract
#SBATCH --account=bsc70
#SBATCH --qos=acc_bsccs
#SBATCH --partition=acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Build the unlabeled GLP training activation store (PLAN §11 / SETUP.md).
# Submit from the repo root:  sbatch scripts/cluster/submit_extract_activations.sh
# Override config/qos/time at submit time, e.g.:
#   sbatch --qos=acc_debug --time=00:30:00 scripts/cluster/submit_extract_activations.sh configs/poc_mac.yaml
#
# ⚠ KNOWN GAP (see docs/PLAN.md §9 W2, docs/INTERPLM_MAP.md): extract_activations.py
# currently only pulls a Swiss-Prot REST slice (same code path as the Mac PoC/local_mid
# runs), not a true UniRef50 streaming producer-consumer — that's still a stub
# (glp_plm/data/producer_consumer.py). --n-seqs below is sized to approximate
# configs/esm8m_h100.yaml's n_train_activations=50,000,000 budget from Swiss-Prot alone;
# if UniRef50 streaming is implemented later, swap this script's invocation instead of
# changing configs/esm8m_h100.yaml (D8 parity contract).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env_mn5.sh

CONFIG="${1:-configs/esm8m_h100.yaml}"
N_SEQS="${2:-200000}"
# 3rd arg: the labeled eval TSV whose proteins must NOT be trained on. Leave empty ONLY
# if the training corpus is genuinely disjoint (e.g. a real UniRef50 stream). The default
# Swiss-Prot queries overlap the eval set ~98% — see docs/PLAN.md §10 risk register.
EVAL_TSV="${3:-}"
if [[ -z "$EVAL_TSV" ]]; then
  RUN_DIR="$(python -c "from glp_plm.config import load_config; print(load_config('$CONFIG').output_dir)")"
  CAND="$RUN_DIR/swissprot/swissprot_annotated_eval.tsv"
  [[ -f "$CAND" ]] && EVAL_TSV="$CAND"
fi

EXCLUDE_ARGS=()
if [[ -n "$EVAL_TSV" ]]; then
  echo "[submit] excluding eval proteins listed in $EVAL_TSV"
  EXCLUDE_ARGS=(--exclude-from "$EVAL_TSV")
else
  echo "[submit] WARNING: no eval TSV found — training may overlap the eval set." >&2
  echo "[submit]          build the eval set first, or pass it as the 3rd argument." >&2
fi

python scripts/extract_activations.py --config "$CONFIG" --store train \
  --n-seqs "$N_SEQS" --embed-batch-size 64 "${EXCLUDE_ARGS[@]}"
