#!/bin/bash
#SBATCH --job-name=glp-probing
#SBATCH --account=bsc70
#SBATCH --qos=acc_bsccs
#SBATCH --partition=acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
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
#   sbatch scripts/cluster/submit_run_probing.sh <config> <metric> <n_shards> <eval_dir> <ckpt> <scaler>
#     — eval_dir/ckpt/scaler overrides let you probe a checkpoint whose config's own
#       output_dir doesn't hold it (e.g. an output_dir renamed after training).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env_mn5.sh

CONFIG="${1:-configs/esm8m_h100.yaml}"
METRIC="${2:-auc}"
# Arg 3: shards per split (default all). Pass 1 to match a run whose AUC was computed
# with the pipeline's default N_SHARDS_PER_SPLIT=1, so a post-hoc F1 pass scores the
# same data as that run's published AUC (e.g. re-probing the completed uniref50_100m).
N_SHARDS_PER_SPLIT="${3:-}"
EVAL_DIR="${4:-}"
CKPT="${5:-}"
SCALER="${6:-}"
OUT_DIR="$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output_dir'])")"
RUN_NAME="$(basename "$OUT_DIR")"

# Group this attempt's logs under runs/<run>/launches/<job_id>/, append the per-run
# launch ledger, and relocate Slurm's .out/.err there on exit (env_mn5.sh).
record_launch "$RUN_NAME" "$@"

# Eval store + frozen concepts resolved by the script: config data.eval_dir (reuse a
# shared store), else the per-run <output_dir>/swissprot/processed default.
SHARD_ARGS=()
[ -n "$N_SHARDS_PER_SPLIT" ] && SHARD_ARGS=(--n-shards-per-split "$N_SHARDS_PER_SPLIT")
EVAL_ARGS=()
[ -n "$EVAL_DIR" ] && EVAL_ARGS=(--eval-dir "$EVAL_DIR")
CKPT_ARGS=()
[ -n "$CKPT" ] && CKPT_ARGS=(--ckpt "$CKPT")
[ -n "$SCALER" ] && CKPT_ARGS+=(--scaler "$SCALER")
# METHODS env-var restricts the probed methods (default: all of bl.METHODS). Set e.g.
# METHODS=raw_layer for the 650M layer pilot, which has no GLP checkpoint or SAE yet
# (docs/plan_650m_scaling.md stage 1).
METHOD_ARGS=()
[ -n "${METHODS:-}" ] && METHOD_ARGS=(--methods "$METHODS")
python scripts/run_probing.py --config "$CONFIG" --metric "$METRIC" "${SHARD_ARGS[@]}" \
  "${EVAL_ARGS[@]}" "${CKPT_ARGS[@]}" "${METHOD_ARGS[@]}"
