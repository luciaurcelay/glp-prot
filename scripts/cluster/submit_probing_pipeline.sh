#!/bin/bash
#SBATCH --job-name=glp-probing-pipeline
#SBATCH --account=bsc70
#SBATCH --qos=acc_debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Run the full probing-extras sequence (run_probing --metric auc -> probe_continuous
# -> test_nonlinearity -> make_probing_outputs) as ONE SLURM job, sequential plain
# `python` calls — same "one acc_debug job, not N congested acc_bsccs jobs" reasoning
# as submit_pipeline.sh (see its header): `acc`/`acc_bsccs` gets congested enough that
# several dependency-chained jobs can sit queued for a long time, while `acc_debug`
# QOS actually starts, at the cost of a 2h cap and rejecting a second concurrent job
# from the same user (so bundling into one job is required, not just convenient).
#
# Every stage checkpoints incrementally (probing_partial[.jsonl],
# probing_continuous_partial.jsonl) — if the 2h cap is hit mid-stage, resubmitting
# this same script resumes rather than restarts.
#
# `--eval-dir`/`--ckpt`/`--scaler` overrides (args 2-4) matter when the checkpoint
# you want to probe doesn't live at its own config's default `<output_dir>/checkpoints`
# — e.g. an output_dir renamed after training (see the depth-scan configs under
# configs/configs_lucia/, where `_steering`-suffixed output_dirs reuse an
# un-suffixed run's checkpoint) — or when reusing a shared eval store built under a
# different run. Leave them empty to use the config's own defaults.
#
# Submit from the repo root:
#   sbatch scripts/cluster/submit_probing_pipeline.sh [config] [eval_dir] [ckpt] [scaler] \
#       [n_shards_per_split] [max_proteins]
#
# For a real (non-debug) budget once the regular queue isn't congested:
#   sbatch --qos=acc_bsccs --time=1-00:00:00 scripts/cluster/submit_probing_pipeline.sh \
#       configs/esm8m_h100.yaml

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env_mn5.sh

CONFIG="${1:-configs/esm8m_h100.yaml}"
EVAL_DIR="${2:-}"
CKPT="${3:-}"
SCALER="${4:-}"
N_SHARDS_PER_SPLIT="${5:-1}"   # matches submit_pipeline.sh's default (bounds the raw_mlp cache)
MAX_PROTEINS="${6:-2000}"

RUN_NAME="$(python -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))['output_dir'].rstrip('/').split('/')[-1])" "$CONFIG")"
echo "[submit_probing_pipeline] config=$CONFIG run=$RUN_NAME eval_dir=${EVAL_DIR:-<default>} ckpt=${CKPT:-<default>}"
record_launch "$RUN_NAME" "$@"

EVAL_ARGS=(); [ -n "$EVAL_DIR" ] && EVAL_ARGS=(--eval-dir "$EVAL_DIR")
CKPT_ARGS=(); [ -n "$CKPT" ] && CKPT_ARGS=(--ckpt "$CKPT")
[ -n "$SCALER" ] && CKPT_ARGS+=(--scaler "$SCALER")
SHARD_ARGS=(); [ -n "$N_SHARDS_PER_SPLIT" ] && SHARD_ARGS=(--n-shards-per-split "$N_SHARDS_PER_SPLIT")

echo "[submit_probing_pipeline] 1/4 run_probing (auc)"
python scripts/run_probing.py --config "$CONFIG" --metric auc \
  "${EVAL_ARGS[@]}" "${CKPT_ARGS[@]}" "${SHARD_ARGS[@]}"

echo "[submit_probing_pipeline] 2/4 probe_continuous"
python scripts/probe_continuous.py --config "$CONFIG" --max-proteins "$MAX_PROTEINS" \
  "${EVAL_ARGS[@]}" "${CKPT_ARGS[@]}" "${SHARD_ARGS[@]}"

echo "[submit_probing_pipeline] 3/4 test_nonlinearity"
python scripts/test_nonlinearity.py --config "$CONFIG" \
  "${EVAL_ARGS[@]}" "${CKPT_ARGS[@]}" "${SHARD_ARGS[@]}"

echo "[submit_probing_pipeline] 4/4 make_probing_outputs"
python scripts/make_probing_outputs.py --config "$CONFIG" "${EVAL_ARGS[@]}"

echo "[submit_probing_pipeline] done — see runs/$RUN_NAME/metrics/probing_summary.json"
