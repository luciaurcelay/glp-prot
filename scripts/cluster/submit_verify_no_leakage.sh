#!/bin/bash
#SBATCH --job-name=glp-verify-leakage
#SBATCH --account=bsc70
#SBATCH --qos=acc_debug
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Train/eval leakage assertion (GLProt probing spec Part 2.2) — CPU-only, no GPU
# needed (reads FASTA/TSV, hashes sequences). Run before run_probing.py, or rely on
# its own default gate (skips with a warning if train_manifest.json is missing, since
# extract_activations.py only started writing one after this feature landed).
# Submit from the repo root:
#   sbatch scripts/cluster/submit_verify_no_leakage.sh [config] [store] [eval_dir]

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env_mn5.sh

CONFIG="${1:-configs/esm8m_h100.yaml}"
STORE="${2:-train}"
EVAL_DIR="${3:-}"
OUT_DIR="$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output_dir'])")"
RUN_NAME="$(basename "$OUT_DIR")"
record_launch "$RUN_NAME" "$@"

EVAL_ARGS=()
[ -n "$EVAL_DIR" ] && EVAL_ARGS=(--eval-dir "$EVAL_DIR")
python scripts/verify_no_leakage.py --config "$CONFIG" --store "$STORE" "${EVAL_ARGS[@]}"
