#!/bin/bash
#SBATCH --job-name=glp-freeze-concepts
#SBATCH --account=bsc70
#SBATCH --qos=gp_bsccs
#SBATCH --partition=gpp
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# ⚠ FAIRNESS (PLAN §7): freeze the headline concept set BEFORE any probing run.
# CPU-only (no GLP forward pass here), so this runs on the gpp partition, not acc.
# Submit from the repo root:  sbatch scripts/cluster/submit_freeze_concepts.sh
#   sbatch scripts/cluster/submit_freeze_concepts.sh configs/esm8m_h100.yaml

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env_mn5.sh

CONFIG="${1:-configs/esm8m_h100.yaml}"
OUT_DIR="$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output_dir'])")"
RUN_NAME="$(basename "$OUT_DIR")"
# Data-sized, not symlinked under runs/ — reached directly at $GLP_DATA_ROOT/<run>/.
EVAL_DIR="$GLP_DATA_ROOT/$RUN_NAME/swissprot/processed"

# #SBATCH --output/--error can't reference $RUN_NAME (unknown until now), so Slurm
# writes them to the submission dir first; relocate to $GLP_RUNS_ROOT/<run>/ (off
# $HOME) on exit, success or failure, with a symlink back under runs/<run>/.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  finalize_logs() {
    local base="${SLURM_JOB_NAME:-glp-freeze-concepts}-${SLURM_JOB_ID}"
    mkdir -p "$GLP_RUNS_ROOT/$RUN_NAME" "runs/$RUN_NAME"
    for ext in out err; do
      [ -f "$base.$ext" ] || continue
      mv "$base.$ext" "$GLP_RUNS_ROOT/$RUN_NAME/$base.$ext"
      ln -sf "$GLP_RUNS_ROOT/$RUN_NAME/$base.$ext" "runs/$RUN_NAME/$base.$ext" 2>/dev/null || true  # no-op if runs/ already resolves into $GLP_RUNS_ROOT (symlinked wholesale)
    done
  }
  trap finalize_logs EXIT
fi

python scripts/freeze_concepts.py --config "$CONFIG" --eval-dir "$EVAL_DIR"
