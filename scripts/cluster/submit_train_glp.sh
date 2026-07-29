#!/bin/bash
#SBATCH --job-name=glp-train
#SBATCH --account=bsc70
#SBATCH --qos=acc_bsccs
#SBATCH --partition=acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Train GLP via flow matching on the activation store (PLAN §11, GATE G2 run).
# Submit from the repo root:  sbatch scripts/cluster/submit_train_glp.sh
#   sbatch scripts/cluster/submit_train_glp.sh configs/esm8m_h100.yaml
# acc_bsccs caps wall time at 2 days; bump --time / rerun if a bigger activation
# budget needs it (no checkpoint/resume in train_glp.py yet — a single run to completion).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env_mn5.sh

CONFIG="${1:-configs/esm8m_h100.yaml}"
OUT_DIR="$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output_dir'])")"
RUN_NAME="$(basename "$OUT_DIR")"

# #SBATCH --output/--error can't reference $RUN_NAME (unknown until now), so Slurm
# writes them to the submission dir first; relocate to $GLP_RUNS_ROOT/<run>/ (off
# $HOME) on exit, success or failure, with a symlink back under runs/<run>/.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  finalize_logs() {
    local base="${SLURM_JOB_NAME:-glp-train}-${SLURM_JOB_ID}"
    mkdir -p "$GLP_RUNS_ROOT/$RUN_NAME" "runs/$RUN_NAME"
    for ext in out err; do
      [ -f "$base.$ext" ] || continue
      mv "$base.$ext" "$GLP_RUNS_ROOT/$RUN_NAME/$base.$ext"
      ln -sf "$GLP_RUNS_ROOT/$RUN_NAME/$base.$ext" "runs/$RUN_NAME/$base.$ext" 2>/dev/null || true  # no-op if runs/ already resolves into $GLP_RUNS_ROOT (symlinked wholesale)
    done
  }
  trap finalize_logs EXIT
fi

python scripts/train_glp.py --config "$CONFIG"
