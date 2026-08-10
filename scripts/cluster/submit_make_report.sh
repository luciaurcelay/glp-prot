#!/bin/bash
#SBATCH --job-name=glp-report
#SBATCH --account=bsc70
#SBATCH --qos=gp_bsccs
#SBATCH --partition=gpp
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Re-render figures from a run's metric artifacts (PLAN §11). Quick/CPU-only —
# fine to just run directly too: `source scripts/cluster/env_mn5.sh && python
# scripts/make_report.py --config configs/esm8m_h100.yaml` on a login node.
# Submit from the repo root:  sbatch scripts/cluster/submit_make_report.sh

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
    local base="${SLURM_JOB_NAME:-glp-report}-${SLURM_JOB_ID}"
    mkdir -p "$GLP_RUNS_ROOT/$RUN_NAME" "runs/$RUN_NAME"
    for ext in out err; do
      [ -f "$base.$ext" ] || continue
      mv "$base.$ext" "$GLP_RUNS_ROOT/$RUN_NAME/$base.$ext"
      ln -sf "$GLP_RUNS_ROOT/$RUN_NAME/$base.$ext" "runs/$RUN_NAME/$base.$ext" 2>/dev/null || true  # no-op if runs/ already resolves into $GLP_RUNS_ROOT (symlinked wholesale)
    done
  }
  trap finalize_logs EXIT
fi

# Record the SLURM allocation (time limit / QOS / nodes) + submit args (see env_mn5.sh).
write_launch_manifest "$RUN_NAME" "$@"

python scripts/make_report.py --config "$CONFIG"
