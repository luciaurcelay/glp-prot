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

# Group this attempt's logs under runs/<run>/launches/<job_id>/, append the per-run
# launch ledger, and relocate Slurm's .out/.err there on exit (env_mn5.sh).
record_launch "$RUN_NAME" "$@"

python scripts/make_report.py --config "$CONFIG"
