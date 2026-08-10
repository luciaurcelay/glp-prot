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

# Group this attempt's logs under runs/<run>/launches/<job_id>/, append the per-run
# launch ledger, and relocate Slurm's .out/.err there on exit (env_mn5.sh).
record_launch "$RUN_NAME" "$@"

python scripts/train_glp.py --config "$CONFIG"
