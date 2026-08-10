#!/bin/bash
#SBATCH --job-name=glp-eval-gen
#SBATCH --account=bsc70
#SBATCH --qos=acc_bsccs
#SBATCH --partition=acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Generation-quality eval: FD (+ FD-vs-steps sweep) and PCA overlap (PLAN §6.4, Gate G2).
# Submit from the repo root:  sbatch scripts/cluster/submit_eval_generation.sh
#   sbatch scripts/cluster/submit_eval_generation.sh configs/esm8m_h100.yaml 50000

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env_mn5.sh

CONFIG="${1:-configs/esm8m_h100.yaml}"
N_SAMPLES="${2:-50000}"
OUT_DIR="$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output_dir'])")"
RUN_NAME="$(basename "$OUT_DIR")"

# Group this attempt's logs under runs/<run>/launches/<job_id>/, append the per-run
# launch ledger, and relocate Slurm's .out/.err there on exit (env_mn5.sh).
record_launch "$RUN_NAME" "$@"

python scripts/eval_generation.py --config "$CONFIG" \
  --n-samples "$N_SAMPLES" --fd-steps 20,50,100,300,1000
