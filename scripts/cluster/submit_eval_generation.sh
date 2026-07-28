#!/bin/bash
#SBATCH --job-name=glp-eval-gen
#SBATCH --account=bsc70
#SBATCH --qos=acc_bsccs
#SBATCH --partition=acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
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

python scripts/eval_generation.py --config "$CONFIG" \
  --n-samples "$N_SAMPLES" --fd-steps 20,50,100,300,1000
