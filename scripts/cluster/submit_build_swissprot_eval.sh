#!/bin/bash
#SBATCH --job-name=glp-swissprot-eval
#SBATCH --account=bsc70
#SBATCH --qos=acc_bsccs
#SBATCH --partition=acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Build the labeled Swiss-Prot eval store (PLAN §11 / SETUP.md W3).
# Submit from the repo root:  sbatch scripts/cluster/submit_build_swissprot_eval.sh
#   sbatch scripts/cluster/submit_build_swissprot_eval.sh configs/esm8m_h100.yaml 20000 30

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env_mn5.sh

CONFIG="${1:-configs/esm8m_h100.yaml}"
N_PROT="${2:-20000}"
MIN_REQUIRED_INSTANCES="${3:-30}"

python scripts/build_swissprot_eval.py --config "$CONFIG" \
  --n-prot "$N_PROT" --min-required-instances "$MIN_REQUIRED_INSTANCES"
