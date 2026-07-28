#!/bin/bash
#SBATCH --job-name=glp-train
#SBATCH --account=bsc70
#SBATCH --qos=acc_bsccs
#SBATCH --partition=acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
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

python scripts/train_glp.py --config "$CONFIG"
