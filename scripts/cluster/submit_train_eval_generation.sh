#!/bin/bash
#SBATCH --job-name=glp-train-eval-gen
#SBATCH --account=bsc70
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# train_glp + eval_generation as ONE job (acc_debug rejects a second queued/running job
# from the same user, so the two-launcher --dependency=afterok chain doesn't work under it).
# Submit from the repo root:  sbatch scripts/cluster/submit_train_eval_generation.sh
#   sbatch scripts/cluster/submit_train_eval_generation.sh configs/esm8m_h100.yaml 50000

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env_mn5.sh

CONFIG="${1:-configs/esm8m_h100.yaml}"
N_SAMPLES="${2:-50000}"
OUT_DIR="$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output_dir'])")"
RUN_NAME="$(basename "$OUT_DIR")"

record_launch "$RUN_NAME" "$@"

echo "[submit_train_eval_generation] 1/2 train_glp"
python scripts/train_glp.py --config "$CONFIG"

echo "[submit_train_eval_generation] 2/2 eval_generation"
python scripts/eval_generation.py --config "$CONFIG" \
  --n-samples "$N_SAMPLES" --fd-steps 20,50,100,300,1000
