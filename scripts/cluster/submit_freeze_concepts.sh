#!/bin/bash
#SBATCH --job-name=glp-freeze-concepts
#SBATCH --account=bsc70
#SBATCH --qos=gp_bsccs
#SBATCH --partition=gpp
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
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
EVAL_DIR="$OUT_DIR/swissprot/processed"

python scripts/freeze_concepts.py --config "$CONFIG" --eval-dir "$EVAL_DIR"
