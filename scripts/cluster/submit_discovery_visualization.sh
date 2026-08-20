#!/bin/bash
#SBATCH --job-name=glp-discovery-viz
#SBATCH --account=bsc70
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Discovery section Part 4 (scripts/discovery_visualization.py): t-SNE projection
# of winning-unit activation vectors (GLP/SAE, colored by Swiss-Prot category —
# illustrative only, per spec) plus the quantitative companion that IS evidence:
# KMeans/HDBSCAN clustering of the same vectors vs. category labels (ARI, purity).
# sklearn's built-in TSNE/HDBSCAN/KMeans — no umap-learn install (decided this
# session: t-SNE + sklearn native clustering, no new dependency).
#
# Run AFTER run_probing.py (--metric auc) — needs its probing.json.
#
# Submit from the repo root:
#   sbatch scripts/cluster/submit_discovery_visualization.sh \
#       runs/uniref50_500m_4096_d24_steering/config.yaml \
#       runs/uniref50_500m_4096_d24_steering \
#       "" /gpfs/scratch/bsc70/bsc070332/glp-prot-data/uniref50_500m/swissprot/processed \
#       runs/uniref50_500m_4096_d24/checkpoints/glp_final.pt \
#       runs/uniref50_500m_4096_d24/checkpoints/rep_statistics.pt
#     Args 5-6 (ckpt/scaler) matter for *_steering run dirs (own checkpoints/ is
#     empty — model lives under the un-suffixed sibling run).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env_mn5.sh

CONFIG="${1:-configs/esm8m_h100.yaml}"
OUT_DIR="${2:-}"
N_SHARDS_PER_SPLIT="${3:-}"
EVAL_DIR="${4:-}"
CKPT="${5:-}"
SCALER="${6:-}"

RUN_OUT_DIR="$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output_dir'])")"
RUN_NAME="$(basename "$RUN_OUT_DIR")"
record_launch "$RUN_NAME" "$@"

ARGS=(--config "$CONFIG")
[ -n "$OUT_DIR" ] && ARGS+=(--out-dir "$OUT_DIR")
[ -n "$N_SHARDS_PER_SPLIT" ] && ARGS+=(--n-shards-per-split "$N_SHARDS_PER_SPLIT")
[ -n "$EVAL_DIR" ] && ARGS+=(--eval-dir "$EVAL_DIR")
[ -n "$CKPT" ] && ARGS+=(--ckpt "$CKPT")
[ -n "$SCALER" ] && ARGS+=(--scaler "$SCALER")

python scripts/discovery_visualization.py "${ARGS[@]}"
