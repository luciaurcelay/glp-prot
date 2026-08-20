#!/bin/bash
#SBATCH --job-name=glp-discovery-hierarchy
#SBATCH --account=bsc70
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Discovery section Part 2 (scripts/discovery_hierarchy.py): does the latent space
# reflect Swiss-Prot family hierarchy (EF-hand 1-4, Ig-like subtypes, ...)? Pools
# winning units' activations to per-protein vectors on a TEST-split protein
# subsample (default 500) for GLP/SAE/raw_mlp, correlates every family-concept
# pair, and runs a self-pair-excluded permutation test. 45min: dominated by 3x GLP
# meta-neuron forward passes (one per u in the grid) over the protein subsample —
# same order of cost as a single probing concept sweep, not the full 104-concept one.
#
# DRAFT: configs/discovery_families.yaml is pending domain-expert review (see its
# header) — this job's output is provisional until that lands.
#
# Run AFTER run_probing.py (--metric auc) — needs its probing.json.
#
# Submit from the repo root:
#   sbatch scripts/cluster/submit_discovery_hierarchy.sh
#   sbatch scripts/cluster/submit_discovery_hierarchy.sh \
#       runs/uniref50_500m_4096_d24_steering/config.yaml \
#       configs/discovery_families.yaml \
#       runs/uniref50_500m_4096_d24_steering \
#       "" /gpfs/scratch/bsc70/bsc070332/glp-prot-data/uniref50_500m/swissprot/processed 500 \
#       runs/uniref50_500m_4096_d24/checkpoints/glp_final.pt \
#       runs/uniref50_500m_4096_d24/checkpoints/rep_statistics.pt
#     Args 7-8 (ckpt/scaler) matter for *_steering run dirs: their own checkpoints/
#     is empty (the trained model lives under the un-suffixed sibling run) —
#     same gotcha submit_probing_pipeline.sh's own header documents.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env_mn5.sh

CONFIG="${1:-configs/esm8m_h100.yaml}"
FAMILIES="${2:-configs/discovery_families.yaml}"
OUT_DIR="${3:-}"
N_SHARDS_PER_SPLIT="${4:-}"
EVAL_DIR="${5:-}"
MAX_PROTEINS="${6:-500}"
CKPT="${7:-}"
SCALER="${8:-}"

RUN_OUT_DIR="$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output_dir'])")"
RUN_NAME="$(basename "$RUN_OUT_DIR")"
record_launch "$RUN_NAME" "$@"

ARGS=(--config "$CONFIG" --families "$FAMILIES" --max-proteins "$MAX_PROTEINS")
[ -n "$OUT_DIR" ] && ARGS+=(--out-dir "$OUT_DIR")
[ -n "$N_SHARDS_PER_SPLIT" ] && ARGS+=(--n-shards-per-split "$N_SHARDS_PER_SPLIT")
[ -n "$EVAL_DIR" ] && ARGS+=(--eval-dir "$EVAL_DIR")
[ -n "$CKPT" ] && ARGS+=(--ckpt "$CKPT")
[ -n "$SCALER" ] && ARGS+=(--scaler "$SCALER")

python scripts/discovery_hierarchy.py "${ARGS[@]}"
