#!/bin/bash
#SBATCH --job-name=glp-novelty
#SBATCH --account=bsc70
#SBATCH --qos=acc_bsccs
#SBATCH --partition=acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Subspace novelty follow-up (scripts/subspace_novelty.py) to subspace_analysis.py:
# three pre-registered checks on G_perp (GLP's meta-neuron activation minus its
# SAE-dictionary-explained part) —
#   1. PCA on the residual itself (no labels used to pick components), then check
#      whether the top components correlate with any of the 94 falsification-check
#      concepts (AUC) or the 10 continuous properties (Spearman rho) — a component
#      matching neither is an "uncatalogued structure" candidate.
#   2. refit the ridge decomposition against the raw (un-standardized) ESM layer R
#      and against the SAE+R concatenation, so the "joint" residual is explained by
#      NEITHER the SAE dictionary NOR the layer it reconstructs — the part of GLP
#      genuinely not inherited from ESM.
#   3. per-GLP-block variance-fraction of the joint-orthogonal residual, checked for
#      a depth trend (does GLP build this information across the diffusion blocks,
#      or carry it from the input?).
# acc_bsccs/3h: three ridge fits (SAE-only p=10240, raw-only p=320, joint
# p=10560) at one noise level, vs. subspace_analysis.py's two ridge fits (SAE-only)
# at two noise levels — roughly comparable total cost. Not resumable (no
# partial-checkpoint loop like subspace_analysis.py's per-concept probe).
#
# Run AFTER subspace_analysis.py (reuses its probing.json/probing_sae_topk.json
# for the 94-concept list).
#
# Submit from the repo root:
#   sbatch scripts/cluster/submit_subspace_novelty.sh
#   sbatch scripts/cluster/submit_subspace_novelty.sh \
#       runs/uniref50_500m_4096_d24_steering/config.yaml \
#       runs/uniref50_500m_4096_d24_steering \
#       "" /gpfs/scratch/bsc70/bsc070332/glp-prot-data/uniref50_500m/swissprot/processed \
#       runs/uniref50_500m_4096_d24/checkpoints/glp_final.pt \
#       runs/uniref50_500m_4096_d24/checkpoints/rep_statistics.pt \
#       20000 20000
#     Args 5-6 (ckpt/scaler) matter for *_steering run dirs (own checkpoints/ is
#     empty). Args 7-8 (n-train/n-test) — shrink for a quick smoke test, e.g. 2000 2000.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env_mn5.sh

CONFIG="${1:-configs/esm8m_h100.yaml}"
OUT_DIR="${2:-}"
N_SHARDS_PER_SPLIT="${3:-}"
EVAL_DIR="${4:-}"
CKPT="${5:-}"
SCALER="${6:-}"
N_TRAIN="${7:-20000}"
N_TEST="${8:-20000}"

RUN_OUT_DIR="$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output_dir'])")"
RUN_NAME="$(basename "$RUN_OUT_DIR")"
record_launch "$RUN_NAME" "$@"

ARGS=(--config "$CONFIG" --n-train "$N_TRAIN" --n-test "$N_TEST")
[ -n "$OUT_DIR" ] && ARGS+=(--out-dir "$OUT_DIR")
[ -n "$N_SHARDS_PER_SPLIT" ] && ARGS+=(--n-shards-per-split "$N_SHARDS_PER_SPLIT")
[ -n "$EVAL_DIR" ] && ARGS+=(--eval-dir "$EVAL_DIR")
[ -n "$CKPT" ] && ARGS+=(--ckpt "$CKPT")
[ -n "$SCALER" ] && ARGS+=(--scaler "$SCALER")

python scripts/subspace_novelty.py "${ARGS[@]}"
