#!/bin/bash
#SBATCH --job-name=glp-subspace
#SBATCH --account=bsc70
#SBATCH --qos=acc_bsccs
#SBATCH --partition=acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Subspace analysis (scripts/subspace_analysis.py): what distinguishes the 15
# not-in-dictionary concepts from the 79 the joint SAE top-k probe recovers?
# Decomposes GLP meta-neuron activations G against the SAE dictionary's span
# (ridge, same regularization as the falsification R^2 check) and probes the
# orthogonal component G_perp for all 94 concepts. acc_bsccs/3h, not acc_debug's
# usual 30min-2h: fitting the ridge projection is a ~20000x10240 SVD per noise
# level (2 by default), reused across all 30,720 GLP units via
# RidgeCV(alpha_per_target=True) — the expensive, non-resumable part. Part 2's
# per-concept probing loop IS resumable (subspace_probing_partial.jsonl); Part 3
# (intrinsic dimension / CKA) is cheap and last, skip with --skip-descriptive if
# only the main hypothesis test is needed.
#
# Run AFTER run_probing.py, test_nonlinearity.py, and probe_sae_topk.py.
#
# Submit from the repo root:
#   sbatch scripts/cluster/submit_subspace_analysis.sh
#   sbatch scripts/cluster/submit_subspace_analysis.sh \
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

python scripts/subspace_analysis.py "${ARGS[@]}"
