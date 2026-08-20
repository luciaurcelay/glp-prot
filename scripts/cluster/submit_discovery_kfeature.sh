#!/bin/bash
#SBATCH --job-name=glp-discovery-kfeature
#SBATCH --account=bsc70
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Discovery section Part 1 (scripts/discovery_kfeature.py): assembles
# discovery_kfeature.csv (k=1 from probing.json + k=2,5,10,20 from
# probe_sae_topk.py's probing_sae_topk.json, all already computed — pure assembly,
# instant) and, for the R^2<threshold candidates, runs the residual probe
# (x - sae.decode(sae.encode(x)), same protocol as probing_1d's raw_layer/raw_mlp) —
# same order of cost as probe_sae_topk.py's own top-k pass, acc_debug's 30min is ample.
#
# Run AFTER run_probing.py (--metric auc), test_nonlinearity.py, and
# scripts/probe_sae_topk.py — needs probing.json, falsification.json, and
# probing_sae_topk.json all present.
#
# Submit from the repo root:
#   sbatch scripts/cluster/submit_discovery_kfeature.sh
#   sbatch scripts/cluster/submit_discovery_kfeature.sh \
#       runs/uniref50_500m_4096_d24_steering/config.yaml 0.6 \
#       runs/uniref50_500m_4096_d24_steering \
#       "" /gpfs/scratch/bsc70/bsc070332/glp-prot-data/uniref50_500m/swissprot/processed
#     — out-dir/eval-dir default to the config's own <output_dir>/{metrics,
#       swissprot/processed}; override the same way submit_probe_sae_topk.sh's
#       README notes do for runs that don't follow that default (as
#       uniref50_500m_4096_d24_steering doesn't for eval-dir).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env_mn5.sh

CONFIG="${1:-configs/esm8m_h100.yaml}"
R2_THRESHOLD="${2:-0.6}"
OUT_DIR="${3:-}"
N_SHARDS_PER_SPLIT="${4:-}"
EVAL_DIR="${5:-}"

RUN_OUT_DIR="$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output_dir'])")"
RUN_NAME="$(basename "$RUN_OUT_DIR")"
record_launch "$RUN_NAME" "$@"

ARGS=(--config "$CONFIG" --r2-threshold "$R2_THRESHOLD")
[ -n "$OUT_DIR" ] && ARGS+=(--out-dir "$OUT_DIR")
[ -n "$N_SHARDS_PER_SPLIT" ] && ARGS+=(--n-shards-per-split "$N_SHARDS_PER_SPLIT")
[ -n "$EVAL_DIR" ] && ARGS+=(--eval-dir "$EVAL_DIR")

python scripts/discovery_kfeature.py "${ARGS[@]}"
