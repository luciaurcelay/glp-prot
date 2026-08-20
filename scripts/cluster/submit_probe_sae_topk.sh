#!/bin/bash
#SBATCH --job-name=glp-probe-sae-topk
#SBATCH --account=bsc70
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# SAE top-k probing follow-up (scripts/probe_sae_topk.py) — the R^2-insufficiency
# check after test_nonlinearity.py: for every concept where the winning meta-neuron's
# R^2 against the full SAE dictionary is below threshold, fit a JOINT SAE logistic
# regression over its top-k features (k=2,5,10,20) and see whether that reaches GLP's
# probing AUC. Cheap (a handful of concepts x 4 k's x one multi-feature LogisticRegressionCV
# each) — acc_debug's 30min is ample; bump --time only if the concept count grows a lot.
#
# Run AFTER run_probing.py (--metric auc) and test_nonlinearity.py — needs both
# probing.json (GLP/SAE-k1 baselines) and falsification.json (the R^2 filter).
#
# Submit from the repo root:
#   sbatch scripts/cluster/submit_probe_sae_topk.sh
#   sbatch scripts/cluster/submit_probe_sae_topk.sh configs/esm8m_h100.yaml 0.55 2,5,10,20 \
#       runs/uniref50_100m_4096_b12/metrics/probing.json \
#       runs/uniref50_100m_4096_b12/metrics/falsification.json \
#       runs/uniref50_100m_4096_b12
#     — probing-json/falsification-json/out-dir default to the config's own
#       <output_dir>/metrics (pre- run_probing.py's `_probing` sibling-dir split, if
#       any); override when the run you're checking put them elsewhere (as
#       uniref50_100m_4096_b12 currently does — both still live in its own metrics/,
#       not a sibling `_probing` dir).
#     Arg 8 (eval-dir) matters for the block-depth-suffixed configs (_b3/_b6/_b12/_b24):
#       they share ONE Swiss-Prot eval store under the un-suffixed base name, e.g.
#       $GLP_DATA_ROOT/uniref50_100m_4096/swissprot/processed — their own
#       <output_dir>/swissprot/processed does not exist. Pass it explicitly for those.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env_mn5.sh

CONFIG="${1:-configs/esm8m_h100.yaml}"
R2_THRESHOLD="${2:-0.55}"
KS="${3:-2,5,10,20}"
PROBING_JSON="${4:-}"
FALSIFICATION_JSON="${5:-}"
OUT_DIR="${6:-}"
N_SHARDS_PER_SPLIT="${7:-}"
EVAL_DIR="${8:-}"

RUN_OUT_DIR="$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output_dir'])")"
RUN_NAME="$(basename "$RUN_OUT_DIR")"
record_launch "$RUN_NAME" "$@"

ARGS=(--config "$CONFIG" --r2-threshold "$R2_THRESHOLD" --ks "$KS")
[ -n "$PROBING_JSON" ] && ARGS+=(--probing-json "$PROBING_JSON")
[ -n "$FALSIFICATION_JSON" ] && ARGS+=(--falsification-json "$FALSIFICATION_JSON")
[ -n "$OUT_DIR" ] && ARGS+=(--out-dir "$OUT_DIR")
[ -n "$N_SHARDS_PER_SPLIT" ] && ARGS+=(--n-shards-per-split "$N_SHARDS_PER_SPLIT")
[ -n "$EVAL_DIR" ] && ARGS+=(--eval-dir "$EVAL_DIR")

python scripts/probe_sae_topk.py "${ARGS[@]}"
