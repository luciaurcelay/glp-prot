#!/usr/bin/env bash
# Common environment for GLP-PLM jobs on MN5 (BSC). Sourced by every
# submit_*.sh (and safe to `source` interactively on a login node too).
#
# Responsibilities:
#   - load the module stack (Python 3.11 + CUDA; InterPLM needs 3.11 — SETUP.md)
#   - activate the venv living on cluster storage (built by setup_venv.sh)
#   - point HF_HOME + runs/ at cluster storage instead of the quota-capped $HOME
#   - put the repo root + InterPLM/ on PYTHONPATH (belt-and-suspenders on top of
#     scripts/_bootstrap.py, which every script already imports)
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

if [ -f scripts/cluster/cluster.env ]; then
  # shellcheck disable=SC1091
  source scripts/cluster/cluster.env
fi
: "${GLP_DATA_ROOT:?GLP_DATA_ROOT is not set. Copy scripts/cluster/cluster.env.example to scripts/cluster/cluster.env and fill it in.}"

module purge
module load python/3.11.5-gcc
module load cuda/12.4

VENV="$GLP_DATA_ROOT/venv"
if [ ! -x "$VENV/bin/python" ]; then
  echo "No venv at $VENV — run scripts/cluster/setup_venv.sh once first (needs internet; run on a login node, not inside a job)." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/InterPLM:${PYTHONPATH:-}"

export HF_HOME="$GLP_DATA_ROOT/hf_cache"
mkdir -p "$HF_HOME"
# Compute nodes on MN5 are not assumed to have internet; weights/SAEs should be
# fetched once from a login node via prefetch_weights.sh, then jobs run offline.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

mkdir -p "$GLP_DATA_ROOT/runs"
if [ -L runs ]; then
  : # already a symlink; leave it (may point elsewhere on purpose)
elif [ -e runs ]; then
  echo "WARNING: $REPO_ROOT/runs already exists and is not a symlink — leaving it alone." \
       "Outputs will land in cluster storage under \$GLP_DATA_ROOT/runs only if you" \
       "point --config output_dir there yourself, since the local runs/ is taking precedence." >&2
else
  ln -s "$GLP_DATA_ROOT/runs" runs
fi

echo "[env_mn5] repo=$REPO_ROOT  data_root=$GLP_DATA_ROOT  python=$(command -v python)  device_check=$(python -c 'import torch;print(torch.cuda.is_available())' 2>/dev/null || echo 'n/a')"
