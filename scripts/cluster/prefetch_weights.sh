#!/usr/bin/env bash
# One-time (or re-run as needed) weight prefetch. Run on a LOGIN node — it needs
# internet access. Downloads only small, released weights:
#   - ESM-2-8M (facebook/esm2_t6_8M_UR50D)
#   - InterPLM's released SAEs for that model (layers 1-6, ~10k-dict each)
#
# Does NOT download UniRef50 (~12 GB compressed) — per CLAUDE.md, that needs an
# explicit go-ahead first, and extract_activations.py doesn't consume it yet
# anyway (see scripts/cluster/README.md).
#
# Jobs run with HF_HUB_OFFLINE=1 (env_mn5.sh) so they never need compute-node
# internet — this script is what populates the cache they read from.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [ -f scripts/cluster/cluster.env ]; then
  # shellcheck disable=SC1091
  source scripts/cluster/cluster.env
fi
: "${GLP_DATA_ROOT:?GLP_DATA_ROOT is not set. Copy scripts/cluster/cluster.env.example to scripts/cluster/cluster.env and fill it in first.}"

module purge
module load python/3.11.5-gcc
module load cuda/12.4
VENV="$GLP_DATA_ROOT/venv"
[ -x "$VENV/bin/python" ] || { echo "Run scripts/cluster/setup_venv.sh first." >&2; exit 1; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/InterPLM:${PYTHONPATH:-}"
export HF_HOME="$GLP_DATA_ROOT/hf_cache"
mkdir -p "$HF_HOME"
export HF_HUB_OFFLINE=0   # this script is the one place that's allowed to hit the network

CONFIG="${1:-configs/esm8m_h100.yaml}"

python scripts/download_interplm_saes.py --config "$CONFIG" --all-layers

python - "$CONFIG" <<'PY'
import sys
sys.path.insert(0, ".")
import _bootstrap  # noqa: F401
from glp_plm.config import load_config
from transformers import EsmModel, EsmTokenizer

cfg = load_config(sys.argv[1])
EsmModel.from_pretrained(cfg.data.model_name)
EsmTokenizer.from_pretrained(cfg.data.model_name)
print(f"[prefetch_weights] cached {cfg.data.model_name}")
PY

echo "[prefetch_weights] done. HF_HOME=$HF_HOME"
