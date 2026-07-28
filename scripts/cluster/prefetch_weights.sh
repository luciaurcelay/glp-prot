#!/usr/bin/env bash
# One-time (or re-run as needed) weight prefetch. Run on a LOGIN node — it needs
# internet access. Downloads only small, released weights:
#   - ESM-2-8M (facebook/esm2_t6_8M_UR50D)
#   - InterPLM's released SAEs for that model (layers 1-6, ~10k-dict each)
# Both land in the HF cache under $GLP_RUNS_ROOT/hf_cache (env_mn5.sh) — durable
# project storage, not scratch, since these are small but annoying to redo if
# scratch gets purged. Raw training/eval *data* (Swiss-Prot, UniRef50) is a
# separate step: scripts/prefetch_data.py.
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

source scripts/cluster/env_mn5.sh
export HF_HUB_OFFLINE=0   # this script is the one place that's allowed to hit the network

CONFIG="${1:-configs/esm8m_h100.yaml}"

python scripts/download_interplm_saes.py --config "$CONFIG" --all-layers

python - "$CONFIG" <<'PY'
import sys
# Unlike `python scripts/x.py` (where Python auto-adds the script's own dir to
# sys.path[0]), this runs as a heredoc piped into `python -`, so _bootstrap.py
# (in scripts/, not the repo root cwd) has to be found explicitly.
sys.path.insert(0, "scripts")
import _bootstrap  # noqa: F401
from glp_plm.config import load_config
from transformers import EsmModel, EsmTokenizer

cfg = load_config(sys.argv[1])
EsmModel.from_pretrained(cfg.data.model_name)
EsmTokenizer.from_pretrained(cfg.data.model_name)
print(f"[prefetch_weights] cached {cfg.data.model_name}")
PY

echo "[prefetch_weights] done. HF_HOME=$HF_HOME"
