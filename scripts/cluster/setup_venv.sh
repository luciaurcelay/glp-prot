#!/usr/bin/env bash
# One-time venv setup for MN5 (BSC). Run this yourself on a LOGIN node — it needs
# internet access to pip-install, which compute nodes are not assumed to have.
#
#   scripts/cluster/setup_venv.sh
#
# Builds the same runtime deps as SETUP.md (Mac PoC), on Python 3.11 (InterPLM's
# requirement) with a CUDA build of torch for the H100 (acc partition) nodes.
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
mkdir -p "$GLP_DATA_ROOT"
python -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip

# cu124 wheels are forward-compatible with the newer driver/toolkit modules also
# present (12.6/12.8); switch the index if a given torch release drops cu124.
pip install --index-url https://download.pytorch.org/whl/cu124 "torch>=2.8"
pip install "numpy>=2.2,<2.3" scipy pandas scikit-learn \
  "transformers>=4.57" biopython "huggingface-hub>=0.28" h5py einops \
  pyyaml tqdm typed-argument-parser matplotlib pytest wandb

python - <<'PY'
import torch
print(f"torch {torch.__version__}  cuda available: {torch.cuda.is_available()}")
PY
echo "[setup_venv] (cuda available is expected to print False here — login nodes have no GPU;" \
     "it should print True inside an acc-partition job)"

echo "[setup_venv] venv ready at $VENV"
echo "[setup_venv] next: scripts/cluster/prefetch_weights.sh, then sbatch scripts/cluster/submit_*.sh"
