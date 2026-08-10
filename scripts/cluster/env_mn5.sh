#!/usr/bin/env bash
# Common environment for GLP-PLM jobs on MN5 (BSC). Sourced by every
# submit_*.sh (and safe to `source` interactively on a login node too).
#
# Responsibilities:
#   - load the module stack (Python 3.11 + CUDA; InterPLM needs 3.11 — SETUP.md)
#   - activate the venv (defaults to $REPO_ROOT/.venv, same layout as local dev;
#     override with GLP_VENV in cluster.env if yours lives elsewhere, e.g. off $HOME)
#   - point HF_HOME (pretrained ESM-2 + SAE weights — small, but not worth
#     re-downloading if scratch gets purged) at GLP_RUNS_ROOT (projects — small but
#     precious), instead of the quota-capped $HOME. Per-run storage (checkpoints,
#     activation stores, Swiss-Prot data) is redirected to GLP_RUNS_ROOT/GLP_DATA_ROOT
#     by glp_plm.utils.resolve_storage_dir, called from within each pipeline script
#     itself (not here) — see that function's docstring. runs/ stays a plain local
#     dir; only metrics/figures get symlinked into it — checkpoints, the activation
#     store, and the embedded Swiss-Prot eval shards are all data-sized and
#     deliberately don't (kept off the small $HOME/repo quota either way, but out of
#     the local runs/ tree — reachable directly via $GLP_RUNS_ROOT/<run>/checkpoints
#     or $GLP_DATA_ROOT/<run>/{train_store,swissprot/processed}).
#   - put the repo root + InterPLM/ on PYTHONPATH (belt-and-suspenders on top of
#     scripts/_bootstrap.py, which every script already imports)
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

if [ -f scripts/cluster/cluster.env ]; then
  # shellcheck disable=SC1091
  source scripts/cluster/cluster.env
fi
: "${GLP_DATA_ROOT:?GLP_DATA_ROOT is not set. Fill it in in scripts/cluster/cluster.env.}"
: "${GLP_RUNS_ROOT:?GLP_RUNS_ROOT is not set. Fill it in in scripts/cluster/cluster.env.}"

module purge
# /apps/ACC/modulefiles (python/3.11.5-gcc, cuda/12.4, ...) isn't on MODULEPATH by
# default for every account — some accounts' login shell profile adds it, others
# don't (found 2026-08-10: crg562661's `sbatch` failed with "unknown module
# python/3.11.5-gcc" even though the file exists on disk, because MODULEPATH only
# carried /apps/GPP/modulefiles/* — the job's environment is captured at submission
# from whatever the submitting login shell had). `module use` here makes this
# account-independent instead of relying on ambient profile state.
module use /apps/ACC/modulefiles/environment /apps/ACC/modulefiles/compilers \
  /apps/ACC/modulefiles/libs /apps/ACC/modulefiles/applications
# python/3.11.5-gcc's Lmod hierarchy doesn't declare its own prereqs (`module
# spider` claims it loads standalone; it doesn't) — the real chain, found by
# trial and error on 2026-07-28: intel -> impi -> mkl -> hdf5 -> python -> cuda.
# Skipping any of these leaves the venv's python unable to find libpython3.11.so.1.0.
module load intel impi mkl hdf5 python/3.11.5-gcc cuda/12.4

# The python module ALSO exports PYTHONHOME/PYTHONPATH pointing at its own shared
# site-packages (a pre-installed torch-adjacent stack: fsspec/mpmath/sympy/networkx/
# filelock/Jinja2, as legacy easy_install .egg dirs) — found by trial and error on
# 2026-07-28 chasing a phantom `ModuleNotFoundError: filelock` even after installing
# it. Left in place this silently breaks venv isolation two ways: PYTHONHOME set at
# all makes CPython ignore the venv's pyvenv.cfg outright, and appending the module's
# old PYTHONPATH (as this line used to do) puts its .egg dirs on sys.path — pip finds
# them and reports "already satisfied", but .egg installs only become importable via
# `site.addsitedir()`'s .pth processing, which a raw PYTHONPATH entry skips, so the
# import fails anyway. Clear both before activating so the venv is actually isolated.
unset PYTHONHOME PYTHONPATH

VENV="${GLP_VENV:-$REPO_ROOT/.venv}"
if [ ! -x "$VENV/bin/python" ]; then
  echo "No venv at $VENV — build one first (see scripts/cluster/README.md; needs internet, run on a login node, not inside a job)." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/InterPLM"

export HF_HOME="$GLP_RUNS_ROOT/hf_cache"
mkdir -p "$HF_HOME"
# Compute nodes on MN5 are not assumed to have internet; weights/SAEs should be
# fetched once from a login node via prefetch_weights.sh, then jobs run offline.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

mkdir -p "$GLP_RUNS_ROOT" "$GLP_DATA_ROOT" runs

# Record how a job was actually launched, next to its .out/.err. The SLURM logs
# don't capture the time limit / QOS / node allocation, so a run that hit its wall
# clock (e.g. uniref50_500m, killed at its --time cap mid-training) otherwise leaves
# no trace of what cap it hit or how it was submitted. `scontrol show job` is the
# authoritative source (reflects any `sbatch --qos=/--time=` overrides, not just the
# #SBATCH defaults). Written at job START so it survives a timeout/kill; no-op
# outside SLURM (local dev). Call once per submit script, after RUN_NAME is resolved:
#   write_launch_manifest "$RUN_NAME" "$@"
# The filename base matches finalize_logs' <job_name>-<job_id> so all three artifacts
# (.out/.err/.launch.txt) sort together per job.
write_launch_manifest() {
  [ -n "${SLURM_JOB_ID:-}" ] || return 0
  local run_name="$1"; shift
  local base="${SLURM_JOB_NAME:-glp-job}-${SLURM_JOB_ID}"
  local dest="$GLP_RUNS_ROOT/$run_name"
  mkdir -p "$dest" "runs/$run_name"
  local manifest="$dest/$base.launch.txt"
  {
    echo "# Launch manifest for $base"
    echo "generated:   $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "job_id:      $SLURM_JOB_ID"
    echo "job_name:    ${SLURM_JOB_NAME:-?}"
    echo "run_name:    $run_name"
    echo "submit_dir:  ${SLURM_SUBMIT_DIR:-$(pwd)}"
    echo "submit_args: $*"
    echo "git_commit:  $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
    echo
    echo "## scontrol show job (authoritative allocation: TimeLimit, QOS, Account, Partition, NodeList, Submit/StartTime)"
    scontrol show job "$SLURM_JOB_ID" 2>/dev/null || echo "(scontrol unavailable on this node)"
  } > "$manifest"
  # Mirror under runs/<run>/ for accounts where runs/ is a real dir; no-op (same
  # inode) where runs/ is symlinked wholesale into $GLP_RUNS_ROOT.
  ln -sf "$manifest" "runs/$run_name/$base.launch.txt" 2>/dev/null || true
  echo "[env_mn5] launch manifest -> $manifest"
}

echo "[env_mn5] repo=$REPO_ROOT  data_root=$GLP_DATA_ROOT  runs_root=$GLP_RUNS_ROOT  python=$(command -v python)  device_check=$(python -c 'import torch;print(torch.cuda.is_available())' 2>/dev/null || echo 'n/a')"
