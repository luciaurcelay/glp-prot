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

# --- Per-launch bookkeeping --------------------------------------------------
# A run can be launched more than once — a timeout/crash is resubmitted and
# train_glp.py auto-resumes from its checkpoint (so the attempts are ONE run, not
# several — see scripts/cluster/README.md). To keep those attempts legible without
# fragmenting the run's shared state (checkpoints/metrics/train_store stay per-run),
# each attempt's ephemera is grouped under runs/<run>/launches/<job_id>/ and one row
# per attempt is appended to a runs/<run>/launches.jsonl ledger:
#
#   runs/<run>/
#     launches.jsonl                 # one JSON row per launch (the attempt ledger)
#     launches/<job_id>/
#       launch.txt                   # human-readable: submit args + scontrol allocation
#       stdout.log  stderr.log       # Slurm's job output, relocated here on exit
#     checkpoints/  metrics/  figures/   # PER-RUN, shared across attempts (unchanged)
#
# `record_launch "$RUN_NAME" "$@"` (called once per submit script, after RUN_NAME is
# resolved) does all of this and installs the exit-time log relocation. No-op outside
# SLURM (local dev). It supersedes the old inline finalize_logs blocks + the separate
# write_launch_manifest.
_glp_relocate_logs() {
  # EXIT trap: move Slurm's .out/.err (written to the submit dir, since #SBATCH can't
  # reference the run name) into this attempt's launches/<job_id>/ folder. Each mv is
  # independently guarded so a failure on one can't strand the other under `set -e`
  # (the bug that used to leave .err in $HOME — see git history).
  [ -n "${_GLP_LAUNCH_DIR:-}" ] || return 0
  mkdir -p "$_GLP_LAUNCH_DIR"
  { [ -f "${_GLP_JOB_BASE}.out" ] && mv "${_GLP_JOB_BASE}.out" "$_GLP_LAUNCH_DIR/stdout.log"; } 2>/dev/null || true
  { [ -f "${_GLP_JOB_BASE}.err" ] && mv "${_GLP_JOB_BASE}.err" "$_GLP_LAUNCH_DIR/stderr.log"; } 2>/dev/null || true
}

record_launch() {
  [ -n "${SLURM_JOB_ID:-}" ] || return 0
  local run_name="$1"; shift
  local jid="$SLURM_JOB_ID"
  local rundir="$GLP_RUNS_ROOT/$run_name"
  local ldir="$rundir/launches/$jid"
  mkdir -p "$ldir"

  local ts git sc
  ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  git="$(git rev-parse --short HEAD 2>/dev/null || echo '?')"
  sc="$(scontrol show job "$jid" 2>/dev/null || true)"

  # Human-readable manifest (authoritative allocation from scontrol: TimeLimit, QOS,
  # Account, Partition, NodeList, Submit/StartTime).
  {
    echo "# Launch manifest for ${SLURM_JOB_NAME:-job}-$jid"
    echo "generated:   $ts"
    echo "job_id:      $jid"
    echo "job_name:    ${SLURM_JOB_NAME:-?}"
    echo "run_name:    $run_name"
    echo "submit_dir:  ${SLURM_SUBMIT_DIR:-$(pwd)}"
    echo "submit_args: $*"
    echo "git_commit:  $git"
    echo
    echo "## scontrol show job"
    if [ -n "$sc" ]; then echo "$sc"; else echo "(scontrol unavailable on this node)"; fi
  } > "$ldir/launch.txt"

  # Machine-readable ledger row (one per attempt). Python assembles the JSON so submit
  # args / node lists are escaped safely, and pulls the allocation fields out of
  # scontrol's key=value output.
  python - "$rundir/launches.jsonl" "$jid" "${SLURM_JOB_NAME:-}" "$run_name" "$ts" "$git" "$*" "$sc" <<'PY' || echo "[env_mn5] WARNING: could not append launch ledger row" >&2
import json, re, sys
ledger, jid, jname, run, ts, git, args, sc = sys.argv[1:9]
def field(key):
    m = re.search(rf'\b{key}=(\S+)', sc)
    return m.group(1) if m else None
row = {"job_id": jid, "job_name": jname, "run_name": run, "timestamp": ts,
       "git_commit": git, "submit_args": args, "time_limit": field("TimeLimit"),
       "qos": field("QOS"), "account": field("Account"),
       "partition": field("Partition"), "nodelist": field("NodeList")}
with open(ledger, "a") as fh:
    fh.write(json.dumps(row) + "\n")
PY

  # Relocate this attempt's Slurm logs into $ldir on exit (globals, so the EXIT trap
  # can read them). Not `local` — they must outlive this function.
  _GLP_LAUNCH_DIR="$ldir"
  _GLP_JOB_BASE="${SLURM_JOB_NAME:-glp-job}-$jid"
  trap _glp_relocate_logs EXIT
  echo "[env_mn5] launch $jid -> $ldir  (ledger: $rundir/launches.jsonl)"
}

echo "[env_mn5] repo=$REPO_ROOT  data_root=$GLP_DATA_ROOT  runs_root=$GLP_RUNS_ROOT  python=$(command -v python)  device_check=$(python -c 'import torch;print(torch.cuda.is_available())' 2>/dev/null || echo 'n/a')"
