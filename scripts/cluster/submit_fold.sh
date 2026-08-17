#!/bin/bash
#SBATCH --job-name=glp-fold
#SBATCH --account=bsc70
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# ESMFold2 pLDDT/pTM for a finished steering run (scripts/fold_steering_outputs.py).
# Writes runs/<...>/<steering-out-dir>/fold_confidence.csv, resumable — rerun the same
# command to pick up where an interrupted pass stopped.
#
# This launcher does NOT source scripts/cluster/env_mn5.sh, on purpose. env_mn5.sh loads
# python/3.11.5-gcc for the project venv, but ESMFold2 lives in a separate Python 3.12 venv
# (it needs the `esm` package, which the 3.11 project venv does not have and cannot get:
# MN5 is offline). Loading the 3.11 module and then running the 3.12 interpreter fails with
# "libpython3.12.so.1.0: cannot open shared object file" — the venv's interpreter needs its
# OWN base module on LD_LIBRARY_PATH (venv312/pyvenv.cfg points at
# /apps/ACC/PYTHON/3.12.1/GCC). Hence the duplicated-looking module block below.
#
# Cost (measured on an H100, 145-residue sequences, --num-sampling-steps 20): ~60s model
# load + ~2.3s/sequence, i.e. ~12 min for a 300-sequence Exp B run. 20 GiB peak GPU. The
# README's default of 50 steps doubles that for a difference well below the between-condition
# differences (~0.01 pLDDT / ~0.003 pTM), so 20 is this script's default.
#
# Submit from the repo root:
#   sbatch scripts/cluster/submit_fold.sh <steering-out-dir> [extra fold_steering_outputs.py args]
# e.g. a cheap smoke pass first:
#   sbatch --account=ehpc394 scripts/cluster/submit_fold.sh \
#     runs/steering_crg/steering_diffmean_hydro_t07_full --limit-per-cell 1
# then the full pass (same command minus the limit; it resumes over the smoke rows):
#   sbatch --account=ehpc394 scripts/cluster/submit_fold.sh \
#     runs/steering_crg/steering_diffmean_hydro_t07_full

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

RUN_DIR="${1:?steering out-dir is required, e.g.: sbatch scripts/cluster/submit_fold.sh runs/steering_crg/steering_diffmean_hydro_t07_full}"
shift

ESMFOLD2_VENV="${ESMFOLD2_VENV:-/gpfs/projects/bsc70/bsc070332/containers/esmfold2/venv312}"

module purge
# same `module use` reason as env_mn5.sh: /apps/ACC/modulefiles isn't on a batch job's
# inherited MODULEPATH, so the ACC trees have to be added explicitly.
module use /apps/ACC/modulefiles/environment /apps/ACC/modulefiles/compilers \
  /apps/ACC/modulefiles/libs /apps/ACC/modulefiles/applications
module load intel impi mkl hdf5 python/3.12.1-gcc cuda/12.4
# the python module exports PYTHONHOME/PYTHONPATH at its own install; left set, PYTHONHOME
# makes CPython ignore the venv's pyvenv.cfg entirely (see env_mn5.sh's longer note).
unset PYTHONHOME PYTHONPATH

# ESMFold2 + its ESMC-6B language model are prefetched in the shared cache; MN5 is offline
# so HF must not attempt any network call.
export HF_HOME="${HF_HOME:-/gpfs/projects/bsc70/bsc070332/glp-prot-runs/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

echo "[submit_fold] run_dir=$RUN_DIR venv=$ESMFOLD2_VENV HF_HOME=$HF_HOME"
"$ESMFOLD2_VENV/bin/python" scripts/fold_steering_outputs.py --run-dir "$RUN_DIR" "$@"
