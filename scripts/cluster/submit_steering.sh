#!/bin/bash
#SBATCH --job-name=glp-steering
#SBATCH --account=bsc70
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Steering (PLAN §11/W4, glprot_steering_spec.md v2): off-manifold diagnostic +
# masked-Gibbs generation + Pareto, for either experiment. One job sweeps the WHOLE
# --alphas list (and both naive/corrected variants) internally — do not submit one
# job per alpha. Exp A and Exp B are deliberately NOT mechanically identical
# (different init/masking/rounds/selection/folding — see run_steering.py's module
# docstring); each experiment's own defaults kick in when ALPHAS/N_SEQUENCES below
# are left blank ("").
#
# acc_debug's 2h cap is a smoke-test default only, and Exp A in particular blows
# through it fast: 100 rounds/sequence (full resample every round) + mandatory
# ESMFold (16-24GB GPU mem) is much heavier per-sequence than Exp B's 20-30-round,
# no-fold loop. Reserve acc_debug for small --n-sequences pilots on EITHER
# experiment; override qos/time for a real sweep, e.g.:
#   sbatch --qos=acc_bsccs --time=1-00:00:00 scripts/cluster/submit_steering.sh \
#     A zinc1 configs/configs_lucia/esm8m_500m_4096_d24.yaml "" 100
#   sbatch --qos=acc_bsccs --time=1-00:00:00 scripts/cluster/submit_steering.sh \
#     B hydro1 configs/configs_lucia/esm8m_500m_4096_d24.yaml
#
# Exp B's first run also builds runs/<run>/steering_diffmean_hydro/gravy_contrast.json
# (UniRef50 GRAVY contrast sets — scripts/build_gravy_contrast.py) if it doesn't
# already exist; that step runs inline inside this same job, before generation.
#
# If `config.output_dir` is a run you don't own on $GLP_RUNS_ROOT (e.g. checkpoints
# produced by a teammate), point output_dir at a directory YOU own instead (so
# record_launch below and RunLogger's writes succeed) and pass --ckpt/--scaler
# (and --eval-dir, if that run has no probing eval store) as EXTRA_ARGS pointing at
# the original run's files — group members can usually read those even without
# write access. See scripts/cluster/README.md and ask the run's owner for a
# `setfacl -R -m u:<you>:rwx` grant if you need real write access instead.
#
# Submit from the repo root:
#   sbatch scripts/cluster/submit_steering.sh <A|B> <NAME> [config] [alphas] [n_sequences] [extra run_steering.py args...]
# NAME is a REQUIRED unique tag for this run (out-dir leaf = steering_<exp>_<NAME>); reusing
# a NAME clobbers that run's metrics/figures. Leave alphas/n_sequences as "" for defaults.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env_mn5.sh

EXPERIMENT="${1:?experiment (A or B) is required, e.g.: sbatch scripts/cluster/submit_steering.sh A run1}"
NAME="${2:?a unique run NAME is required — the steering out-dir leaf becomes steering_<exp>_<NAME> and RunLogger keys the metrics/figures redirect on it, so a non-unique NAME clobbers another run. e.g.: sbatch scripts/cluster/submit_steering.sh B dani_hydro1}"
CONFIG="${3:-configs/configs_lucia/esm8m_500m_4096_d24.yaml}"
ALPHAS="${4:-}"
N_SEQUENCES="${5:-}"
OUT_DIR="$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output_dir'])")"
RUN_NAME="$(basename "$OUT_DIR")"
shift $(( $# < 5 ? $# : 5 ))   # remaining "$@" (if any) forwarded verbatim below

EXTRA_ARGS=("$@")
[ -n "$N_SEQUENCES" ] && EXTRA_ARGS=(--n-sequences "$N_SEQUENCES" "${EXTRA_ARGS[@]}")
[ -n "$ALPHAS" ] && EXTRA_ARGS=(--alphas "$ALPHAS" "${EXTRA_ARGS[@]}")

# Group this attempt's logs under runs/<run>/launches/<job_id>/, append the per-run
# launch ledger, and relocate Slurm's .out/.err there on exit (env_mn5.sh).
record_launch "$RUN_NAME" "$CONFIG" "$EXPERIMENT" "$ALPHAS" "$N_SEQUENCES" "$@"

python scripts/run_steering.py --config "$CONFIG" --experiment "$EXPERIMENT" \
  --exp-name "$NAME" "${EXTRA_ARGS[@]}"
