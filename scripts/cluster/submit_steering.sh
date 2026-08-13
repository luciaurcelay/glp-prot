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
# Exp 4.1 steering (PLAN §11/W4, glprot_steering_spec.md): off-manifold diagnostic +
# masked-Gibbs SAE-clamp generation + Pareto. One job sweeps the WHOLE --alphas list
# (and both naive/corrected variants) internally — do not submit one job per alpha.
#
# acc_debug's 2h cap is a smoke-test default (Reserve acc_debug for small --n-sequences,
# per scripts/cluster/README.md); override qos/time for a real sweep, e.g.:
#   sbatch --qos=acc_bsccs --time=1-00:00:00 scripts/cluster/submit_steering.sh \
#     configs/configs_lucia/esm8m_500m_4096_d24.yaml 0.5,1,2,4,8 50
#
# If `config.output_dir` is a run you don't own on $GLP_RUNS_ROOT (e.g. checkpoints
# produced by a teammate), point output_dir at a directory YOU own instead (so
# record_launch below and RunLogger's writes succeed) and pass --ckpt/--scaler
# (and --eval-dir, if that run has no probing eval store) as EXTRA_ARGS pointing at
# the original run's files — group members can usually read those even without
# write access. See scripts/cluster/README.md and ask the run's owner for a
# `setfacl -R -m u:<you>:rwx` grant if you need real write access instead.
#
# Submit from the repo root:  sbatch scripts/cluster/submit_steering.sh
#   sbatch scripts/cluster/submit_steering.sh <config> <alphas> <n_sequences> [extra run_steering.py args...]

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env_mn5.sh

CONFIG="${1:-configs/configs_lucia/esm8m_500m_4096_d24.yaml}"
ALPHAS="${2:-0.5,1,2,4,8}"
N_SEQUENCES="${3:-50}"
OUT_DIR="$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output_dir'])")"
RUN_NAME="$(basename "$OUT_DIR")"
shift $(( $# < 3 ? $# : 3 ))   # remaining "$@" (if any) forwarded verbatim below

# Group this attempt's logs under runs/<run>/launches/<job_id>/, append the per-run
# launch ledger, and relocate Slurm's .out/.err there on exit (env_mn5.sh).
record_launch "$RUN_NAME" "$CONFIG" "$ALPHAS" "$N_SEQUENCES" "$@"

python scripts/run_steering.py --config "$CONFIG" \
  --alphas "$ALPHAS" --n-sequences "$N_SEQUENCES" "$@"
