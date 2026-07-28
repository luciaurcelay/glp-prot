#!/usr/bin/env bash
# Chain the full cluster pipeline as dependent SLURM jobs (PLAN §11 cluster steps).
# This is a plain script you *run* on a login node (not itself an sbatch script) —
# it calls `sbatch` for each stage with --dependency=afterok on the previous job.
#
# Usage (from the repo root):
#   scripts/cluster/submit_pipeline.sh [config] [n_seqs_for_extract] [n_prot_for_eval]
#
# Prerequisite one-time steps (not chained here — run once, on a login node):
#   scripts/cluster/setup_venv.sh
#   scripts/cluster/prefetch_weights.sh
#
# Stages: extract_activations -> build_swissprot_eval -> train_glp -> eval_generation
#         -> freeze_concepts -> run_probing (auc) -> make_report
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CONFIG="${1:-configs/esm8m_h100.yaml}"
N_SEQS="${2:-200000}"
N_PROT="${3:-20000}"

submit() {
  # $1 = script, remaining = extra positional args; prints just the numeric job id
  local dep_arg=()
  if [ -n "${PREV_JOB:-}" ]; then dep_arg=(--dependency="afterok:$PREV_JOB"); fi
  sbatch --parsable "${dep_arg[@]}" "$@"
}

echo "[pipeline] config=$CONFIG n_seqs=$N_SEQS n_prot=$N_PROT"

PREV_JOB=$(submit scripts/cluster/submit_extract_activations.sh "$CONFIG" "$N_SEQS")
echo "[pipeline] extract_activations       -> job $PREV_JOB"

PREV_JOB=$(submit scripts/cluster/submit_build_swissprot_eval.sh "$CONFIG" "$N_PROT")
echo "[pipeline] build_swissprot_eval      -> job $PREV_JOB"

PREV_JOB=$(submit scripts/cluster/submit_train_glp.sh "$CONFIG")
echo "[pipeline] train_glp                 -> job $PREV_JOB"

PREV_JOB=$(submit scripts/cluster/submit_eval_generation.sh "$CONFIG")
echo "[pipeline] eval_generation (Gate G2) -> job $PREV_JOB"

PREV_JOB=$(submit scripts/cluster/submit_freeze_concepts.sh "$CONFIG")
echo "[pipeline] freeze_concepts           -> job $PREV_JOB"

PREV_JOB=$(submit scripts/cluster/submit_run_probing.sh "$CONFIG" auc)
echo "[pipeline] run_probing (auc)         -> job $PREV_JOB"

PREV_JOB=$(submit scripts/cluster/submit_make_report.sh "$CONFIG")
echo "[pipeline] make_report               -> job $PREV_JOB"

echo "[pipeline] all stages queued; monitor with: squeue -u \$USER"
