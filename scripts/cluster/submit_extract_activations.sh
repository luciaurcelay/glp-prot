#!/bin/bash
#SBATCH --job-name=glp-extract
#SBATCH --account=bsc70
#SBATCH --qos=acc_bsccs
#SBATCH --partition=acc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Build the unlabeled GLP training activation store (PLAN §11 / SETUP.md).
# Submit from the repo root:  sbatch scripts/cluster/submit_extract_activations.sh
# Override config/qos/time at submit time, e.g.:
#   sbatch --qos=acc_debug --time=00:30:00 scripts/cluster/submit_extract_activations.sh configs/poc_mac.yaml
#
# Data source is config-only: set `data.train_fasta: uniref50` in $CONFIG to sample
# from the local UniRef50 pull instead of Swiss-Prot REST (glp_plm/config.py).
# ⚠ Still `store_mode: precache`, not true streaming (glp_plm/data/producer_consumer.py
# stub) — fine at 50-300M acts on MN5 scratch, but `store_mode: stream` is
# unimplemented and will raise if set.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env_mn5.sh

CONFIG="${1:-configs/esm8m_h100.yaml}"
N_SEQS="${2:-200000}"
N_PROT="${3:-20000}"          # must match whatever --n-prot build_swissprot_eval.py used
# 4th arg: the labeled eval TSV whose proteins must NOT be trained on, overriding the
# auto-detected shared path below. Leave empty ONLY if the training corpus is genuinely
# disjoint (e.g. a real UniRef50 stream). The default Swiss-Prot queries overlap the
# eval set ~98% — see docs/PLAN.md §10 risk register.
OUT_DIR="$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output_dir'])")"
RUN_NAME="$(basename "$OUT_DIR")"

# #SBATCH --output/--error can't reference $RUN_NAME (unknown until now), so Slurm
# writes them to the submission dir first; relocate to $GLP_RUNS_ROOT/<run>/ (off
# $HOME) on exit, success or failure, with a symlink back under runs/<run>/.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  finalize_logs() {
    local base="${SLURM_JOB_NAME:-glp-extract}-${SLURM_JOB_ID}"
    mkdir -p "$GLP_RUNS_ROOT/$RUN_NAME" "runs/$RUN_NAME"
    for ext in out err; do
      [ -f "$base.$ext" ] || continue
      mv "$base.$ext" "$GLP_RUNS_ROOT/$RUN_NAME/$base.$ext"
      ln -sf "$GLP_RUNS_ROOT/$RUN_NAME/$base.$ext" "runs/$RUN_NAME/$base.$ext" 2>/dev/null || true  # no-op if runs/ already resolves into $GLP_RUNS_ROOT (symlinked wholesale)
    done
  }
  trap finalize_logs EXIT
fi

EVAL_TSV="${4:-}"
if [[ -z "$EVAL_TSV" ]]; then
  # Shared, not per-run (glp_plm.utils.resolve_shared_dir) — same file regardless of
  # which run's output_dir this is, keyed by n_prot so different eval-slice sizes
  # don't collide.
  CAND="$GLP_DATA_ROOT/swissprot/swissprot_annotated_eval_${N_PROT}.tsv"
  [[ -f "$CAND" ]] && EVAL_TSV="$CAND"
fi

EXCLUDE_ARGS=()
if [[ -n "$EVAL_TSV" ]]; then
  echo "[submit] excluding eval proteins listed in $EVAL_TSV"
  EXCLUDE_ARGS=(--exclude-from "$EVAL_TSV")
else
  echo "[submit] WARNING: no eval TSV found — training may overlap the eval set." >&2
  echo "[submit]          build the eval set first, or pass it as the 3rd argument." >&2
fi

python scripts/extract_activations.py --config "$CONFIG" --store train \
  --n-seqs "$N_SEQS" --embed-batch-size 64 "${EXCLUDE_ARGS[@]}"
