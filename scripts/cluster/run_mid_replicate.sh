#!/bin/bash
#SBATCH --job-name=glp-mid-replicate
#SBATCH --account=bsc70
#SBATCH --qos=acc_debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Replicate the canonical local_clean result (docs/results/run_summary.md, §10
# "Reproduce") on MN5 as runs/mid_mn5, to check it reproduces off the Mac before
# trusting a scale-up to esm8m_h100.yaml. Despite the file/run names (this predates
# the local_mid -> local_clean rename after the train/eval contamination fix,
# run_summary.md §7), the recipe below is the decontaminated one: build_swissprot_eval
# runs BEFORE extract_activations and --exclude-from is passed explicitly. This is the
# last run to ever run "local"-equivalent hyperparameters for comparison — everything
# after this is cluster-only (configs/mid_mn5.yaml).
#
# Submit from the repo root — the #SBATCH directives above request the GPU
# allocation automatically, no manual salloc needed:
#   sbatch scripts/cluster/run_mid_replicate.sh
# (~10M activations / 1 epoch is the same budget as the local run, which took
# well under an hour on an M4 — comfortably fits the 4h cap above. Monitor with
# `squeue -u $USER` / `tail -f glp-mid-replicate-<jobid>.out`.)
#
# Storage split is handled by each script itself via glp_plm.utils.resolve_storage_dir /
# resolve_shared_dir, not by this launcher:
#   - checkpoints, the activation store, and the embedded eval shards are all
#     data-sized and land directly under $GLP_RUNS_ROOT/<run>/checkpoints,
#     $GLP_DATA_ROOT/<run>/train_store, $GLP_DATA_ROOT/<run>/swissprot/processed —
#     NOT symlinked under runs/, reach them there directly
#   - metrics/figures are redirected to $GLP_RUNS_ROOT per-run and symlinked back
#     under runs/<run>/, so they stay browsable locally
#   - the raw Swiss-Prot TSV/FASTA are SHARED across all runs (same REST query ->
#     same result, regardless of output_dir) at $GLP_DATA_ROOT/swissprot/, not
#     nested per-run either; only the embedded processed/ eval shards above are per-run
# UniRef50 isn't consumed by anything yet (producer_consumer.py is a stub) — see
# scripts/prefetch_data.py --with-uniref50 if you want it prefetched anyway.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

source scripts/cluster/env_mn5.sh   # modules, venv, HF_HOME

CONFIG="${1:-configs/mid_mn5.yaml}"
N_SEQS="${2:-110000}"          # matches local_clean's reproduce recipe
N_PROT="${3:-20000}"
MIN_REQUIRED_INSTANCES="${4:-30}"

RUN_NAME="$(python -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))['output_dir'].rstrip('/').split('/')[-1])" "$CONFIG")"
echo "[run_mid_replicate] config=$CONFIG run=$RUN_NAME data_root=$GLP_DATA_ROOT runs_root=$GLP_RUNS_ROOT"

# #SBATCH --output/--error can't reference $RUN_NAME (unknown until now), so Slurm
# writes them to the submission dir first; relocate to $GLP_RUNS_ROOT/<run>/ (off
# $HOME) on exit, success or failure, with a symlink back under runs/<run>/ — same
# treatment as metrics/figures.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  finalize_logs() {
    local base="${SLURM_JOB_NAME:-glp-mid-replicate}-${SLURM_JOB_ID}"
    mkdir -p "$GLP_RUNS_ROOT/$RUN_NAME" "runs/$RUN_NAME"
    for ext in out err; do
      [ -f "$base.$ext" ] || continue
      mv "$base.$ext" "$GLP_RUNS_ROOT/$RUN_NAME/$base.$ext"
      ln -sf "$GLP_RUNS_ROOT/$RUN_NAME/$base.$ext" "runs/$RUN_NAME/$base.$ext" 2>/dev/null || true  # no-op if runs/ already resolves into $GLP_RUNS_ROOT (symlinked wholesale)
    done
  }
  trap finalize_logs EXIT
fi

EVAL_TSV="$GLP_DATA_ROOT/swissprot/swissprot_annotated_eval_${N_PROT}.tsv"
EVAL_DIR="$GLP_DATA_ROOT/$RUN_NAME/swissprot/processed"
python scripts/build_swissprot_eval.py --config "$CONFIG" --n-prot "$N_PROT" --min-required-instances "$MIN_REQUIRED_INSTANCES"
python scripts/extract_activations.py  --config "$CONFIG" --store train --n-seqs "$N_SEQS" --exclude-from "$EVAL_TSV"
python scripts/train_glp.py            --config "$CONFIG"
python scripts/eval_generation.py      --config "$CONFIG" --fd-steps 20,50,100,300,1000
python scripts/freeze_concepts.py      --config "$CONFIG" --eval-dir "$EVAL_DIR"
python scripts/run_probing.py          --config "$CONFIG" \
    --eval-dir "$EVAL_DIR" \
    --concepts "$EVAL_DIR/frozen_concepts.json" \
    --n-shards-per-split 1
python scripts/make_report.py          --config "$CONFIG"

echo "[run_mid_replicate] done — compare runs/$RUN_NAME/metrics/ against docs/results/run_summary.md"
