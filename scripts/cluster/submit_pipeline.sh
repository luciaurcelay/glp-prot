#!/bin/bash
#SBATCH --job-name=glp-pipeline
#SBATCH --account=bsc70
#SBATCH --qos=acc_debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Run the full pipeline (build_swissprot_eval -> extract_activations -> train_glp ->
# eval_generation -> freeze_concepts -> run_probing -> make_report) as ONE SLURM job,
# sequential plain `python` calls — not seven `sbatch --dependency=afterok` submissions.
#
# Why: MN5 gets congested enough that the practical way to get a job to actually start
# is `acc_debug`, and `acc_debug`'s QOS rejects a second job from the same user while one
# is still queued/running — so the old fan-out-seven-jobs-with-dependencies design (each
# sitting in the queue from submission, not just once its dependency clears) doesn't work
# under that QOS. One job sidesteps it entirely. This replaces both the old
# multi-sbatch `submit_pipeline.sh` and the one-off `run_mid_replicate.sh` (retired,
# 2026-07-29 — its job, proving the launchers reproduce `local_clean` on MN5, is done;
# see `configs/esm8m_h100.yaml` header + `scripts/cluster/README.md` "Config unification").
#
# The individual `submit_*.sh` launchers are still there (per-stage sbatch, `acc_bsccs`
# by default) if you want to rerun just one stage without repeating the whole chain.
#
# Submit from the repo root:
#   sbatch scripts/cluster/submit_pipeline.sh [config] [n_seqs] [n_prot] \
#       [min_required_instances] [n_samples] [metric] [n_shards_per_split]
#
# `acc_debug` is time-capped (the #SBATCH --time above, 2h, matches what
# `run_mid_replicate.sh` used successfully at 10M activations) — that is very unlikely
# to be enough for a full 100M-activation run (extract + 1 epoch of training + FD/PCA +
# probing over ~100 concepts). For an actual 100M+ run, override QOS/time, e.g.:
#   sbatch --qos=acc_bsccs --time=1-00:00:00 scripts/cluster/submit_pipeline.sh \
#       configs/esm8m_h100.yaml 5000000 20000
# Reserve the `acc_debug` default for smoke tests at small `n_seqs`/`n_train_activations`.
#
# `build_swissprot_eval` must run before `extract_activations` (not just "in any
# order") — the latter's `--exclude-from` needs the former's eval TSV to keep
# training/eval disjoint (PLAN §9/§10: an earlier run got this backwards and trained on
# 98.4% of its own eval set).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/cluster/env_mn5.sh

CONFIG="${1:-configs/esm8m_h100.yaml}"
N_SEQS="${2:-200000}"
N_PROT="${3:-20000}"
MIN_REQUIRED_INSTANCES="${4:-30}"
N_SAMPLES="${5:-50000}"
METRIC="${6:-auc}"
N_SHARDS_PER_SPLIT="${7:-1}"   # bounds the raw_mlp feature-bank cache — validated at 1 by run_mid_replicate.sh

RUN_NAME="$(python -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))['output_dir'].rstrip('/').split('/')[-1])" "$CONFIG")"
echo "[submit_pipeline] config=$CONFIG run=$RUN_NAME n_seqs=$N_SEQS n_prot=$N_PROT data_root=$GLP_DATA_ROOT runs_root=$GLP_RUNS_ROOT"

# Group this attempt's logs under runs/<run>/launches/<job_id>/, append the per-run
# launch ledger, and relocate Slurm's .out/.err there on exit (env_mn5.sh).
record_launch "$RUN_NAME" "$@"

EVAL_TSV="$GLP_DATA_ROOT/swissprot/swissprot_annotated_eval_${N_PROT}.tsv"
EVAL_DIR="$GLP_DATA_ROOT/$RUN_NAME/swissprot/processed"

echo "[submit_pipeline] 1/7 build_swissprot_eval"
python scripts/build_swissprot_eval.py --config "$CONFIG" \
    --n-prot "$N_PROT" --min-required-instances "$MIN_REQUIRED_INSTANCES"

echo "[submit_pipeline] 2/7 extract_activations"
python scripts/extract_activations.py --config "$CONFIG" --store train \
    --n-seqs "$N_SEQS" --embed-batch-size 64 --exclude-from "$EVAL_TSV"

echo "[submit_pipeline] 3/7 train_glp"
python scripts/train_glp.py --config "$CONFIG"

echo "[submit_pipeline] 4/7 eval_generation (Gate G2)"
python scripts/eval_generation.py --config "$CONFIG" \
    --n-samples "$N_SAMPLES" --fd-steps 20,50,100,300,1000
'''
echo "[submit_pipeline] 5/7 freeze_concepts"
python scripts/freeze_concepts.py --config "$CONFIG" --eval-dir "$EVAL_DIR"

echo "[submit_pipeline] 6/7 run_probing ($METRIC)"
# METRIC (arg 6): "auc" (default) | "f1" | "both". AUC is the primary GLP-protocol
# metric; a standard run does the AUC pass only. InterPLM-native domain-adjusted F1
# (D5) is implemented and resumable but OFF by default — it adds a full extra probing
# pass and relies on a rate-matched-threshold deviation that needs justifying if
# reported, so opt in explicitly (METRIC=both, or f1) when you actually want it. Both
# passes are independently resumable via metrics/probing_partial{,_f1}.jsonl.
run_probe() { python scripts/run_probing.py --config "$CONFIG" --eval-dir "$EVAL_DIR" \
    --concepts "$EVAL_DIR/frozen_concepts.json" --metric "$1" \
    --n-shards-per-split "$N_SHARDS_PER_SPLIT"; }
case "$METRIC" in
  both) run_probe auc; run_probe f1 ;;
  *)    run_probe "$METRIC" ;;
esac
'''
echo "[submit_pipeline] 7/7 make_report"
python scripts/make_report.py --config "$CONFIG"

echo "[submit_pipeline] done — see runs/$RUN_NAME/run_summary.md"
