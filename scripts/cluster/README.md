# MN5 (BSC) launchers

SLURM job launchers for the `esm8m_h100.yaml` spine (PLAN §8/§9, §11 "W2 cluster"
section) on MareNostrum 5. Verified against this account on 2026-07-27:
account `bsc70`, GPU partition `acc` (H100 nodes, 4 GPU/node, `acc_bsccs`/`acc_debug`/
`acc_interactive` QOS), CPU partition `gpp` (`gp_bsccs`/`gp_debug` QOS). Python
3.11.5-gcc and CUDA 11.8-12.8 modules are available under `/apps/ACC/modulefiles`.

## One-time setup

0. **Init the submodules** — a plain `git clone` leaves `InterPLM/`/
   `generative_latent_prior/` empty, which fails as `ModuleNotFoundError: No module
   named 'interplm'`:
   ```bash
   git submodule update --init --recursive
   ```

1. **Fill in `scripts/cluster/cluster.env`** (gitignored, per-user) with two
   storage paths, one for data and one for runs, and create them. `$HOME` has only
   ~80 GB quota (as of 2026-07-27, ~6 GB free) — don't put `GLP_VENV` there either;
   `torch` + pip's download cache can fill it and turn into confusing half-installed
   `pip` errors instead of a clear "disk full". Point `GLP_VENV` at `$GLP_RUNS_ROOT`
   and set `PIP_CACHE_DIR="$GLP_DATA_ROOT/pip_cache"`. `bsc70` **project** quota:
   ~860 GB free group-wide; `bsc70` **scratch**: ~28 TB free (periodically purged).
   ```bash
   mkdir -p "$(grep GLP_DATA_ROOT scripts/cluster/cluster.env | cut -d= -f2)" \
            "$(grep GLP_RUNS_ROOT scripts/cluster/cluster.env | cut -d= -f2)"
   ```

2. **Build the venv** (login node, needs internet — same `requirements.txt` as the
   Mac PoC, Python 3.11 + CUDA modules for the H100 nodes). Defaults to
   `$REPO_ROOT/.venv`; if you want it elsewhere (e.g. off `$HOME`'s tight quota),
   set `GLP_VENV` in `cluster.env` to match:
   ```bash
   module load intel impi mkl hdf5 python/3.11.5-gcc cuda/12.4
   python -m venv "${GLP_VENV:-.venv}"
   source "${GLP_VENV:-.venv}/bin/activate"
   pip install --upgrade pip
   # optional: keep pip's download cache off $HOME (tight quota) once cluster.env is set:
   #   export PIP_CACHE_DIR="$GLP_DATA_ROOT/pip_cache"
   pip install -r requirements.txt
   ```

3. **Prefetch model + SAE weights** (login node, needs internet; small — ESM-2-8M
   + released InterPLM SAEs, not UniRef50). Lands in the HF cache under
   `$GLP_RUNS_ROOT/hf_cache` (durable project storage, not scratch):
   ```bash
   scripts/cluster/prefetch_weights.sh
   ```
   Jobs run with `HF_HUB_OFFLINE=1` (set in `env_mn5.sh`) so they never depend on
   compute-node internet access — this step is what populates the cache they read.

4. **Prefetch Swiss-Prot data** (login node, needs internet) — the labeled eval
   slice and the unlabeled training slice, written to `$GLP_DATA_ROOT/swissprot/`.
   This is shared across every run/config (same REST query -> same result, so no
   `--config` needed), not nested per-run:
   ```bash
   source scripts/cluster/env_mn5.sh
   python scripts/prefetch_data.py --n-seqs 200000 --n-prot 20000
   ```
   `scripts/extract_activations.py` / `scripts/build_swissprot_eval.py` skip
   re-fetching if these files already exist, so this just warms the cache ahead of
   the (internet-less) compute-node jobs — as long as `--n-seqs`/`--n-prot` match
   what you'll pass those scripts (baked into the filenames). Pass `--with-uniref50`
   to also pull the full ~12 GB UniRef50 FASTA into `$GLP_DATA_ROOT/uniref50/` — an
   explicit go-ahead per `CLAUDE.md`; nothing consumes it yet (`producer_consumer.py`
   is still a stub), this just lands it in its reserved spot.

## Storage layout

`runs/<run>/` (a plain local directory, not a symlink) ends up holding only the
lightweight "browse this" artifacts — everything large or checkpoint-like is
redirected to cluster storage by `glp_plm.utils.resolve_storage_dir` (per-run) or
`resolve_shared_dir` (shared across runs), called from inside each pipeline script
itself (not a separate setup step, so it happens automatically no matter which
script runs first):

```
$GLP_DATA_ROOT/
├── swissprot/                       # SHARED — same REST pull regardless of run
│   ├── swissprot_annotated_eval_<n_prot>.tsv
│   └── train_slice_<n_seqs>.fasta
└── <run>/
    ├── train_store/                 # NOT symlinked under runs/ (see below)
    └── swissprot/processed/         # NOT symlinked under runs/ (see below)
                                        # embedded eval shards — per-run (model+layer
                                        # specific), unlike the raw TSV/FASTA above

$GLP_RUNS_ROOT/
├── hf_cache/                        # SHARED — ESM-2 + SAE weights
└── <run>/
    ├── checkpoints/                 # NOT symlinked under runs/ (see below)
    ├── metrics/                     -> runs/<run>/metrics    (symlink)
    └── figures/                     -> runs/<run>/figures    (symlink)
```

The raw Swiss-Prot TSV/FASTA are identical for any run pulling the same
`--n-prot`/`--n-seqs`, so they live once under `$GLP_DATA_ROOT/swissprot/` (keyed by
that count in the filename) rather than duplicated per-run — same idea as
`$GLP_DATA_ROOT/uniref50/`. The *embedded* eval store (`processed/`, per-AA labels
aligned to activations at this run's model+layer) is run-specific and stays
redirected under `$GLP_DATA_ROOT/<run>/swissprot/processed/` — it is NOT
preprocessed anywhere under `$GLP_RUNS_ROOT`.

Checkpoints (`glp_final.pt`, `glp_ckpt.pt`, `rep_statistics.pt`), the activation
store, and the embedded Swiss-Prot eval shards are all data-sized and **not**
symlinked under `runs/` at all — they live only at `$GLP_RUNS_ROOT/<run>/checkpoints/`
/ `$GLP_DATA_ROOT/<run>/train_store/` / `$GLP_DATA_ROOT/<run>/swissprot/processed/`,
reached directly (`--ckpt`/`--scaler`/`--store-dir`/`--eval-dir` on the relevant
scripts if you need to point at one explicitly). Only `metrics/`/`figures/` — small
and worth browsing — get the local symlink.

An existing real (non-symlink) directory at any of these paths is left alone with a
warning instead of being clobbered.

## Running the pipeline

Submit from the repo root. Every `submit_*.sh` takes `configs/esm8m_h100.yaml` as
its default first arg (override with e.g. `configs/poc_mac.yaml` for a smoke test),
and sources `scripts/cluster/env_mn5.sh`, which loads the module stack, activates the
repo venv, and points `HF_HOME` at `$GLP_RUNS_ROOT/hf_cache`. The per-run storage
split above is handled separately, inside each pipeline script.

```bash
sbatch scripts/cluster/submit_build_swissprot_eval.sh      # labeled Swiss-Prot eval store
sbatch scripts/cluster/submit_extract_activations.sh       # unlabeled GLP training store
sbatch scripts/cluster/submit_train_glp.sh                 # flow-matching training
sbatch scripts/cluster/submit_eval_generation.sh           # FD / PCA — Gate G2
sbatch scripts/cluster/submit_freeze_concepts.sh           # §7 fairness: freeze BEFORE probing
sbatch scripts/cluster/submit_run_probing.sh               # Deliverable 2 (add "f1" arg for F1)
sbatch scripts/cluster/submit_make_report.sh               # re-render figures
```

Or chain the whole thing as SLURM-dependent jobs in one call:
```bash
scripts/cluster/submit_pipeline.sh configs/esm8m_h100.yaml 200000 20000
```
(Runs on the login node, just calling `sbatch` seven times with `--dependency=afterok`.)

To replicate the canonical `local_clean` Mac result (`docs/results/run_summary.md`
§10) on `configs/mid_mn5.yaml` before trusting a scale-up, use `run_mid_replicate.sh`
instead — one `sbatch` job running the whole chain sequentially:
```bash
sbatch scripts/cluster/run_mid_replicate.sh
```
`build_swissprot_eval` must run before `extract_activations` (not just "run individually
in any order") — the latter's `--exclude-from` needs the former's eval TSV to keep
training/eval disjoint (`docs/results/run_summary.md` §7). Running them out of order,
or `extract_activations` alone without having prefetched/built the eval TSV first,
silently trains on the eval set — `submit_extract_activations.sh` prints a WARNING if
it can't find one, but doesn't fail the job.

Override QOS/time/account per submission with normal `sbatch` flags, e.g.:
```bash
sbatch --qos=acc_debug --time=00:30:00 scripts/cluster/submit_train_glp.sh configs/poc_mac.yaml
```

## Smoke-testing without waiting on a big pull

`extract_activations.py` fetches its Swiss-Prot slice via a **live UniProt REST
call**, and compute nodes are not assumed to have internet (see prefetch note
above) — but it skips the fetch entirely if the destination FASTA already
exists. So to test the pipeline on a small subset without needing network
inside the job: pre-fetch that file from a login node (which has internet),
then submit as normal. `n_seqs` must match the value you'll pass to
`--n-seqs`/`submit_extract_activations.sh`, since it's baked into the filename:

```bash
source scripts/cluster/env_mn5.sh   # activates the venv
python scripts/prefetch_data.py --n-seqs 500 --n-prot 2000
sbatch --qos=acc_debug --time=00:30:00 scripts/cluster/submit_extract_activations.sh configs/poc_mac.yaml 500 2000
```

## Known gaps (don't be surprised)

- **`extract_activations.py` doesn't stream UniRef50 yet.** It only pulls a
  Swiss-Prot REST slice (the same code path as the Mac PoC / `local_clean` run),
  sized up via `--n-seqs`. `glp_plm/data/producer_consumer.py` (true streaming
  producer-consumer over UniRef50, PLAN §6.1/§9 W2) is still a stub. Getting to the
  full 50→300M activation budget in `esm8m_h100.yaml` needs that implemented, or a
  much larger Swiss-Prot REST pull (rate-limit risk) — check with whoever's driving
  the paper before assuming `submit_extract_activations.sh` alone reaches 50M+ acts.
- **UniRef50 is not downloaded by anything here.** Per `CLAUDE.md`, that's an
  "ask first" large download (~12 GB compressed) — `prefetch_weights.sh`
  deliberately does not fetch it.
- **No launcher for `run_steering.py` / `auto_interp_metaneurons.py`.** Both are
  still CLI stubs (`NotImplementedError`, PLAN §9 W4) — add a launcher once they're
  implemented, following the pattern in the other `submit_*.sh` files.
- **No checkpoint/resume in `train_glp.py`.** A run is single-shot to completion;
  `acc_bsccs` QOS caps wall time at 2 days. `run_probing.py` is separately resumable
  via `metrics/probing_partial*.jsonl`, independent of this.
- **The `python/3.11.5-gcc` module exports `PYTHONHOME`/`PYTHONPATH`** pointing at
  its own shared site-packages (legacy `.egg` installs of torch's deps). Left in
  place this breaks venv isolation — `pip` reports packages "already satisfied"
  there, but `.egg` dirs aren't actually importable, so `import X` still
  `ModuleNotFoundError`s. `env_mn5.sh` `unset`s both after the module load; do the
  same if you ever load modules by hand.
- **Login-node shells may have a low `ulimit -t` (CPU time) cap** (300s as of
  2026-07-28) that can kill a `pip install` mid-`torch`-build. Check `ulimit -St`;
  if hit, use `--only-binary=:all:` or install packages one at a time.
