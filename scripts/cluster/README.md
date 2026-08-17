# MN5 (BSC) launchers

SLURM job launchers for the `esm8m_h100.yaml` spine (PLAN §8/§9, §11 "W2 cluster"
section) on MareNostrum 5. Verified against this account on 2026-07-27:
account `bsc70`, GPU partition `acc` (H100 nodes, 4 GPU/node, `acc_bsccs`/`acc_debug`/
`acc_interactive` QOS), CPU partition `gpp` (`gp_bsccs`/`gp_debug` QOS). Python
3.11.5-gcc and CUDA 11.8-12.8 modules are available under `/apps/ACC/modulefiles`.

## One-time setup

Five steps, each **run once per account** (not per run/job). When done, skip straight to [Running the pipeline](#running-the-pipeline) —
`env_mn5.sh` (sourced by every `submit_*.sh`) activates the existing venv
automatically.

1. **Submodules.** A plain `git clone` leaves `InterPLM/`/`generative_latent_prior/`
   empty (`ModuleNotFoundError: No module named 'interplm'` otherwise):
   ```bash
   git submodule update --init --recursive
   ```

2. **`scripts/cluster/cluster.env`** (gitignored, per-user). Three exports, then
   create the two directories:
   ```bash
   export GLP_DATA_ROOT=/gpfs/scratch/bsc70/$USER/glp-prot-data    # ~28TB, periodically purged
   export GLP_RUNS_ROOT=/gpfs/projects/bsc70/$USER/glp-prot-runs   # ~860GB group-wide, durable
   export GLP_VENV=/home/bsc/bsc070332/envs/glpprot_venv           # now in $HOME, ideally out of here
   mkdir -p "$GLP_DATA_ROOT" "$GLP_RUNS_ROOT"
   ```
   *Why not `$HOME`:* only ~80 GB quota total — `torch` + pip's cache alone can
   blow through it, and the failure mode is a confusing half-installed `pip` error
   rather than a clear "disk full". (This account's `GLP_VENV` is actually still
   under `$HOME` at `~/envs/glpprot_venv` — pre-dates this guidance; fine to leave,
   just don't copy that path for a new account.)

3. **Venv** (login node, needs internet). If `$GLP_VENV` already exists:

   ```bash
   source scripts/cluster/env_mn5.sh
   ```
   Otherwise:
   
   ```bash
   module load intel impi mkl hdf5 python/3.11.5-gcc cuda/12.4
   python -m venv "$GLP_VENV"
   source "$GLP_VENV/bin/activate"
   pip install --upgrade pip
   PIP_CACHE_DIR="$GLP_DATA_ROOT/pip_cache" pip install -r requirements.txt
   ```
4. **Model + SAE weights** (login node, needs internet; small — ESM-2-8M + the
   released InterPLM SAEs, not UniRef50). Lands in `$GLP_RUNS_ROOT/hf_cache`:
   ```bash
   scripts/cluster/prefetch_weights.sh
   ```
   Jobs run with `HF_HUB_OFFLINE=1` (set in `env_mn5.sh`), so this is what they
   read from.

5. **Swiss-Prot data** (login node, needs internet). Labeled eval slice + unlabeled
   training slice, shared across every run/config (same REST query → same result,
   so no `--config` needed). Needs the venv active and `GLP_DATA_ROOT` set — already
   true if you're continuing the same shell from step 3, otherwise `source
   scripts/cluster/env_mn5.sh` first:
   ```bash
   python scripts/prefetch_data.py --n-seqs 200000 --n-prot 20000
   ```
   `--n-seqs`/`--n-prot` are baked into the output filenames — match whatever
   you'll later pass `submit_extract_activations.sh`/`submit_build_swissprot_eval.sh`,
   or this just warms the cache without being used. Add `--with-uniref50` to also
   pull the full ~12 GB UniRef50 FASTA into `$GLP_DATA_ROOT/uniref50/` — an explicit
   go-ahead per `CLAUDE.md`; nothing consumes it yet (`producer_consumer.py` is
   still a stub), this just pre-lands it.

## Storage layout

By design, `runs/<run>/` is a plain local directory in the repo holding only the
lightweight output artifacts. Everything large or checkpoint-like is
redirected to cluster storage by `glp_plm.utils.resolve_storage_dir` (per-run) or
`resolve_shared_dir` (shared across runs), called from inside each pipeline script
itself (not a separate setup step, so it happens automatically no matter which
script runs first). **Check `runs` isn't itself a symlink on your account first**
(`stat runs`) — on this account it turned out to be `runs -> $GLP_RUNS_ROOT`
wholesale (predates this doc), which collapses the two trees below into one, so
e.g. `checkpoints/` shows up "under `runs/`" too despite the "NOT symlinked" notes
below. Harmless (same data, reachable either path).

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
    ├── figures/                     -> runs/<run>/figures    (symlink)
    ├── launches.jsonl              # one row per launch attempt (the ledger)
    └── launches/<job_id>/          # per-attempt ephemera, grouped by job id
        ├── launch.txt              #   submit args + `scontrol show job` allocation
        ├── stdout.log  stderr.log  #   Slurm's job output, relocated here on exit
```

A run can be launched more than once — a timeout/crash is resubmitted and
`train_glp.py` auto-resumes from its checkpoint, so those attempts are **one run**,
not several. Their shared state (`checkpoints/`, `metrics/`, `train_store/`) is
per-run and must not be split by job id (resume depends on the next job finding the
previous job's `glp_ckpt.pt`). Only the per-attempt ephemera — the launch manifest
and Slurm's `.out`/`.err` — is grouped under `launches/<job_id>/` (via
`record_launch` in `env_mn5.sh`), with a `launches.jsonl` ledger giving a one-glance
history of every attempt (job id, TimeLimit, QOS, submit args, git commit). For a
genuinely *separate* experiment (different seed, independent training) use a distinct
`output_dir` — that, not a job-id subfolder, is the run-identity boundary.

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

**Reusing one activation store across many runs (scaling/architecture grids).** The
store depends only on the *data* (model, layer, sequences, seed, count) — not on the
GLP architecture (`n_blocks`) or `batch_size` — so a depth/batch grid over identical
activations should share one store, never re-extract per cell. Set `data.store_dir` in
each grid config to an existing store (absolute, or relative to `$GLP_DATA_ROOT`, e.g.
`uniref50_500m/train_store`); `train_glp`/`eval_generation` read from there instead of
the per-run default (CLI `--store-dir` overrides). `extract_activations` ignores it and
always writes the per-run store, so a shared store is never overwritten by an accidental
re-extract. This replaces the older hand-made per-run `train_store` symlink.

The **labeled Swiss-Prot eval store** works the same way via `data.eval_dir`. It depends
only on model+layer+n_prot (not the GLP run), so many runs share one prebuilt
`.../swissprot/processed` (shards + labels + `frozen_concepts.json`), e.g.
`data.eval_dir: uniref50_100m/swissprot/processed`. `freeze_concepts`/`run_probing` read
from there (CLI `--eval-dir` overrides); `build_swissprot_eval` always writes the per-run
store. When either field is set, `submit_pipeline.sh` **skips the corresponding stage**:
`store_dir` skips extract (2/7), `eval_dir` skips build (1/7) and freeze (5/7). (Both live
on scratch, which is periodically purged — a purge means one rebuild, not per-run.)

## Running the pipeline

Submit from the repo root. Every launcher takes `configs/esm8m_h100.yaml` — the single
canonical config, see Config unification below — as its default first arg, and sources
`scripts/cluster/env_mn5.sh`, which loads the module stack, activates the repo venv, and
points `HF_HOME` at `$GLP_RUNS_ROOT/hf_cache`. The per-run storage split above is
handled separately, inside each pipeline script.

**One job, all seven stages** (`submit_pipeline.sh`) is the default way to run this —
sequential plain `python` calls inside a single `sbatch` job, not seven separate
submissions:
```bash
sbatch scripts/cluster/submit_pipeline.sh [config] [n_seqs] [n_prot] \
    [min_required_instances] [n_samples] [metric] [n_shards_per_split]
```
This exists (replacing an earlier version that called `sbatch` seven times with
`--dependency=afterok`, and the one-off `run_mid_replicate.sh`, both retired
2026-07-29) because MN5 is congested enough that `acc_debug` is the QOS that actually
gets a job started, and `acc_debug` rejects a second job from the same user while one
is still queued — a seven-job fan-out doesn't work under it even with dependencies,
since all seven sit in the queue immediately on submission. One job sidesteps that.
`acc_debug` is also time-capped, though (the script defaults to the same 2h that
worked for `run_mid_replicate.sh` at 10M activations) — for a real 100M+ run, override
QOS/time, e.g.:
```bash
sbatch --qos=acc_bsccs --time=1-00:00:00 scripts/cluster/submit_pipeline.sh \
    configs/esm8m_h100.yaml 5000000 20000
```
Reserve the `acc_debug` default for smoke tests at small `n_seqs`.

`build_swissprot_eval` must run before `extract_activations` (not just "run
individually in any order") — the latter's `--exclude-from` needs the former's eval
TSV to keep training/eval disjoint (PLAN §9/§10). Running them out of order, or
`extract_activations` alone without having prefetched/built the eval TSV first,
silently trains on the eval set.

**Per-stage launchers** are still there if you want to rerun just one stage (e.g. after
a crash) without repeating the whole chain — each is its own `sbatch` job, `acc_bsccs`
by default:
```bash
sbatch scripts/cluster/submit_build_swissprot_eval.sh      # labeled Swiss-Prot eval store
sbatch scripts/cluster/submit_extract_activations.sh       # unlabeled GLP training store
sbatch scripts/cluster/submit_train_glp.sh                 # flow-matching training
sbatch scripts/cluster/submit_eval_generation.sh           # FD / PCA — Gate G2
sbatch scripts/cluster/submit_freeze_concepts.sh           # §7 fairness: freeze BEFORE probing
sbatch scripts/cluster/submit_run_probing.sh               # Deliverable 2 (add "f1" arg for F1)
sbatch scripts/cluster/submit_make_report.sh               # re-render figures
```
These still chain fine with `--dependency=afterok` one at a time if you're not on
`acc_debug` (e.g. `acc_bsccs`, which does allow multiple queued jobs).

**Probing extras** (GLProt probing spec: continuous-property probing, the
nonlinearity falsification test, and Part 7 output assembly — `probe_continuous.py`,
`test_nonlinearity.py`, `make_probing_outputs.py`, on top of `run_probing.py`) follow
the same one-job pattern as the main pipeline, via `submit_probing_pipeline.sh`,
rather than a separate per-stage launcher for each — `acc`/`acc_bsccs` congestion
(2026-08-17) made a 3-job `--dependency=afterok` chain for these impractically slow
to even start, and unlike the main pipeline's per-stage launchers (each independently
useful — rerun training, rerun just the FD gate, etc.), these three have no track
record of standalone use, so bundling them is not a loss of flexibility:
```bash
sbatch scripts/cluster/submit_probing_pipeline.sh [config] [eval_dir] [ckpt] [scaler] \
    [n_shards_per_split] [max_proteins]
```
`eval_dir`/`ckpt`/`scaler` override the config's own defaults — needed when reusing a
checkpoint whose output_dir was renamed after training, or a shared eval store built
under a different run (see the script's header). `run_probing.py` itself keeps its
own `submit_run_probing.sh` launcher above (proven standalone use: AUC vs F1 reruns).

Leakage verification (`verify_no_leakage.py` — exact-sequence check between a
training manifest and every eval protein, PLAN §7 `⚠ FAIRNESS`) is its own launcher,
`submit_verify_no_leakage.sh`, since its natural point in the workflow is right after
`extract_activations.py`, independent of any particular probing run; `run_probing.py`
and `probe_continuous.py` also run this gate automatically by default.

### Config unification (2026-07-29)

There used to be three near-identical configs (`mid_mn5.yaml`, `uniref50_100m.yaml`,
and this one) differing only in data source/scale/output_dir. `mid_mn5.yaml` (and its
`run_mid_replicate.sh` launcher) proved the cluster launchers reproduce the local
`local_clean` result at 10M Swiss-Prot activations — done, job 43929721, 2026-07-28 —
so both are retired. `configs/esm8m_h100.yaml` is now the single canonical config and
carries the next step forward: real UniRef50 at 100M activations. See its header
comment for what changed and why (still fp32/precache — bf16/streaming aren't
implemented yet).

That run completed 2026-07-29 (job 43965559, `runs/uniref50_100m/`): loss and probing
AUC both improved over `local_clean`, but the FD generation-quality ratio (3.79x vs
1.55x) needs verification before trusting it as a regression — the real-vs-real FD
lower bound shrank 5.6x on UniRef50, so the absolute FD gap actually got smaller even
though the ratio got worse. See `runs/uniref50_100m/run_summary.md` for the numbers.

To train on **100M activations pooled from a 5M-sequence UniRef50 reservoir sample**
(matching InterPLM's own SAE training distribution, PLAN line 95):
```bash
# one-time, needs internet (login node) — ~12 GB compressed, per CLAUDE.md's
# "ask before large downloads" rule; only needed once, shared across runs
source scripts/cluster/env_mn5.sh
python scripts/prefetch_data.py --n-seqs 200000 --n-prot 20000 --with-uniref50

# then queue the pipeline as one job — 5,000,000 = the sequence pool size (--n-seqs,
# reservoir-sampled by read_fasta_slice down to whatever n_train_activations needs),
# 20000 = --n-prot for the eval set; override qos/time since 2h/acc_debug won't cover
# a 100M-activation run end-to-end (see above)
sbatch --qos=acc_bsccs --time=1-00:00:00 scripts/cluster/submit_pipeline.sh \
    configs/esm8m_h100.yaml 5000000 20000
```

Override QOS/time/account per submission with normal `sbatch` flags, e.g.:
```bash
sbatch --qos=acc_debug --time=00:30:00 scripts/cluster/submit_train_glp.sh configs/esm8m_h100.yaml
```

## Smoke-testing without waiting on a big pull

The canonical config's `data.train_fasta: uniref50` reads from the local UniRef50 pull
via reservoir sampling (`read_fasta_slice`), which needs `uniref50.fasta.gz` prefetched
first (`prefetch_data.py --with-uniref50`, one-time, ~12 GB — see one-time setup step
5). A small `--n-seqs` still does a full pass over that file (reservoir sampling is
one-pass but not proportional-cost), so it saves you the download/rate-limit risk and
lets you exercise the full pipeline end-to-end on a tiny store — not a fast scan:

```bash
source scripts/cluster/env_mn5.sh   # activates the venv
python scripts/prefetch_data.py --n-seqs 500 --n-prot 2000 --with-uniref50
sbatch --qos=acc_debug --time=00:30:00 scripts/cluster/submit_extract_activations.sh configs/esm8m_h100.yaml 500 2000
```

To smoke-test the *original* Swiss-Prot-REST training path instead (fast — a live
UniProt REST call, paginated, skipped if the destination FASTA already exists), point
at a config with `data.train_fasta: null` — there is no such config checked in right
now (the canonical one uses real UniRef50); copy `configs/esm8m_h100.yaml` and flip
that one field if you need this path exercised.

## Known gaps (don't be surprised)

- **`record_launch` (`env_mn5.sh`) relocates each job's `.out`/`.err` into
  `runs/<run>/launches/<job_id>/{stdout.log,stderr.log}` on exit** (EXIT trap). The
  two `mv`s are independently guarded (`{ …; } 2>/dev/null || true`) so a failure on
  one can't abort the trap and strand the other in the submit dir — the bug that
  used to leave `.err` in `$HOME` under `set -e` (2026-07-28, job 43929721). If a
  job's `glp-*-<id>.out`/`.err` are still sitting in the repo root well after it
  *finished* (not mid-run), that's a real miss worth investigating. The relocation is
  a cross-filesystem `mv` (submit dir on `$HOME`/gpfs-home, destination on
  gpfs-projects), i.e. copy+delete, so a few lines of epilog stdout written *after*
  the trap are not captured — expected, not a bug.
- **`extract_activations.py` doesn't truly *stream* UniRef50 (as in, unbounded, off a
  live source) yet.** `data.train_fasta: uniref50` (the canonical config's setting)
  does work today — it reservoir-samples `--n-seqs` sequences out of a **local**
  prefetched `uniref50.fasta.gz` (`read_fasta_slice`), one pass, capped at
  `n_train_activations` activations. What's still a stub is
  `glp_plm/data/producer_consumer.py` — true producer-consumer streaming that would
  let a run exceed what fits precached on scratch (PLAN §6.1/§9 W2). 100M activations
  (64 GB fp16) fits fine precached; going past a few hundred M will need that
  implemented — check with whoever's driving the paper before assuming otherwise.
- **UniRef50 is not downloaded by anything here.** Per `CLAUDE.md`, that's an
  "ask first" large download (~12 GB compressed) — `prefetch_weights.sh`
  deliberately does not fetch it.
- **No launcher for `run_steering.py` / `auto_interp_metaneurons.py`.** Both are
  still CLI stubs (`NotImplementedError`, PLAN §9 W4) — add a launcher once they're
  implemented, following the pattern in the other `submit_*.sh` files.
- **`train_glp.py` checkpoints and auto-resumes.** It writes a periodic
  `checkpoints/glp_ckpt.pt` ({model, optimizer, step, total_steps, history}) every
  2000 steps, atomically (temp + rename, so a SIGKILL at the QOS wall-clock cap can't
  corrupt it). Resubmitting the *same* `submit_train_glp.sh <config>` after a timeout
  auto-continues from the last checkpoint — no flag needed (pass `--restart` to force
  a fresh run; a resume whose `total_steps` disagrees with the checkpoint, i.e. a
  changed `n_train_activations`/`num_epochs`, is refused rather than run on a desynced
  LR schedule). `acc_bsccs` still caps a single job at 2 days, but the run no longer
  has to fit in one — e.g. `runs/uniref50_500m` stopped at step 174000/≈976000 and is
  resumable today. `run_probing.py` is separately resumable via
  `metrics/probing_partial*.jsonl`, independent of this.
- **The `python/3.11.5-gcc` module exports `PYTHONHOME`/`PYTHONPATH`** pointing at
  its own shared site-packages (legacy `.egg` installs of torch's deps). Left in
  place this breaks venv isolation — `pip` reports packages "already satisfied"
  there, but `.egg` dirs aren't actually importable, so `import X` still
  `ModuleNotFoundError`s. `env_mn5.sh` `unset`s both after the module load; do the
  same if you ever load modules by hand.
- **Login-node shells may have a low `ulimit -t` (CPU time) cap** (300s as of
  2026-07-28) that can kill a `pip install` mid-`torch`-build. Check `ulimit -St`;
  if hit, use `--only-binary=:all:` or install packages one at a time.
