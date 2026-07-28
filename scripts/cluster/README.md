# MN5 (BSC) launchers

SLURM job launchers for the `esm8m_h100.yaml` spine (PLAN §8/§9, §11 "W2 cluster"
section) on MareNostrum 5. Verified against this account on 2026-07-27:
account `bsc70`, GPU partition `acc` (H100 nodes, 4 GPU/node, `acc_bsccs`/`acc_debug`/
`acc_interactive` QOS), CPU partition `gpp` (`gp_bsccs`/`gp_debug` QOS). Python
3.11.5-gcc and CUDA 11.8-12.8 modules are available under `/apps/ACC/modulefiles`.

## One-time setup

1. **Pick a storage path** for the venv, HF cache, activation stores, and
   checkpoints/runs. As of 2026-07-27, `$HOME` has only ~6 GB free (80 GB quota),
   the `bsc70` **project** quota has only ~860 GB free group-wide, and `bsc70`
   **scratch** has ~28 TB free (but scratch is typically purged periodically —
   confirm the retention policy before relying on it for anything not reproducible
   from `docs/results/` + a re-run). Neither space had a directory for this user yet.
   ```bash
   cp scripts/cluster/cluster.env.example scripts/cluster/cluster.env
   # edit scripts/cluster/cluster.env, set GLP_DATA_ROOT=/gpfs/.../glp-prot-data
   ```
   `cluster.env` is gitignored — it's per-user, not shared.

2. **Build the venv** (login node, needs internet):
   ```bash
   scripts/cluster/setup_venv.sh
   ```

3. **Prefetch model + SAE weights** (login node, needs internet; small — ESM-2-8M
   + released InterPLM SAEs, not UniRef50):
   ```bash
   scripts/cluster/prefetch_weights.sh
   ```
   Jobs run with `HF_HUB_OFFLINE=1` (set in `env_mn5.sh`) so they never depend on
   compute-node internet access — this step is what populates the cache they read.

## Running the pipeline

Submit from the repo root. Every `submit_*.sh` takes `configs/esm8m_h100.yaml` as
its default first arg (override with e.g. `configs/poc_mac.yaml` for a smoke test),
and sources `scripts/cluster/env_mn5.sh`, which: loads the module stack, activates
the venv, symlinks `runs/` at the repo root to `$GLP_DATA_ROOT/runs` (only if
`runs/` doesn't already exist locally — it won't clobber a real local directory),
and points `HF_HOME` at cluster storage.

```bash
sbatch scripts/cluster/submit_extract_activations.sh      # unlabeled GLP training store
sbatch scripts/cluster/submit_build_swissprot_eval.sh      # labeled Swiss-Prot eval store
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
(`extract_activations` and `build_swissprot_eval` build independent stores — PLAN
§6.1 "two stores" — chaining them sequentially here is just for simplicity/QOS
concurrency headroom, not a real dependency; run them in parallel yourself if you'd
rather not wait.)

Override QOS/time/account per submission with normal `sbatch` flags, e.g.:
```bash
sbatch --qos=acc_debug --time=00:30:00 scripts/cluster/submit_train_glp.sh configs/poc_mac.yaml
```

## Known gaps (don't be surprised)

- **`extract_activations.py` doesn't stream UniRef50 yet.** It only pulls a
  Swiss-Prot REST slice (the same code path as the Mac PoC / `local_mid` runs),
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
