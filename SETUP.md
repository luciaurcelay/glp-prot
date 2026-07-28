# Setup (Apple-Silicon PoC)

## 0. Get the submodules

`InterPLM/` and `generative_latent_prior/` are git submodules, not vendored code — a
plain `git clone` leaves both as empty directories. Needed before anything below,
on Mac or cluster alike:
```bash
git submodule update --init --recursive
```
(`git submodule status` shows a leading `-` on an uninitialized submodule; no prefix
once it's checked out.)

InterPLM requires **Python 3.11** (the repo's system Python 3.9 is too old).

```bash
brew install python@3.11
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip

# Runtime deps for the W1 pipeline (torch, transformers, sklearn, scipy, pandas,
# h5py, einops, huggingface-hub, biopython, typed-argument-parser, matplotlib, pytest)
.venv/bin/pip install "torch>=2.8" "numpy>=2.2,<2.3" scipy pandas scikit-learn \
  "transformers>=4.57" biopython "huggingface-hub>=0.28" h5py einops \
  pyyaml tqdm typed-argument-parser matplotlib pytest wandb
```

## Making `interplm` and `glp_plm` importable

We do **not** `pip install -e InterPLM`: InterPLM's `setup.py` uses `find_packages()`
but `interplm/train/__init__.py` is missing upstream, so a non-editable install drops
the whole `interplm.train` subtree (which `interplm.sae.inference` imports). Editable
installs also proved flaky here because Homebrew's Python ships a `sitecustomize.py`
that shadows venv site hooks.

Instead, both packages are put on `sys.path` explicitly:
- **Tests:** `tests/conftest.py` adds the repo root and `InterPLM/`.
- **Scripts:** each imports `scripts/_bootstrap.py` first, which does the same.
- **Ad-hoc:** `PYTHONPATH="$(pwd):$(pwd)/InterPLM" .venv/bin/python ...`

## Verify

```bash
.venv/bin/python -m pytest tests/ -q          # 6 passed (hook-parity is live)
```

## W1 pipeline (all use configs/poc_mac.yaml)

```bash
.venv/bin/python scripts/download_interplm_saes.py --config configs/poc_mac.yaml
.venv/bin/python scripts/extract_activations.py    --config configs/poc_mac.yaml --store train
.venv/bin/python scripts/build_swissprot_eval.py   --config configs/poc_mac.yaml
.venv/bin/python scripts/train_glp.py              --config configs/poc_mac.yaml   # + training_loss/lr figures
.venv/bin/python scripts/eval_generation.py        --config configs/poc_mac.yaml   # Gate G1 + FD/PCA figures
.venv/bin/python scripts/make_report.py            --config configs/poc_mac.yaml   # re-render all figures
```

Figures and metrics are written under `runs/<name>/` (`figures/*.png`, `metrics/`).
`runs/` is gitignored; committed results live in `docs/results/`.

## Cluster (MN5, BSC)

Don't hand-run `.venv/bin/python` over SSH for the cluster spine — use the SLURM
launchers in `scripts/cluster/` (separate venv, on cluster storage, built against
Python 3.11.5-gcc + CUDA via `module load`). See `scripts/cluster/README.md` for
one-time setup and `sbatch` invocations.

## W3 probing (the canonical run used `configs/local_clean.yaml`)

```bash
# labeled eval set (20k densely-annotated Swiss-Prot proteins)
.venv/bin/python scripts/build_swissprot_eval.py --config configs/local_clean.yaml --n-prot 20000 --min-required-instances 30

# freeze the concept set BEFORE probing (PLAN §7 fairness rule)
.venv/bin/python scripts/freeze_concepts.py --config configs/local_clean.yaml --eval-dir runs/local_clean/swissprot/processed

# 4-method probing; add --metric f1 for InterPLM domain-adjusted F1
.venv/bin/python scripts/run_probing.py --config configs/local_clean.yaml --eval-dir runs/local_clean/swissprot/processed --concepts runs/local_clean/swissprot/processed/frozen_concepts.json --n-shards-per-split 1
```

Probing is **resumable**: every `(concept, method)` is checkpointed to
`metrics/probing_partial[_f1].jsonl`, so an interruption costs one probe. Pass
`--restart` to ignore a saved checkpoint.

## Memory notes (16 GB Mac)

RAM, not compute, is the local ceiling. 10M activations = 6.4 GB fp16 is comfortable;
25M = 16 GB is not. When adding a new feature bank, stream it to a memmap instead of
accumulating in RAM, and clear the MPS cache inside long loops.
