# Setup

The pipeline itself is **cluster-only now** (see `## Cluster (MN5, BSC)` below) — the
local Mac PoC configs (`poc_mac.yaml`, `local_clean.yaml`) that W1/W1.5 ran are retired.
Everything through "Verify" below still applies for local dev/tests (editing code,
running `pytest`); it's just no longer how you run training/eval.

## 0. Get the submodules

`InterPLM/` and `generative_latent_prior/` are git submodules, not vendored code — a
plain `git clone` leaves both as empty directories. Needed before anything below,
on Mac or cluster alike:
```bash
git submodule update --init --recursive
```
(`git submodule status` shows a leading `-` on an uninitialized submodule; no prefix
once it's checked out.)

Then apply our local InterPLM patches:
```bash
scripts/apply_interplm_patches.sh
```
InterPLM is pinned at upstream `ElanaPearl/InterPLM`, so we don't bump the submodule
pointer for our fixes (a SHA not on upstream would break `git submodule update` on a
fresh clone). Instead they live as patch files in `patches/` and this idempotent script
applies them to the submodule working tree — **re-run it after any `git submodule
update`**, which resets the submodule to the pinned commit and drops the patches.
Currently: `interplm-nnsight-compat.patch` adapts `interplm/sae/intervention.py` to the
installed `nnsight` API; without it, steering (`scripts/run_steering.py`) crashes during
generation with an `expand()` shape error.

InterPLM requires **Python 3.11** (the repo's system Python 3.9 is too old).

```bash
brew install python@3.11
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip

# Runtime deps for the W1 pipeline (torch, transformers, sklearn, scipy, pandas,
# h5py, einops, huggingface-hub, biopython, typed-argument-parser, matplotlib, pytest);
# modlAMP is only needed for continuous-property probing (aliphatic index —
# glp_plm/eval/continuous_properties.py, GLProt probing spec Part 1)
.venv/bin/pip install "torch>=2.8" "numpy>=2.2,<2.3" scipy pandas scikit-learn \
  "transformers>=4.57" biopython modlamp "huggingface-hub>=0.28" h5py einops \
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

## Cluster (MN5, BSC)

Don't hand-run `.venv/bin/python` over SSH for the pipeline — use the SLURM launchers
in `scripts/cluster/` (separate venv, on cluster storage, built against Python
3.11.5-gcc + CUDA via `module load`). Default entrypoint is one job running all seven
stages sequentially:
```bash
sbatch scripts/cluster/submit_pipeline.sh configs/esm8m_h100.yaml 5000000 20000
```
See `scripts/cluster/README.md` for one-time setup, the per-stage launchers (for
rerunning a single stage), and the `acc_debug` QOS caveat (time-capped, one job at a
time — override `--qos`/`--time` for a real run past smoke-test scale).

Figures and metrics land under `runs/<name>/` (`figures/*.png`, `metrics/`, plus a
per-run `run_summary.md` write-up). `runs/` is gitignored — results are not committed
to git; each run's own directory is the source of truth.

Probing is **resumable**: every `(concept, method)` is checkpointed to
`metrics/probing_partial[_f1].jsonl`, so an interruption costs one probe. Pass
`--restart` to ignore a saved checkpoint.

## Historical: Mac PoC memory notes (16 GB Mac, retired 2026-07-29)

W1/W1.5 ran locally on Apple Silicon before moving to MN5. RAM, not compute, was the
local ceiling there: 10M activations = 6.4 GB fp16 was comfortable, 25M = 16 GB was
not. The code-level fix (stream to a memmap instead of accumulating in RAM) is a
permanent property of `ActivationStore`, not a Mac-only workaround, so it still holds
on cluster; only the MPS-cache-clearing habit was Mac-specific.
