# Setup (Apple-Silicon PoC)

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

Figures and metrics are written under `runs/<name>/` (`figures/*.png|pdf`, `metrics/`).
