# Build the ESMFold2 Apptainer container on MN5 (agent instructions)

**Audience:** a Claude Code agent running on Dani's MN5 login node — the one with
**full internet + Apptainer + no GitHub block**. Lucia's login node has *no* internet, so
she cannot build this; you (Dani's side) own the build. The two nodes share the same
`/gpfs` filesystems, so **build directly onto shared storage — no file transfer to
anyone is needed.**

## Goal / deliverable

A working, **fully offline-runnable** ESMFold2 Apptainer image on MN5, usable from GPU
(`acc`/H100) compute nodes that have **no internet**, plus its pre-fetched model weights.
Everything lives under (durable, non-purged `projects`, NOT scratch):

```
/gpfs/projects/bsc70/bsc070332/containers/esmfold2/
  esmfold2.sif        # the Apptainer image (code + deps only; weights are bind-mounted)
  weights/            # Hugging Face cache of all repos the fold path loads
  esmfold2.def        # the definition file you write (kept for reproducibility)
  run_fold.sh         # thin run wrapper (offline: HF_HUB_OFFLINE=1, --nv)
  NOTES.md            # what you built: base image, versions, HF repos prefetched, run cmd
```

## Hard constraints (do not violate)

1. **Runtime is offline.** The GPU compute nodes have no internet. The container must fold
   with `HF_HUB_OFFLINE=1` reading a **local** HF cache. Any `from_pretrained(...)` that
   would hit the network at run time is a failure — all weights must be pre-fetched into
   `weights/` at build time (you have internet; the compute nodes do not).
2. **Apptainer, not Docker.** Deliverable is a `.sif`. Build with `apptainer build
   --fakeroot` (unprivileged build on a login node).
3. **Location = `projects`, not `scratch`.** Scratch is periodically purged; a rebuild is
   expensive (offline elsewhere). Put the `.sif` + weights in `projects` as above.
4. **CUDA must match MN5.** MN5's GPU stack is CUDA 12.x (the project uses `cuda/12.4`) on
   H100 nodes; `apptainer exec --nv` injects the host driver. Build torch for **CUDA 12.4**
   (`cu124` wheels). Do not fold on the login node (no GPU) — GPU verification is Lucia's
   step via `sbatch` on `acc`.

## Package facts (from the repo, verify against it as you go)

- Repo: `https://github.com/Biohub/esm` (package `esm`, v3.3.0). Install:
  `pip install "esm@git+https://github.com/Biohub/esm.git@main"`.
- **Python 3.12 only** (`>=3.12,<3.13`).
- `torch>=2.2`, `numpy>=2.0,<2.4`, and deps: `biotite`, `einops`, `rdkit`, `biopython`,
  `scikit-learn`, `boto3`, `accelerate`, plus a `dockq` fork.
- **`transformers` is a Biohub git fork** (`git+https://github.com/Biohub/transformers.git@main`),
  not PyPI — pip resolves it automatically from the `esm` install (needs GitHub at build
  time, which you have).
- Weights on Hugging Face: `biohub/ESMFold2` (structure model). ESMFold2 likely also loads
  an **ESMC language model** at fold time (e.g. a `biohub/ESMC-*` repo) — confirm from the
  repo's fold example and prefetch **every** HF repo id the fold path touches (Step 4).
- Fold API (from the README):
  ```python
  from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
  model = ESMFold2Model.from_pretrained("biohub/ESMFold2").cuda()
  # ESMFold2InputBuilder().fold(model, structure_input, num_sampling_steps=100)
  ```

## Step 1 — set up paths and clone

```bash
export ROOT=/gpfs/projects/bsc70/bsc070332/containers/esmfold2
mkdir -p "$ROOT/weights"
cd "$ROOT"
git clone https://github.com/Biohub/esm.git esm-src        # for reading its docs/fold example
module load apptainer 2>/dev/null || module load singularity   # whichever MN5 provides
apptainer --version
```

Read `esm-src`'s README and any `examples/`/fold scripts to confirm: the exact fold entry
point, and **which HF repo ids** are loaded (ESMFold2 + any ESMC LM). You will prefetch all
of them in Step 4.

## Step 2 — write `esmfold2.def`

Start from this template (Ubuntu 24.04 ships Python 3.12 natively; a venv avoids PEP-668).
Adjust if the build errors (e.g. extra system libs for `rdkit`/`biotite`, or a different
available CUDA-12.4 base tag):

```
Bootstrap: docker
From: nvidia/cuda:12.4.1-cudnn-runtime-ubuntu24.04

%post
    export DEBIAN_FRONTEND=noninteractive
    apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip git build-essential \
        libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
    python3.12 -m venv /opt/esm
    /opt/esm/bin/pip install --no-cache-dir --upgrade pip wheel
    # torch for CUDA 12.4 to match MN5's H100 driver
    /opt/esm/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124
    # ESMFold2 — pulls the Biohub/transformers fork + all deps from GitHub/PyPI (needs
    # internet at BUILD time, which this node has)
    /opt/esm/bin/pip install --no-cache-dir "esm@git+https://github.com/Biohub/esm.git@main"
    # smoke-import (CPU, no weights): fails the build early if the stack is broken
    /opt/esm/bin/python -c "import torch, transformers; from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model; print('import ok', torch.__version__)"

%environment
    export PATH=/opt/esm/bin:$PATH
    export PYTHONNOUSERSITE=1
    # NOTE: HF_HOME and HF_HUB_OFFLINE are set at RUN time (Step 4 / run_fold.sh), not baked
    # in, so the same image can download (build node) and run offline (compute node).

%runscript
    exec python "$@"
```

## Step 3 — build the image

```bash
cd "$ROOT"
apptainer build --fakeroot esmfold2.sif esmfold2.def
```

If `--fakeroot` is unavailable on this login node, try `apptainer build` with the
`APPTAINER_TMPDIR` on a large filesystem, or a `--sandbox` build then `apptainer build
esmfold2.sif sandbox/`. The build downloads several GB (base image + torch + weights of the
fork's deps) — run it on a filesystem with room.

## Step 4 — pre-fetch ALL weights into the offline cache (you have internet here)

Download every HF repo the fold path loads into `$ROOT/weights` (an HF cache). Run inside
the container so versions match, with `HF_HUB_OFFLINE=0` (override) and `HF_HOME` pointed at
the cache. At minimum `biohub/ESMFold2`; **add the ESMC LM repo id(s) you found in Step 1**:

```bash
apptainer exec \
  --bind "$ROOT/weights":/weights \
  --env HF_HOME=/weights --env HF_HUB_OFFLINE=0 \
  "$ROOT/esmfold2.sif" python - <<'PY'
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
ESMFold2Model.from_pretrained("biohub/ESMFold2")   # downloads to /weights cache (CPU, no .cuda())
# TODO: also load / download any ESMC language model + tokenizer the fold example uses, e.g.:
# from transformers import AutoModel; AutoModel.from_pretrained("biohub/ESMC-<size>")
print("weights prefetched")
PY
du -sh "$ROOT/weights"
```

The most robust check that you caught *everything*: after this, re-run the same load with
`--env HF_HUB_OFFLINE=1` (below). If it loads with no network, the cache is complete; if it
errors trying to reach HF, a repo is missing — add it and re-prefetch.

## Step 5 — offline sanity check (CPU, on this login node)

```bash
apptainer exec \
  --bind "$ROOT/weights":/weights \
  --env HF_HOME=/weights --env HF_HUB_OFFLINE=1 \
  "$ROOT/esmfold2.sif" python -c "
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
m = ESMFold2Model.from_pretrained('biohub/ESMFold2')   # must succeed with NO network
print('offline load ok:', type(m).__name__)"
```

This proves the container + cache are offline-complete. (It does **not** fold — folding needs
a GPU and is Lucia's `sbatch` verification step.)

## Step 6 — write the run wrapper + NOTES, then hand off

Write `$ROOT/run_fold.sh` (the command the compute node will use):

```bash
#!/bin/bash
# Usage: run_fold.sh <input.fasta> <out_dir>   (submit inside an acc/H100 sbatch job)
set -euo pipefail
ROOT=/gpfs/projects/bsc70/bsc070332/containers/esmfold2
apptainer exec --nv \
  --bind "$ROOT/weights":/weights \
  --env HF_HOME=/weights --env HF_HUB_OFFLINE=1 \
  "$ROOT/esmfold2.sif" python /gpfs/projects/bsc70/bsc070332/containers/esmfold2/fold.py "$@"
```

(You do not need to write `fold.py` — Lucia's side will, from the repo's fold example, when
integrating pLDDT scoring. Just leave the wrapper + a note pointing at the fold API.)

Write `$ROOT/NOTES.md` recording: the exact base image tag used, `esm`/`torch`/`transformers`
commit or version actually installed (`apptainer exec esmfold2.sif pip freeze | grep -Ei
'esm|torch|transformers'`), the full list of HF repo ids prefetched into `weights/`, the
`weights/` size, and the exact `apptainer exec --nv ...` run command.

## Success criteria (all must hold before handoff)

- [ ] `esmfold2.sif` exists in `/gpfs/projects/bsc70/bsc070332/containers/esmfold2/`.
- [ ] Step 5 passes: `ESMFold2Model.from_pretrained` loads with `HF_HUB_OFFLINE=1` and no network.
- [ ] `weights/` contains every HF repo the fold path loads (confirmed by the offline load).
- [ ] `NOTES.md` records versions + prefetched repos + the run command.

## Handoff to Lucia's side (do not do these — no GPU on the login node)

- GPU verification: `sbatch` on `acc` (H100) running `run_fold.sh` on a short test sequence,
  confirming a structure + pLDDT come out and CUDA works via `--nv`.
- Integration: a `fold.py` that folds sequences and emits pLDDT, wired into the
  steering/generation work as the foldability metric.
