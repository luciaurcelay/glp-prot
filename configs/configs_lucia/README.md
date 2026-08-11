# configs_lucia — GLP depth-scaling grid (500M activations, batch 4096)

A depth scan of the GLP diffusion model over the same 500M-activation UniRef50 data,
following the GLP paper's own protocol of comparing architectures by **Fréchet Distance
(FD)**. Only `n_blocks` varies; everything else is fixed.

| config | n_blocks | params | meta-neurons | output_dir | status |
|---|---|---|---|---|---|
| `esm8m_500m_4096_d3.yaml`  | 3  | ~11.1M | 3,840  | `runs/uniref50_500m_4096_d3`  | to run |
| `esm8m_500m_4096_d6.yaml`  | 6  | ~20.9M | 7,680  | `runs/uniref50_500m_4096`     | **done** |
| `esm8m_500m_4096_d12.yaml` | 12 | ~40.6M | 15,360 | `runs/uniref50_500m_4096_d12` | to run |
| `esm8m_500m_4096_d24.yaml` | 24 | ~80.0M | 30,720 | `runs/uniref50_500m_4096_d24` | to run |

The 6-block cell is the already-completed `runs/uniref50_500m_4096`; its config points
there for grid completeness — **do not relaunch training on it** (see the file's header).

## Reuse (no re-extraction)

Every cell sets `data.store_dir: uniref50_500m/train_store`, so all cells read the one
already-extracted 320GB activation store — nothing re-extracts. The activations depend
only on the data (model/layer/seqs/seed/count), not on `n_blocks`, so sharing is exact.

## Launching (FD-only)

Each cell is train → FD/PCA, via the individual launchers (not `submit_pipeline.sh`, which
would also probe). Both stages read `store_dir`. From the repo root on MN5, for each of
d3/d12/d24 (crg562661 must override account/QOS — see scripts/cluster/README.md):

```bash
CFG=configs/configs_lucia/esm8m_500m_4096_d24.yaml   # etc.
TRAIN=$(sbatch --account=ehpc394 --qos=acc_ehpc --time=1-12:00:00 \
    scripts/cluster/submit_train_glp.sh "$CFG" | awk '{print $NF}')
sbatch --account=ehpc394 --qos=acc_ehpc --dependency=afterok:$TRAIN \
    scripts/cluster/submit_eval_generation.sh "$CFG"
```

The 6-block run finished in ~2.6h; deeper cells cost more compute but the same one-epoch
data budget. Compare cells with `scripts/compare_runs.py` (FD ratio — see below).

## Reading the comparison

- Compare **FD/lower-bound ratio** at a fixed sampler-step count (1000), within this
  fixed 500M data scale. (`compare_runs.py` currently keys its x-axis on `n_activations`;
  it needs a small tweak to key on `n_blocks` for the depth plot.)
- Two caveats if you later switch this grid from FD to **probing**:
  1. **Selection-pool confound** — deeper models have more meta-neurons (3,840 → 30,720),
     so the top-512-candidate filter selects from a bigger pool and wins partly on pool
     size, not per-unit quality. Control by subsampling to a common pool before probing.
  2. **Under-training** — the 24-block cell may not converge in one epoch; check its loss
     curve is flat before trusting a weak result as "depth doesn't help."
- Replicating this grid at **100M** (a second data scale) turns the 1-D depth slice into
  a depth × data surface — the actual scaling result.
