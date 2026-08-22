# ESM-2-650M scaling experiment — plan

Status: **design frozen 2026-08-21**; not yet launched. Stretch / non-MVP (PLAN §D3),
hard-gated behind "W1–W3 clean". Scope confirmed with Lucia this session.

## 1. Objective

Test whether the GLP result ports from the toy PLM (ESM-2-8M, d=320) to a real-scale
PLM (ESM-2-650M, d=1280) — a **160× larger** model that is also the one InterPLM features
in its own main results. Two metrics, in priority order:

- **A — FD depth scan (PRIMARY, do first).** Replicate the `configs_lucia/` grid
  (`n_blocks` 3/6/12/24 over one fixed activation store, compared by Fréchet Distance) at
  650M. Echoes the GLP paper's own "loss/FD predicts utility" protocol. Deliverable: two
  FD-vs-`n_blocks` curves at a matched 100M-activation scale — 8M and 650M.
- **B — probing scaling point (if time before submission).** Meta-neuron 1-D probing AUC
  vs InterPLM SAE features at 650M, i.e. does *our* claim 1 survive at scale. Uses the
  released `Elana/InterPLM-esm2-650m` SAE at the same hooked layer.

## 2. Decisions (locked this session)

| Decision | Value | Rationale |
|---|---|---|
| Metric order | **A first, B if time** | A is the cheaper, self-contained scaling curve; B is the headline but needs the SAE + eval extraction |
| Activation scale | **100M** (not 500M) | 100M store = ~256 GB vs 1.28 TB; 5× cheaper train; disk-safe |
| Hooked layer | **18** (decided by pilot 2026-08-21) | Raw-layer probe on 166 concepts: L18 mean test AUC 0.8509 vs L24 0.8438, wins 107–59, paired Wilcoxon p=3e-4. See §4 stage 1. |
| Steering (Exp B whole-sequence) | **OUT of scope** | Separate claim with its own folding/contrast machinery; too big a lift |
| Training precision | **fp32** | Matches the 8M `configs_lucia` grid exactly (cleanest same-protocol comparison); store stays fp16. ~1B-param d24 still fits an 80 GB H100 in fp32. (Revisit to bf16 only if d24 training time becomes the bottleneck.) |

## 3. Fixed parameters

- **PLM:** `facebook/esm2_t33_650M_UR50D` (canonical `EsmForMaskedLM`, **never** `nv_esm`),
  d=1280, 33 layers. Released SAE layers: {1, 9, 18, 24, 30, 33}.
- **GLP (per depth cell):** d=1280, width=2560, MLP inner=5120.

  | n_blocks | ≈ params | meta-neurons (n_blocks·5120) |
  |---|---|---|
  | 3  | ~120M | 15,360 |
  | 6  | ~250M | 30,720 |
  | 12 | ~500M | 61,440 |
  | 24 | ~1B   | 122,880 |

  (Param counts approximate — the training script prints the true count; do not trust the
  estimate, per PLAN §6.2.)
- **Data:** UniRef50, 100M activations, batch 4096, 1 epoch, fp16 store.
- **Store size:** 100M × 1280 × 2 B ≈ **256 GB** (validated: the 8M/100M store is 60 GB at
  d=320; ×4 for d=1280).

## 4. Stages

### Stage 0 — inputs (local download → scp; all < 3 GB, MN5 has no internet)
- ESM-2-650M weights (~2.5 GB) — needed for all extraction.
- InterPLM-650m SAEs, layers **18 and 24** (~0.4 GB each) — only needed for metric B /
  pilot probing-vs-SAE; the raw-layer pilot (below) needs only ESM.
- **650M hook-parity test.** `tests/test_hook_parity.py` guards 8M/layer-4 only.
  Parametrize it for the chosen 650M layer and run it *before* the 256 GB extraction — this
  is the "hook parity is sacred" invariant and is expensive to get wrong.

### Stage 1 — layer pilot (decides the layer)
Extract the Swiss-Prot eval set at **layers 18 and 24** (two builds), run the 1-D
**raw-layer** probe on the qualifying concepts, and commit the 256 GB train store to
whichever layer makes concepts more linearly accessible. Labels are layer-independent so
the qualifying concept set is identical across the two builds — the comparison is paired
and needs no frozen-concept file. The layer-N eval store is also reusable for metric B.
Default if the pilot is inconclusive: **layer 24** (depth-parity with 8M layer-4).

**Operational facts (verified 2026-08-21):**
- **Use `N_PROT=20000`** — the only cached annotation TSV is
  `$GLP_DATA_ROOT/swissprot/swissprot_annotated_eval_20000.tsv`, and
  `fetch_uniprot_annotated_tsv` is a no-op when it exists (offline-safe on compute nodes).
  Any other n-prot would trigger a UniProt REST fetch that fails offline.
- **Account/QOS:** `--account=ehpc394 --qos=acc_ehpc` (crg562661; CLI overrides the
  scripts' `bsc70`/`acc_bsccs` directives). Available QOS: acc_ehpc, acc_debug, acc_interactive.
- **ESM-650M weights** live at `$HF_HOME/hub/models--facebook--esm2_t33_650M_UR50D`
  (scp'd from local; HF_HUB_OFFLINE=1 reads them). Hook-parity test passes for layers 18/24.
- **Cost:** each build re-runs the full ESM-650M forward over the 20k-protein eval set
  (~15–30 min GPU); two builds. (One-pass dual-layer extraction is possible but not worth
  the code change for a one-off pilot.)

**Launch (from the MN5 repo root, once weights are in place):**
```bash
cd /home/crg/crg562661/projects/glp-prot
for L in 18 24; do
  B=$(sbatch --account=ehpc394 --qos=acc_ehpc --time=04:00:00 \
        scripts/cluster/submit_build_swissprot_eval.sh configs/esm650m_pilot_l${L}.yaml 20000 30 \
      | awk '{print $NF}')
  sbatch --account=ehpc394 --qos=acc_ehpc --time=02:00:00 --export=ALL,METHODS=raw_layer \
        --dependency=afterok:$B scripts/cluster/submit_run_probing.sh configs/esm650m_pilot_l${L}.yaml
done
```
Decide by mean raw-layer AUC across qualifying concepts (paired over concepts), l18 vs l24.

### Stage 2 — extract 100M/650M train store at chosen layer (~4.5 GPU-hr, 256 GB) — DONE 2026-08-22
Store built and verified: `$GLP_DATA_ROOT/esm650m_100m_l18/train_store` —
`meta.json {"n":100000000,"d":1280,"dtype":"float16"}`, `activations.dat` = 256e9 bytes
(exactly 100M×1280×fp16), `rep_statistics.pt` + `train_manifest.json` present. The
launcher auto-excluded the 20k eval accessions (guardrail satisfied; 15,237 UniRef50
headers skipped). Reservoir-sampled 600k of 38.78M UniRef50 seqs, seed 0.

**Store-path gotcha (cost an rmdir once):** `extract_activations.py` writes to
`$GLP_DATA_ROOT/<output_dir basename>/train_store`, **ignoring** `data.store_dir` (which is
read only at *train* time). Running extraction with the d6 cell's config landed the store
at `esm650m_100m_4096_d6/train_store`; it was renamed to `esm650m_100m_l18/train_store` to
match the grid configs' `store_dir`. To avoid the rename next time, either extract with a
config whose `output_dir` basename equals the store name, or `mv` afterward.

### Stage 3 — depth scan (metric A): train d3/d6/d12/d24, FD/PCA each
Reuse the one store across all four cells (activations depend only on data/model/layer/seed,
not `n_blocks`). Per-cell: train → `submit_eval_generation` (FD, `fd_num_steps=1000`; PCA
overlap). **Not** `submit_pipeline.sh` (which would also probe). Watch-items in §6.

### Metric A results — COMPLETE (2026-08-22)

FD@1000 steps (all 8 cells sweep `[20,50,100,300,1000]`). **Absolute FD is not comparable
across PLMs** (different dims/scales: 650M FD~1347, floor 317.9; 8M FD~4.3, floor ~0.96) —
compare the **lower-bound ratio** (FD ÷ intrinsic floor).

| depth | 8M ratio | 650M ratio | Δ (8M−650M) |
|-------|----------|------------|-------------|
| d3    | 4.4667   | 4.2363     | +0.230 |
| d6    | 4.4701   | 4.2344     | +0.236 |
| d12   | 4.1477   | 3.9740     | +0.174 |
| d24   | 4.0579   | 3.8610     | +0.197 |

**The depth-scaling behavior ports cleanly from 8M to 650M:**
1. **Same curve shape** — flat d3→d6 (depth below 6 blocks doesn't help), then monotone
   improvement at d12/d24 at *both* scales.
2. **Same magnitude of depth benefit** — d6→d24 ratio drops 9.2% (8M) vs 8.8% (650M); the
   two curves are essentially parallel.
3. **650M uniformly better** — lower ratio at every depth (Δ≈0.17–0.24): the GLP fits the
   larger PLM's activation manifold at least as faithfully in relative terms.

650M train times (fp32, 1×H100): d3 0:47, d6 2:35 (I/O contention w/ d3), d12 ~4h, d24 6:42.

**Eval-time gotcha:** `submit_eval_generation.sh` defaults to `--time=02:00:00`; the d24 FD
sweep (generation to 1000 steps with the 24-block model, ~4× d6's per-step cost) exceeds 2h.
Its first run wrote the full, valid sweep and then TIMED OUT on a downstream step — the d24
numbers above are correct. Re-run deep cells' eval with `--time=10:00:00` if a clean exit is
needed. (d3/d6/d12 evals finished in ~18 min.)

### Stage 4 — compare (metric A deliverable)
Plot FD (and lower-bound ratio) vs `n_blocks` at fixed 1000 sampler steps, 650M curve
overlaid on the **already-complete 8M/100M curve**
(`uniref50_100m_4096_b{3,6,12,24}/metrics/fd_vs_steps.json`). `compare_runs.py` currently
keys its x-axis on `n_activations`; needs a small tweak to key on `n_blocks`.

### Stage 5 — probing (metric B, only if time)
Eval extraction with `--exclude-from <eval tsv>` (**never train on eval proteins**), the
650M eval store (4× bigger, still small absolute), download SAE at the hooked layer, then
1-D probing: GLP meta-neurons vs SAE features vs raw-layer/raw-MLP baselines. Control the
selection-pool confound (§6.5).

## 5. Resources & storage

| Item | Estimate |
|---|---|
| Train store (100M/650M) | **256 GB** on `$GLP_DATA_ROOT` (bsc70 scratch, ~35 TB free — GO) |
| Extraction | ~1–3 GPU-hr (H100, bf16) — soft estimate, throughput unmeasured at 650M |
| Train per cell | ~½–1 GPU-day (d6→d24 heavier) |
| Full depth scan (4 cells) | ~**1–1.5 GPU-days** compute at 100M + extraction |
| Downloads | ESM-650M ~2.5 GB + 1–2 SAEs ~0.8 GB (local→scp) |

Paths (shared with Dani `bsc070332`, per memory):
`GLP_DATA_ROOT=/gpfs/scratch/bsc70/bsc070332/glp-prot-data`,
`GLP_RUNS_ROOT=/gpfs/projects/bsc70/bsc070332/glp-prot-runs`. Use **distinct** output_dirs
(e.g. `runs/esm650m_100m_4096_d{3,6,12,24}`) — never reuse an existing run's leaf.

## 6. Watch-items (carry over from the 8M grid; worse at 650M)

1. **Hook parity** — verify the 650M layer parity test before extracting (Stage 0).
2. **FD stability at d=1280.** Fréchet needs a covariance over 1280 dims → rank-deficient
   unless eval generations ≫ 1280 (want ≥ 10–50k). Non-issue at d=320; a real sampling
   requirement here.
3. **Under-training.** The 24-block/~1B cell may not converge in one epoch — check its loss
   curve is flat before reading a weak FD as "depth doesn't help."
4. **GLP hyperparameter re-tune.** lr/batch are d=320/d=2048 heuristics (PLAN §6.2). Budget
   a quick lr sanity check at d=1280.
5. **Selection-pool confound (metric B).** Pool grows 15,360→122,880 across depths (and
   vs 7,680 at 8M); top-512 selection wins partly on pool size. Subsample to a common pool
   before any cross-depth or cross-scale probing comparison.
6. **VRAM.** ESM-650M (~2.5 GB) + GLP up to ~1B params on one 80 GB H100 — fits for
   extraction and bf16 training; confirm the d24 optimizer state fits.

## 7. Value & framing (honest)

Upside: generality of claim 1 at a real-size PLM, on InterPLM's home turf. Cost: it is
explicitly stretch/non-MVP and can pull focus off the core paper. The 8M scale-matched
baseline for metric A is **already complete**, which is what makes A cheap enough to be
worth doing first.

## 8. Still-open items

- Run the layer pilot vs commit to layer 24 outright.
- Confirm timeline slack against the workshop deadline before committing GPU-days.
