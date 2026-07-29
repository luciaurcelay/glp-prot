# glp_plm

Generative Latent Prior (GLP) ported from LLM activations to **protein-LM (ESM-2)**
activations, reusing InterPLM's released SAEs + Swiss-Prot pipeline. See top-level
`docs/PLAN.md` for the spec, `docs/INTERPLM_MAP.md` for how each piece maps to the
InterPLM / GLP reference code, and `SETUP.md` for environment setup.

## Status
W1 complete (Gate G1 passed) plus a mid-scale local run and most of W3 probing.
Results: `runs/local_clean/run_summary.md` (gitignored — see CLAUDE.md repo layout); checklist: PLAN §9.

## Layout
```
glp_plm/
  config.py                 # single dataclass config schema (YAML-loaded)
  data/
    activation_extractor.py # THIN wrapper over interplm.embedders.esm.ESM (hook parity)
    activation_store.py     # fp16 memmap shards + Scaler persistence
    swissprot_labels.py     # UniProt REST fetch + InterPLM annotation orchestration
    producer_consumer.py    # streaming buffer (W2/cluster stub)
  model/
    timestep_embed.py       # sinusoidal(t) -> MLP
    swiglu_block.py         # LayerNorm + SwiGLU + multiplicative t-gate; exposes meta-neuron
    glp.py                  # GLPDenoiser u(z_t, t); meta_neurons() for probing
  diffusion/
    flow_matching.py        # z_t=(1-t)z0+t eps ; target=eps-z0 ; MSE
    sampler.py              # reverse Euler; sample / sample_on_manifold (SDEdit)
    scaler.py               # per-dim standardizer, stores {mean, var} (GLP-compatible)
  train/trainer.py          # flow-matching loop, cosine+warmup, grad clip, param-count print
  eval/
    frechet.py              # Frechet distance (+ real-vs-real lower bound)
    pca_overlap.py          # top-2 PCA, real vs generated
    meta_neurons.py         # GLP meta-neuron bank at probing u (u = 1 - t)
    baselines.py            # SAE / raw-layer / raw-MLP (ESM FFN hook) banks
    probing_1d.py           # GLP AUC protocol: top-512 -> 1-D LogRegCV -> val/test
    probing_f1.py           # InterPLM domain-adjusted F1, rate-matched two-sided
    concept_selection.py    # pre-registered §7 concept freeze
                            # loss_recovered = W2, deferred
  experiments/              # RunLogger (metrics) + figure registry + report (see below)
  utils/                    # device, seeds
configs/  esm8m_h100.yaml (single canonical config — MN5/H100, 6-blk/d=320, real UniRef50 @ 100M acts)
scripts/  one CLI per phase (see PLAN §11) + make_report.py
tests/    test_flow_matching_roundtrip.py  test_hook_parity.py
```

## Figures & metrics (`glp_plm/experiments/`)
Compute logs structured metrics; plotting reads them. Running the pipeline auto-renders
figures to `runs/<name>/figures/` (train → `training_loss`, `lr_schedule`; eval →
`generation_pca`, `generation_fd`). A figure is one registered function that declares the
metric artifacts it needs, so `make_report` renders whatever the current phase produced —
new figures (W2 `%LR`, W3 probing AUC, W4 steering Pareto) slot in without touching the
runner. Re-render anytime: `python scripts/make_report.py --config configs/esm8m_h100.yaml`.

## Design points (verified against the reference code — see `docs/INTERPLM_MAP.md`)
- **Meta-neuron** = input to `down_proj` = `SiLU(gate_proj(x)·time_proj(t_emb))·up_proj(x)`.
- **LayerNorm** (not RMSNorm); time conditioning is **plain multiplicative** on the gate.
- **Width rule**: `d_model = 2·d`, `d_mlp = 4·d`; meta-neurons = `n_blocks · d_mlp`.
- **Hook parity**: extractor wraps InterPLM's embedder (already cls/eos-dropped); no re-strip.

## Run
Environment + full pipeline commands are in `SETUP.md`. Quick test:
```bash
.venv/bin/python -m pytest tests/ -q     # 6 pass (hook-parity is live once weights are cached)
```
