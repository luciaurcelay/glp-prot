# glp_plm

Generative Latent Prior (GLP) ported from LLM activations to **protein-LM (ESM-2)**
activations, reusing InterPLM's released SAEs + Swiss-Prot pipeline. See top-level
`docs/PLAN.md` for the spec, `docs/INTERPLM_MAP.md` for how each piece maps to the
InterPLM / GLP reference code, and `SETUP.md` for environment setup.

## Status
Week 1 complete (Gate G1 passed): full data pipeline + Mac PoC run end-to-end. See the
§9 checklist in `docs/PLAN.md`.

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
  eval/                     # frechet (FD), pca_overlap  (loss_recovered = W2)
  utils/                    # device, seeds
configs/  poc_mac.yaml  esm8m_h100.yaml   # differ only in device/precision/scale/buffer
scripts/  one CLI per phase (see PLAN §11)
tests/    test_flow_matching_roundtrip.py  test_hook_parity.py
```

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
