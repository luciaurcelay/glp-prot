# glp_plm

Generative Latent Prior (GLP) ported from LLM activations to **protein-LM (ESM-2)**
activations, reusing InterPLM's released SAEs + Swiss-Prot pipeline. See top-level
`docs/PLAN.md` for the spec and `RECON.md` for how each piece maps to the verified
InterPLM / GLP reference code.

## Status (this session)
Scaffold + model/diffusion + two load-bearing tests. Data pipeline, training at
scale, and eval are gated behind RECON review (PLAN §9).

## Layout
```
glp_plm/
  config.py                 # single dataclass config schema (YAML-loaded)
  data/activation_extractor.py   # THIN wrapper over interplm.embedders.esm.ESM (RECON §1)
  model/
    timestep_embed.py       # sinusoidal(t) -> MLP
    swiglu_block.py         # LayerNorm + SwiGLU + multiplicative t-gate; exposes meta-neuron
    glp.py                  # GLPDenoiser û_theta(z_t, t); meta_neurons() for probing
  diffusion/
    flow_matching.py        # z_t=(1-t)z0+t eps ; target=eps-z0 ; MSE
    sampler.py              # reverse Euler; sample / sample_on_manifold (SDEdit)
    scaler.py               # per-dim standardizer, stores {mean, var} (GLP-compatible)
  train/trainer.py          # flow-matching loop, cosine+warmup, grad clip, param-count print
  utils/                    # device, seeds
configs/  poc_mac.yaml  esm8m_h100.yaml   # differ only in device/precision/scale/buffer (D8)
scripts/  train_glp.py (real) + phase stubs (PLAN §11)
tests/    test_flow_matching_roundtrip.py  test_hook_parity.py
```

## Design points verified against the reference code (RECON.md)
- **Meta-neuron** = input to `down_proj` = `SiLU(gate_proj(x)·time_proj(t_emb))·up_proj(x)`.
- **LayerNorm** (not RMSNorm); time conditioning is **plain multiplicative** on the gate.
- **Width rule**: `d_model = 2·d`, `d_mlp = 4·d`; meta-neurons = `n_blocks · d_mlp`.
- **Hook parity**: extractor wraps InterPLM's embedder (already cls/eos-dropped); no re-strip.

## Run the tests
```bash
pip install -r requirements.txt
pip install -e InterPLM          # enables the hook-parity test (needs ESM-2-8M weights)
pytest tests/ -v
```
The hook-parity test SKIPS cleanly until InterPLM + ESM-2-8M weights are available.
