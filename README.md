# GLP-Prot

Porting the **Generative Latent Prior (GLP)** — a diffusion "meta-model" of neural
activations — from LLM activations to **protein-LM (ESM-2) activations**, reusing
**InterPLM**'s released SAEs + Swiss-Prot pipeline as the substrate. Target: a NeurIPS
2026 interpretability-for-discovery workshop paper.

## Where to look
- **`CLAUDE.md`** — start here (onboarding: what this is, layout, how to run, guardrails).
- **`docs/PLAN.md`** — the authoritative spec (thesis, decisions, week-by-week checklist).
- **`docs/INTERPLM_MAP.md`** — reuse map: what to call in the InterPLM/GLP clones (`file:line`) + gotchas.
- **`SETUP.md`** — Python 3.11 env + how to run the pipeline.
- **`glp_plm/README.md`** — the package layout.

## Submodules (ground truth over the papers)
- `InterPLM/` — ElanaPearl/InterPLM (SAEs, embedder, Swiss-Prot annotation, steering).
- `generative_latent_prior/` — g-luo/generative_latent_prior (the released GLP code).

## Status
Week 1 complete — Mac proof-of-concept runs end-to-end and **Gate G1 passed**
(hook parity live; GLP trains; generated activations overlap real). See the §9
checklist in `docs/PLAN.md`. Next: W2 cluster scale-up → Gate G2.

## Quick start
```bash
# see SETUP.md for the full environment
.venv/bin/python -m pytest tests/ -q
```
