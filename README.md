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
- **`docs/results/`** — committed results (runs/ itself is gitignored).
- **`glp_plm/README.md`** — the package layout.

## Submodules (ground truth over the papers)
- `InterPLM/` — ElanaPearl/InterPLM (SAEs, embedder, Swiss-Prot annotation, steering).
- `generative_latent_prior/` — g-luo/generative_latent_prior (the released GLP code).

## Status
W1 complete (**Gate G1 passed**), plus a mid-scale local run and most of W3 probing,
all on an Apple M4 — the ESM-2-8M spine turned out to be RAM-bound, not compute-bound.

**First result (Deliverable D2).** On 104 pre-registered Swiss-Prot concepts, 1-D
probing test AUC: **GLP 0.947** > raw-MLP 0.916 > SAE 0.900 > raw-layer 0.861 (all
paired-bootstrap significant). On InterPLM's domain-adjusted **F1** the order reverses:
**SAE 0.286** > GLP 0.231 — sparse features fire rarely but precisely, dense ones rank
better. The two metrics measure different things, and that complementarity is the
finding. Full write-up: [`docs/results/local_mid_summary.md`](docs/results/local_mid_summary.md).

Next: Gate G2 (SAE-recon FD + %Loss-Recovered), then W4 steering.

## Quick start
```bash
# see SETUP.md for the full environment
.venv/bin/python -m pytest tests/ -q
```
