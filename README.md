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
- **`runs/<run>/run_summary.md`** — per-run write-up (`runs/` is gitignored, not committed).
- **`glp_plm/README.md`** — the package layout.

## Submodules (ground truth over the papers)
- `InterPLM/` — ElanaPearl/InterPLM (SAEs, embedder, Swiss-Prot annotation, steering).
- `generative_latent_prior/` — g-luo/generative_latent_prior (the released GLP code).

## Status
W1 complete (**Gate G1 passed**), plus a mid-scale local run and most of W3 probing,
all on an Apple M4 — the ESM-2-8M spine turned out to be RAM-bound, not compute-bound.

**First result (Deliverable D2).** On 104 pre-registered Swiss-Prot concepts, 1-D
probing test AUC: **GLP 0.945** > raw-MLP 0.916 > SAE 0.900 > raw-layer 0.861 — all
paired-bootstrap significant. On InterPLM's domain-adjusted **F1** the SAE is more
*precise* (0.347 vs 0.241) and leads numerically (0.286 vs 0.264), though that gap is
**not** significant at n=14; GLP significantly beats both raw baselines there too.
Sparse features fire rarely but correctly; dense ones rank better.
Full write-up: `runs/local_clean/run_summary.md` (gitignored — not on GitHub, local/cluster only).

Training excludes every eval protein (verified 0 overlap) — an earlier run overlapped
98.4%, which turned out to be worth only −0.002 AUC.

Next: Gate G2 (SAE-recon FD + %Loss-Recovered), then W4 steering.

## Quick start
```bash
# see SETUP.md for the full environment
.venv/bin/python -m pytest tests/ -q
```
