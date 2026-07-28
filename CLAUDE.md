# CLAUDE.md — agent onboarding for `glp-prot`

Read this first. It tells a new agent what this repo is, how it's laid out, how to run
things, and the non-negotiable rules. For the full spec read `docs/PLAN.md`; for what to
reuse from the InterPLM/GLP clones (with `file:line` pointers) read `docs/INTERPLM_MAP.md`;
for environment setup read `SETUP.md`.

## What this project is
Porting the **Generative Latent Prior (GLP)** — a diffusion "meta-model" of neural
activations — from LLM activations to **protein-LM (ESM-2) activations**, reusing
**InterPLM**'s released SAEs + Swiss-Prot pipeline as the substrate. Target: a NeurIPS
2026 workshop paper. Two claims: (1) GLP "meta-neurons" isolate biological concepts at
least as cleanly as InterPLM's SAE features (1-D probing on Swiss-Prot); (2) GLP as an
on-manifold prior makes SAE concept-steering produce more natural proteins.

## Ground truth (trust these over priors)
- `docs/PLAN.md` — authoritative spec (single working version, no changelog). §9 is a
  **progress checklist** — read it to see what's done.
- `docs/INTERPLM_MAP.md` — the reuse map: where to find / what to call in the InterPLM and
  GLP clones (with `file:line`), plus the live gotchas. Consult before reimplementing anything.
- The two clones are **git submodules** and are ground truth over the papers:
  - `InterPLM/` (ElanaPearl/InterPLM) — SAEs, embedder, Swiss-Prot annotation, steering.
  - `generative_latent_prior/` (g-luo/generative_latent_prior) — the released GLP code.
- Reference PDFs are in `docs/papers/` (gitignored, local only).

## Repo layout
```
glp_plm/            our package
  config.py         single dataclass config schema (YAML-loaded)
  data/             activation_extractor (wraps InterPLM), activation_store (memmap+scaler),
                    swissprot_labels (annotation orchestration + REST fetch), producer_consumer (W2 stub)
  model/            swiglu_block (LayerNorm + t-modulated gate; exposes meta-neuron),
                    glp (GLPDenoiser + meta_neurons()), timestep_embed
  diffusion/        flow_matching, sampler (Euler; sample / sample_on_manifold), scaler
  train/            trainer (flow-matching loop, cosine+warmup)
  eval/             frechet, pca_overlap, meta_neurons, baselines, probing_1d,
                    probing_f1, concept_selection   (loss_recovered = W2, deferred)
  experiments/      RunLogger (metrics) + figure registry + make_report (auto-renders figures)
configs/            poc_mac.yaml (2-blk/2M) · local_clean.yaml (6-blk/10M, CANONICAL) ·
                    esm8m_h100.yaml (cluster; differs only in device/precision/scale/buffer)
scripts/            _bootstrap.py + one CLI per phase (§11)
  cluster/          MN5 (BSC) SLURM launchers — see scripts/cluster/README.md
tests/              test_hook_parity.py (live), test_flow_matching_roundtrip.py
docs/results/       committed metrics + the run write-up
runs/               outputs (gitignored)
```

## Environment (see SETUP.md)
- InterPLM needs **Python 3.11**; use the repo `.venv` (`.venv/bin/python`).
- **Do NOT `pip install -e InterPLM`** — its `setup.py` misses `interplm/train/__init__.py`
  so a non-editable install drops `interplm.train`, and editable `.pth` handling is flaky
  here (Homebrew Python ships a shadowing `sitecustomize`). Instead: scripts import
  `scripts/_bootstrap.py` first, tests use `tests/conftest.py`; both put the repo root and
  `InterPLM/` on `sys.path`. Ad-hoc: `PYTHONPATH="$(pwd):$(pwd)/InterPLM" .venv/bin/python …`.

## Running things
```bash
.venv/bin/python -m pytest tests/ -q                                   # 6 pass (hook-parity live)
.venv/bin/python scripts/download_interplm_saes.py --config configs/poc_mac.yaml
.venv/bin/python scripts/extract_activations.py    --config configs/poc_mac.yaml --store train
.venv/bin/python scripts/build_swissprot_eval.py   --config configs/poc_mac.yaml
.venv/bin/python scripts/train_glp.py              --config configs/poc_mac.yaml
.venv/bin/python scripts/eval_generation.py        --config configs/poc_mac.yaml   # Gate G1
```
Cluster (W2) is a config swap to `esm8m_h100.yaml` — no code changes (D8 parity contract).
On MN5 (BSC, account `bsc70`, `acc` partition = H100), use the SLURM launchers in
`scripts/cluster/` instead of invoking `.venv/bin/python` directly — see
`scripts/cluster/README.md` for one-time setup (submodules first!) and known gaps
(UniRef50 streaming stub; no steering launcher yet; login-node PYTHONHOME/quota/ulimit
gotchas — all written up there, don't re-derive).

## Non-negotiable guardrails (expensive to get wrong)
1. **Hook parity is sacred.** GLP trains on byte-identical activations to what the SAE saw:
   residual stream after ESM-2-8M block 4 (`hidden_states[4]`), cls+eos dropped. Always go
   through `interplm.embedders.esm.ESM` (the extractor wraps it) — never reimplement extraction.
   `tests/test_hook_parity.py` guards this.
2. **Do not retrain SAEs.** Download `Elana/InterPLM-esm2-8m` via `load_sae_from_hf`.
3. **Canonical `facebook/esm2_t6_8M_UR50D`** (EsmModel/EsmForMaskedLM). Never the NVIDIA
   `nv_esm` repackaging (breaks hook parity).
4. **Meta-neuron** = input to `down_proj` = `SiLU(gate_proj(x)·time_proj(t_emb))·up_proj(x)`
   (post-gating, includes the time factor). Block uses **LayerNorm**, and time
   conditioning is **plain multiplicative** on the gate (no (1+γ)/β).

## Working agreement (how to operate here)
- **Verify before you build.** Check claims against the actual clone files and cite
  `file:line`; prefer reusing InterPLM code over reimplementing.
- **Ask before:** any large download (UniRef50 ~12 GB), any large/long training run, or any
  irreversible/destructive op. Small downloads (weights, small REST slices) and the Mac PoC
  are fine.
- **Git:** committing directly to `main` is fine for ordinary work; branch only for a large/
  significant new feature. Commit independently when it makes sense (a coherent unit of work
  is done) — you don't need to ask each time. Push only when asked. End commit messages with
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Local-first:** develop/run the PoC locally on Apple Silicon (MPS/CPU, fp32), then move to
  H100 by swapping config only.

## Current status
W1 complete (**Gate G1 passed**) plus a mid-scale **local** run (`configs/local_clean.yaml`)
and most of W3 probing — see the PLAN §9 checklist and `docs/results/run_summary.md`.

Headline: on 104 pre-registered concepts GLP meta-neurons lead 1-D probing AUC
(**0.945** vs raw-MLP 0.916, SAE 0.900, raw-layer 0.861 — all paired-significant). On
InterPLM-native **F1** the SAE is more precise (0.347 vs 0.241) and leads numerically
(0.286 vs 0.264), but **not significantly** at n=14; GLP beats both raw baselines there.
Don't overclaim either direction — extend F1 to all 104 concepts to settle it.

**Never train on eval proteins.** `extract_activations.py --exclude-from <eval tsv>` is
mandatory: the default REST queries overlap ~98%, which silently hands GLP activations
it will be probed on (measured effect was small, -0.002 AUC, but the SAE gets no such
advantage so the comparison is asymmetric).

**Still open:** Gate G2 (SAE-reconstruction FD + %Loss-Recovered — both local-feasible,
not run), F1 on all 104 concepts, W4 steering, cluster scale-up.

## Local-scale gotchas (learned the hard way — don't regress these)
The dev Mac has **16 GB RAM**, and RAM/disk — not compute — is the ceiling.
- Never materialize a whole activation store: stores stream into memmaps and are read
  lazily; standardize **per batch**, not per split.
- Free feature banks between methods and call `torch.mps.empty_cache()` per concept;
  the MPS allocator cache otherwise grows until the OS kills the process.
- Long sweeps must checkpoint — `run_probing.py` resumes from
  `metrics/probing_partial[_f1].jsonl`.
- Quote FD **with its sampler step count** (`fd_num_steps=1000`); 20-step FD reports
  sampler error, not model quality.
