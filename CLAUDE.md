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
configs/            esm8m_h100.yaml — the single canonical config (6-blk/d=320, MN5/H100,
                    fp32/precache, real UniRef50 @ 100M acts). Historical Mac PoC configs
                    (poc_mac.yaml, local_clean.yaml, mid_mn5.yaml) are retired — see its
                    header comment and scripts/cluster/README.md "Config unification".
scripts/            _bootstrap.py + one CLI per phase (§11)
  cluster/          MN5 (BSC) SLURM launchers — see scripts/cluster/README.md
tests/              test_hook_parity.py (live), test_flow_matching_roundtrip.py
runs/               outputs (gitignored) — runs/<run>/{metrics,figures,run_summary.md}
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
```
The project moved to **cluster-only** (2026-07-29 — the local Mac PoC configs are
retired, see repo layout above); the venv above is for tests/local dev, not for running
the pipeline. On MN5 (BSC, account `bsc70`, `acc` partition = H100), use the SLURM
launchers in `scripts/cluster/` — default to `sbatch scripts/cluster/submit_pipeline.sh`
(one job, all seven stages, `python` calls sequentially — designed to work under the
`acc_debug` QOS's one-job-at-a-time limit; override `--qos`/`--time` for a real run past
smoke-test scale) rather than invoking `.venv/bin/python` directly. See
`scripts/cluster/README.md` for one-time setup (submodules first!) and known gaps
(UniRef50 true streaming stub; no steering launcher yet; login-node PYTHONHOME/quota/ulimit
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
  irreversible/destructive op. Small downloads (weights, small REST slices) are fine.
- **Git:** committing directly to `main` is fine for ordinary work; branch only for a large/
  significant new feature. Commit independently when it makes sense (a coherent unit of work
  is done) — you don't need to ask each time. Push only when asked. End commit messages with
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Cluster-only now:** W1/W1.5 developed/ran the PoC locally on Apple Silicon (MPS/CPU,
  fp32); that path is retired (2026-07-29) now that RAM/disk headroom is no longer the
  binding constraint — everything from W2 on runs on MN5 via `configs/esm8m_h100.yaml`
  and `scripts/cluster/`.

## Current status
W1 complete (**Gate G1 passed**) plus a mid-scale **local** run (`configs/local_clean.yaml`)
and most of W3 probing — see the PLAN §9 checklist and `runs/local_clean/run_summary.md`.

Headline: on 104 pre-registered concepts GLP meta-neurons lead 1-D probing AUC
(**0.945** vs raw-MLP 0.916, SAE 0.900, raw-layer 0.861 — all paired-significant). On
InterPLM-native **F1** the SAE is more precise (0.347 vs 0.241) and leads numerically
(0.286 vs 0.264), but **not significantly** at n=14; GLP beats both raw baselines there.
Don't overclaim either direction — extend F1 to all 104 concepts to settle it.

**Never train on eval proteins.** `extract_activations.py --exclude-from <eval tsv>` is
mandatory: the default REST queries overlap ~98%, which silently hands GLP activations
it will be probed on (measured effect was small, -0.002 AUC, but the SAE gets no such
advantage so the comparison is asymmetric).

**Still open (probing):** Gate G2 (SAE-reconstruction FD + %Loss-Recovered — both
local-feasible, not run), F1 on all 104 concepts.

### W4 steering — Exp B (diff-of-means × hydrophobicity) is measured
Canonical run: `runs/steering_crg/steering_diffmean_hydro_bidir_n40` (bidirectional, 40
held-out starts per direction, t_start 0.7, 1120 generations, all folded). Launch with
`scripts/cluster/submit_steering.sh B <UNIQUE-NAME> …` — the name is **required** (see
below), then `submit_fold.sh`, then `plot_steering_report.py` +
`analyze_steering_naturalness.py`.

Four findings that change how results must be reported:

1. **The correction moves the concept on its own.** At alpha=0 — no steering, the arms
   differ only by whether `glp_correct` ran — GLP-corrected gains **+1.15 ΔGRAVY** vs
   naive's +0.35, via Leu/Val enrichment to ~2× natural abundance. Never quote the raw
   corrected-vs-naive gap; quote `steering_attributable` (differenced per init against
   that arm's own alpha=0 run).
2. **That drift is a fixed hydrophobic bias, not mean-reversion.** Steering down from
   HIGH-GRAVY starts, naive flips sign (−0.28) while corrected stays positive (+0.54).
   The up-only design could not distinguish these.
3. **Both model-based naturalness metrics are gamed.** The sequences
   `is_degenerate_sequence` rejects (64% Leu+Val) score pLDDT 67 vs clean 54
   (p=3e-06) and pseudo-perplexity 7.3 vs 14.8 (p=7e-18) — i.e. *better* than our own
   output. pTM is merely indistinguishable (p=0.86). **The hand-tuned degenerate filter,
   not any metric, is what keeps junk out of the results** — its thresholds are the
   biggest unjustified assumption in the experiment. Check metrics on the REJECTED rows,
   never only the kept ones (a pppl-vs-pLDDT correlation computed on kept rows alone
   looks like mutual validation and is shared bias).
4. **Do not filter on pLDDT.** A pLDDT≥70 cut keeps 87% of the degenerate sequences and
   12% of the clean ones, taking a set from 8% to 38% junk. Use "fraction passing" as an
   *outcome* (`pass_rate_*` in summary.json), never as a selection step — the arms also
   survive at different rates, so filtering then comparing is survivor-biased.

Caveat for both arms: every clean generation is structurally mediocre (pTM ~0.28 vs 0.74
for a natural protein), and at alpha=0 the correction cuts confidently-folded yield ~3×.

**Steering runs need a unique `--exp-name`.** `RunLogger` redirects metrics/figures to
`$GLP_RUNS_ROOT/<basename(out_dir)>/`, keyed on the leaf alone, so the old fixed leaf made
every Exp B run silently clobber the previous one's metrics and figures.

**Figures state what is plotted, not what it means** — interpretation goes in the paper
caption. A claim baked into a PNG went stale against its own panel once already.

**`t_start` is 0.7 for Exp B** (0.5 for A, which is unmeasured here). 0.5-0.6 is a
pathological band: most drift, +11pp composition damage, and — the tell — the *best*
pseudo-perplexity. Switching to 0.7 cut degenerate output 26% → 7.7% and removed the
apparent "breakdown at alpha=5", which was a t_start artefact, not a property of steering.

**InterPLM needs a local patch to run steering at all** — `patches/`, applied by
`scripts/apply_interplm_patches.sh`; re-run it after any `git submodule update` (SETUP.md
step 0). Without it generation dies in `nnsight` with an `expand()` shape error.

5. **The generation loop, not GLP, costs foldability.** The 80 natural held-out starting
   proteins fold at pTM 0.373 (low-GRAVY set) / 0.629 (high-GRAVY set); the alpha=0 **naive**
   arm — no steering, no GLP — comes out at 0.275 / 0.319. Masked-Gibbs roughly halves
   foldability on its own and both arms land in the same place, so the naturalness claim
   fails for a reason unrelated to GLP. Two defensible operating points: 30 rounds/T=1.0
   (divergence +3.38 at alpha=5, pTM 0.275/0.319) and 5 rounds/T=0.5 (+1.56, pTM
   0.332/0.435, 6x cheaper). **Do not raise rounds at T=0.5** — it mode-collapses, and the
   collapsed sequences score HIGH pTM, so it looks like an improvement to any analysis that
   drops degenerate rows instead of counting them.
6. **The degeneracy gate is calibrated, and must not be tightened on complexity.** It
   rejects 3/80 (3.8%) of real proteins vs 2.9% of our output — not tuned against us.
   Complexity is affected by the treatment, so gating on it removes the successfully-steered
   sequences; SEG's standard 2.2-bit trigger would reject 75% of the natural starts. Quote
   effects **gated and ungated** — they agree to ~1%.

**Baseline discipline:** compare foldability against the held-out set the run actually
generates from (0.373 or 0.629), never the pooled 0.501 average.

**Still open (steering):** Exp A has never been re-run with any of the above; the
correction's own foldability cost is suggested by three marginal signals but unestablished;
we steer one layer / masked positions only where Huang et al. steer all layers and tokens
(the untested deviation); neither operating point reaches natural foldability. ESMFold2
folding is a separate stage in its own Python 3.12 venv (`submit_fold.sh`) because the
project venv has no `esm`.

Full record: `docs/steering_history.md` §13; current pipeline:
`docs/steering_methodology.md`.

## Local-scale gotchas (learned the hard way — don't regress these)
From the retired Mac PoC phase (dev Mac had **16 GB RAM**, RAM/disk — not compute — was
the ceiling); kept because the underlying code-level invariants (streaming, per-batch
standardization, checkpointing) are still load-bearing on cluster too, not Mac-only
workarounds — only the MPS-specific cache-clearing line below doesn't apply there.
- Never materialize a whole activation store: stores stream into memmaps and are read
  lazily; standardize **per batch**, not per split.
- Free feature banks between methods and call `torch.mps.empty_cache()` per concept;
  the MPS allocator cache otherwise grows until the OS kills the process.
- Long sweeps must checkpoint — `run_probing.py` resumes from
  `metrics/probing_partial[_f1].jsonl`.
- Quote FD **with its sampler step count** (`fd_num_steps=1000`); 20-step FD reports
  sampler error, not model quality.
