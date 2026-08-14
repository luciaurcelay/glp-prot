# On-manifold steering — methodology (current)

What `scripts/run_steering.py` does today, in plain terms. No results, no
history of what changed and why — for that, see `docs/steering_history.md`.
Tracks PLAN.md §6.6/W4; run via `scripts/cluster/submit_steering.sh <A|B> ...`.

## The idea being tested

"Steering" a protein language model means pushing its internal activations
toward some target concept (e.g. "more zinc-finger-like", "more hydrophobic")
and seeing what sequence comes out. Pushed too hard, the activations end up
somewhere the model never saw during training — off the manifold of real
protein activations — and the output degrades (garbled sequences, or
sequences that don't actually carry the intended property).

GLP is a diffusion model trained to recognize what "on-manifold" activations
look like. The thesis: after steering pushes an activation off-manifold, using
GLP to pull it back on-manifold (a cleanup step) should let you reach a
stronger concept effect at a given level of naturalness than steering alone —
or equivalently, more natural output at a given concept strength. Every part
of the pipeline below exists to test that one claim, on two different kinds of
steering.

## The two experiments

**Experiment A** — SAE-feature steering on the zinc-finger concept
(`--experiment A`). Uses InterPLM's released sparse autoencoder (SAE) on
ESM-2-8M layer-4 activations. A zinc-finger-associated SAE feature is pushed
up, either by:
- `clamp`: hard-set the feature to a fixed target value, or
- `scale_shift`: scale and shift whatever value the feature already has
  (`z* = a·z + b`).

Success is judged by a strict regex for the C2H2 zinc-finger motif
(`C.{2,4}C.{12}H.{3,5}H`) in the generated sequence, plus ESMFold pLDDT as a
secondary structural-naturalness check.

**Experiment B** — diff-of-means steering on hydrophobicity
(`--experiment B`). A single direction vector `v` is built from the difference
between high-GRAVY and low-GRAVY real sequences' activations (GRAVY = Grand
Average of Hydropathy — higher means more hydrophobic). Steering adds a
multiple of that vector to the activation and renormalizes back to the
original vector length: `h' = renorm(h + alpha·v)`.

Success is judged by ΔGRAVY: how much more hydrophobic the finished sequence
is than the real, held-out low-GRAVY sequence it started from.

These two are deliberately not run identically (different init, different
masking schedule, different selection rule) — each follows its own source
paper's protocol rather than a shared generic recipe. See the module
docstring in `run_steering.py` for the exact citations.

## The shared pipeline

**Part 0 — off-manifold diagnostic (no sequence generation).** Take real
held-out activations, apply the steering intervention at increasing `alpha`,
and measure GLP's own diffusion loss on the result. Loss should rise with
`alpha` — confirms steering is actually pushing off-manifold before trusting
anything generation-based.

**Part 1 — generation loop** (`masked_gibbs_generate`). Start from an
initial sequence (random amino acids for Exp A, a real held-out sequence for
Exp B), then repeat for `n_rounds`: mask a random fraction of positions
(`mask_fraction`) → run the model forward → apply the steering intervention to
those positions' activations → **optionally** run GLP's cleanup step
(`glp_correct`, see below) on the steered activations → run the model forward
again with that override → sample new amino acids for the masked positions.
Every alpha is run twice: once with the cleanup step skipped ("naive"
steering) and once with it applied ("GLP-corrected") — that pairing is the
whole comparison.

**`mask_fraction` defaults differ sharply between the two experiments, and
that's not an incidental knob difference.** Exp A defaults to
`mask_fraction=1.0` — every round re-masks and re-samples the *entire*
sequence from scratch, for 100 rounds (`plm-sae`'s own protocol: full
resample each round, keep the round that maximizes the clamped feature's
activation). Exp B defaults to `mask_fraction=0.10` — only ~10% of positions
are touched per round, so each round is a small edit on top of the previous
round's output, for 30 rounds (Huang et al.'s partial-refinement protocol,
starting from a real sequence rather than random noise). So "the generation
loop" is shared machinery, but Exp A's rounds are independent full
re-generations under a steered feature, while Exp B's rounds are incremental,
compounding edits — that difference is exactly why Exp B's round-loop can
bootstrap a mild per-round bias into the full sequence-collapse described in
`docs/steering_history.md` §10, in a way Exp A's full-resample rounds don't.

**Which round is "the sequence"?** After `n_rounds`, the loop doesn't just
return the last one. Exp A scores every round by the target SAE feature's own
activation on the sequence decoded so far, and returns the best-scoring round
(`select_fn`, arXiv:2502.09135's z_k-argmax criterion) — *except* its
unsteered anchor arm (`alpha` = the no-op value), which keeps the last round
instead, since best-of-100 selection on a run that was never steered would
just cherry-pick a lucky-looking round and inflate the baseline. Exp B never
selects — every alpha, both naive and GLP-corrected, always keeps the last of
its 30 rounds.

**The cleanup step (`glp_correct`) — SDEdit.** GLP is a diffusion model: it
learns to turn random noise into a realistic activation over many small
steps, going from a "noise level" of 1 down to 0. `glp_correct` takes the
steered (off-manifold) activation, mixes it with random noise up to a chosen
noise level `t_start`, then runs GLP's reverse process from `t_start` back
down to 0:

```
z_noisy = (1 - t_start) * z_edit + t_start * eps      # eps = random noise
z_clean = reverse_diffusion(z_noisy, from t_start to 0)
```

`t_start` controls how much of the steering edit survives the cleanup:
`t_start=0` barely changes anything (edit stays exactly as forced, likely
still off-manifold); `t_start=1` erases the edit entirely (output is just a
typical unsteered activation); `t_start=0.5` (current default for both
experiments) is the middle ground. `--t-start-pilot` runs a small
`{0.3, 0.5, 0.7}` sweep to help pick this value for a given experiment.

**Part 2 — per-sequence scoring.** For each finished sequence: the concept
score (motif regex hit for Exp A, ΔGRAVY for Exp B), pseudo-perplexity
(naturalness — lower is more natural, i.e. more like real protein sequences),
GLP's own diffusion loss on the finished sequence's activations, and
degeneracy checks (max homopolymer run length, unique-amino-acid count,
overall single-residue composition) to catch generations that collapsed into
repetitive low-complexity junk rather than real steering — pseudo-perplexity
scores repetitive sequences as artificially *more* natural, so these have to
be filtered before Part 3, not just eyeballed away (`is_degenerate_sequence`
in `run_steering.py`).

**Part 3 — analysis.**
- **Pareto plot**: naturalness (x) vs. concept score (y), naive vs.
  GLP-corrected, mean trajectory per alpha. The thesis predicts corrected
  points sit up-and-left of naive points — better on both axes, not a
  tradeoff.
- **Three-regime breakdown**: bin samples into safe/transition/broken thirds
  by naive diffusion loss, and check whether GLP-correction helps more in the
  regimes that are more off-manifold to begin with.
- **Rescue check**: for each GLP-corrected point, does it beat the best naive
  concept score achieved at matched-or-better naturalness? This is the
  sharpest single number for "did on-manifold correction actually help."

## Key knobs, in plain terms

| flag | what it controls |
|---|---|
| `--alphas` | steering strength(s) to sweep |
| `--sae-intervention` | Exp A only: `clamp` (hard-set) vs `scale_shift` (scale+shift) |
| `--shift-b` | Exp A `scale_shift` only: the additive shift term |
| `--t-start` | how much noise the cleanup step adds before re-denoising (see above) |
| `--mask-fraction` | fraction of positions re-sampled per round — default **1.0 for Exp A** (whole sequence, every round), **0.10 for Exp B** (small incremental edit) |
| `--n-rounds` | how many mask/steer/resample rounds per sequence — default **100 for Exp A**, **30 for Exp B** |
| `--n-sequences` | sequences per (alpha, naive/corrected) cell |
| `--fold` | run ESMFold for pLDDT (Exp A default: on; Exp B: off, no folding needed) |

## Outputs

`runs/uniref50_500m_4096_d24_steering/steering_<name>/` — one directory per
experiment/variant, containing the per-sequence CSV (`expA_sae_zincfinger.csv`
/ `expB_diffmean_gravy.csv`), `metrics/summary.json` (aggregate stats, logged
via `RunLogger`), and `pareto.png`. `launches.jsonl` in the parent run dir
records the exact job/args behind each output.
