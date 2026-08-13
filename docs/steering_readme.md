# On-manifold steering (Exp 4.1) — methodology + results

Tracks PLAN.md §6.6/W4 (on-manifold steering) and the risk-register Gate G2 note
("if G2 fails → drop steering, ship probing-only paper" — **G2 has not been run
yet**; these experiments are running ahead of that gate, not instead of it).
Implementation: `scripts/run_steering.py`, launched via `scripts/cluster/submit_steering.sh`.
Outputs live under `runs/uniref50_500m_4096_d24_steering/steering_exp41*/` (gitignored).

## 1. Methodology

Thesis being tested: using GLP as an on-manifold prior lets SAE-feature clamping
reach a given concept strength at better naturalness than clamping alone
("naive" steering) — the concept-vs-fluency Pareto frontier should push outward.

**Testbed:** SAE feature(s) for `Zinc finger_C2H2-type` (InterPLM's released
esm2-8m/layer-4 SAE, `Elana/InterPLM-esm2-8m` via `load_sae_from_hf`), detected by
the spec's C2H2 regex `C.{2,4}C.{12}H.{3,5}H` (+ a looser variant).

**Part 0 — off-manifold diagnostic (no generation).** Clamp the target feature(s)
on real held-out activations at increasing `alpha` and measure GLP flow-matching
loss; loss should rise monotonically with `alpha` (clamping pushes off the learned
manifold) before generation is trusted to mean anything.

**Part 1 — masked-Gibbs generation loop** (`masked_gibbs_generate`): fully-masked
init → unsteered round-0 fill → `n_rounds` of {re-mask a random `mask_fraction` of
positions → forward pass → SAE-clamp those positions' layer-4 activations
("UPDATE-TOKENS") → optionally GLP SDEdit-clean the clamped activations
(`glp_correct`: standardize → `sample_on_manifold` from `t_start` → destandardize)
→ forward again with the override → sample new tokens for those positions}. A
`select_fn` tracks the target SAE feature's own activation on the sequence
decoded so far and returns the best-scoring round instead of unconditionally the
last one (the clamp only biases that round's logits, it isn't a persisted
property of the finished protein).

**Part 2 — Exp 4.1 evaluation**, per generated sequence: strict/loose C2H2 regex
concept score, pseudo-perplexity (naturalness), final-sequence GLP diffusion loss,
max homopolymer run + unique-AA count (degeneracy red flags), and the target SAE
feature's own mean/max activation + peak-activating residue on the *finished*,
re-embedded sequence (independent of the regex).

**Part 4 — Pareto plot + rescue sanity check.** Naive (SAE-clamp only) vs.
GLP-corrected points on a naturalness-vs-concept-score scatter; a sanity check
that, at matched-or-better naturalness, GLP-corrected concept score beats whatever
naive steering achieves there ("rescue").

## 2. Exp 4.1 (baseline) + exp41_pilot — results: null on concept, and why

Two runs, same testbed, different knobs:

| | `steering_exp41` (job 44534177) | `steering_exp41_pilot` (job 44540398) |
|---|---|---|
| feature(s) | 7033 only (val AUC 0.952) | 7033 + 3684 (val AUC 0.952 / 0.898) |
| alphas | 0.5, 1, 2, 4, 8 | 1, 2, 4 |
| mask_fraction / n_rounds | 0.2 / 8 (defaults) | 1.0 / 20 |
| n_sequences / cell | 50 | 15 |
| rows | 500 | 90 |

**Concept score was 0 in every single row of both runs** (strict regex: 0/500,
0/90; loose regex: 0/90). The rescue-sanity-check warning fired on the pilot
("holds for only 29%") — but that 29% turned out to be an artifact, not signal
(see §3).

Off-manifold diagnostic (Part 0) did behave as expected — loss rises with
`alpha`, confirming the clamp mechanism itself works, and that two-feature
clamping is markedly more off-manifold than one at matched `alpha`:

| alpha | exp41 (1 feature) mean loss | exp41_pilot (2 features) mean loss |
|---|---|---|
| baseline | 0.869 | 0.869 |
| 1.0 | 0.945 | 1.695 |
| 2.0 | 1.522 | 3.725 |
| 4.0 | 4.952 | 13.761 |
| 8.0 | 19.404 | *(not run)* |

But the target SAE feature's activation on the *finished* sequence never climbed
out of the noise floor either — across all 1890 logged rounds in the pilot, the
single highest value ever observed was **0.29** (mean ≈0.03); GLP-correction did
not raise this (corrected 0.028 mean vs. naive 0.032 mean during rounds; 0.072 vs.
0.085 on final sequences). Whatever the clamp injects during a round does not
survive being "baked into" the sampled tokens.

## 3. Root cause: the auto-selected feature isn't a motif detector

`target_feature_peak_residue` (pilot CSV) — the residue where the "zinc-finger"
feature fires hardest on each finished sequence — should land on Cys/His far
above the ~10% two-letter base rate if it's a real Cys-X(2-4)-Cys-X(12)-His-X(3-5)-His
detector. Across 90 generations:

| residue | R | E | K | G | S/N/T/C/V (each) | W |
|---|---|---|---|---|---|---|
| share | 44.4% | 20.0% | 14.4% | 7.8% | 2.2% | 1.1% |

**Cys appears at base rate (2.2%); His appears 0% of the time.** The feature
picked by plain AUC (0.952 against the annotation span) is almost certainly
keying on the **Lys/Arg/Glu-rich linker** that strings C2H2 zinc-finger arrays
together (the classic TGEKP-type linker), not the CxxC…HxH spacing motif itself.
AUC is rank-only — it can't distinguish "precise to the motif" from "correlated
with the annotated span" — so a high-AUC pick sailed through Part 0/1/2 without
ever being able to produce the concept it was nominally targeting. This explains
the null result independent of `alpha`, `mask_fraction`, or `n_rounds`.

This also explains the pilot's spurious 29% "rescue": `pareto_and_sanity_check`'s
old fallback counted a GLP-corrected point as "rescued" whenever it had *no*
naturalness-matched naive comparator (`best_naive` defaulted to `-1.0`, so
`concept_score=0 > -1.0` passed trivially) — i.e. 29% of corrected points simply
had better naturalness than every naive point, with zero real concept-score
contribution either way.

## 4. Fixes for Exp 4.1c (`scripts/run_steering.py`)

1. **Feature selection now uses InterPLM's own validated criterion — domain-adjusted
   F1 (select-on-valid / confirm-on-test), not AUC.** `select_zinc_finger_features`
   still uses AUC only as a cheap top-64 shortlist on a balanced subsample (mirrors
   `run_probing.py`'s own F1 path), then re-ranks that shortlist by the exact same
   F1 protocol that produced this project's headline SAE F1 numbers
   (`glp_plm/eval/probing_f1.py`, a faithful port of
   `interplm/analysis/concepts/calculate_f1.py`, `docs/INTERPLM_MAP.md:16`) —
   `pf.run_concept_f1`, called `top_n` times, dropping the previous winner each
   round. `zinc_finger_feature.json` now records both val and test F1 per feature.
2. **Rescue-fraction fix.** Corrected points with no naturalness-matched naive
   comparator are now excluded from both the numerator and denominator instead of
   counted as a free win; `pareto_rescue_comparable` / `pareto_rescue_n` are logged
   alongside the fraction so a small/zero comparable set is visible, not silently
   averaged away.

## 5. Exp 4.1c — updated hyperparameters

Methodology (Parts 0–4) unchanged; only the sweep values changed, based on what
Exp 4.1/exp41_pilot showed:

| knob | old | new | why |
|---|---|---|---|
| feature selection | top-AUC | top-F1 (val-select/test-confirm) | §3 — AUC picked a linker proxy |
| `alphas` | 0.5,1,2,4,8 | **1,2,3,4,6** | 0.5 barely moves off baseline loss (0.755 vs. 0.869); 8 is already destroyed pre-generation (loss 19.4) |
| `n_sequences` | 15 (pilot) / 50 (exp41) | **64** | n=15 can't distinguish a ~5% vs ~15% concept-hit rate (arXiv:2502.09135's own top-1-vs-top-2 numbers) |
| `mask_fraction` | 1.0 (pilot) | **0.15** | full remask every round can't let a locally-built motif persist into the next round |
| `n_rounds` | 8 (exp41) / 20 (pilot) | **24** | cheap to raise now that each round perturbs less |
| `t_start` | 0.5 (hardcoded default, never tuned) | *picked by `--t-start-pilot`* | the script already implements a `{0.3,0.5,0.7}` pilot sweep; neither prior run actually invoked it |

## 6. Exp 4.1c — results

Ran on MN5: t_start pilot (job 44541009) then the main sweep (job 44541070),
`runs/uniref50_500m_4096_d24_steering/steering_exp41c/`. Both finished well inside
acc_debug's 2h cap (no qos/time override needed in the end).

**t_start pilot** (3 t_starts × 8 samples, mid-alpha=3.0): t_start=0.5 gave the
lowest mean pseudo-perplexity (15.1, vs. 18.1 @0.3 and 15.5 @0.7) — it confirms
the default both prior runs already used, rather than overturning it. Used
`--t-start 0.5` for the main sweep.

**Feature selection actually changed the pick.** The F1-based re-rank landed on
**features 6158 and 4421** (val/test F1 ≈0.12/0.09 and ≈0.12/0.09) — nothing like
the old AUC pick (7033, AUC 0.952). Lower-looking numbers, but F1 and AUC aren't
on the same scale, so that's not a regression signal by itself; the peak-residue
check below is the real test.

**Peak-residue validity check — the fix holds.** Across all 640 finished
sequences, `target_feature_peak_residue`:

| residue | C | D | V | H | L | R | G | I | E | other |
|---|---|---|---|---|---|---|---|---|---|---|
| share | 13.9% | 12.2% | 11.6% | 9.1% | 7.2% | 7.2% | 6.9% | 6.2% | 5.5% | 20.2% |

**Cys is now the single most common peak residue, and His shows up at 9.1%**
(0% for the old AUC pick). C+H combined = 23.0%, well above the ~10% two-letter
base rate. Still not overwhelming enrichment (D and V, not obviously
motif-related, are close behind C), but this is a categorically different
result from §3's 78.8% R/E/K-and-zero-His — the F1-selected features are
plausibly reading *something* Cys/His-related, not purely a linker-composition
proxy.

**Concept score is no longer uniformly zero.** Strict C2H2 regex: **4/640
(0.6%)**; loose regex: **31/640 (4.8%)**. Both still far short of
arXiv:2502.09135's own 1.7%/13% (top-1/top-2), but no longer a flat null. The
per-(alpha, corrected) breakdown:

| alpha | corrected | pppl | diffusion_loss | strict hits | loose hits |
|---|---|---|---|---|---|
| 1 | naive | 17.19 | 0.709 | 0/64 | 0/64 |
| 1 | GLP | 16.20 | 0.749 | 0/64 | 1/64 |
| 2 | naive | 17.92 | 0.712 | 0/64 | 1/64 |
| 2 | GLP | 17.34 | 0.728 | **1/64** | 6/64 |
| 3 | naive | 17.93 | 0.690 | 0/64 | 4/64 |
| 3 | GLP | 17.19 | 0.730 | **1/64** | 5/64 |
| 4 | naive | 17.82 | 0.704 | 0/64 | 3/64 |
| 4 | GLP | 16.97 | 0.743 | **2/64** | 9/64 |
| 6 | naive | 17.00 | 0.724 | 0/64 | 0/64 |
| 6 | GLP | 17.29 | 0.733 | 0/64 | 2/64 |

Two real, checkable findings here, both small-n but internally consistent:

- **All 4 strict hits, and the majority of loose hits (23/31), occurred under
  `corrected=True`; zero naive (uncorrected) sequences ever hit the strict
  regex**, at any alpha. That's the qualitative direction the whole thesis
  predicts — GLP-correction helping the concept manifest — even though n per
  cell is far too small (1–2 strict hits) to claim significance.
- **GLP-correction is also more natural on average at every alpha except 6**
  (e.g. alpha=4: pppl 16.97 corrected vs. 17.82 naive), consistent with §1's
  Pareto framing, not just a lucky coincidence with the concept hits.
- Alpha=6 is the outlier where correction stops helping on either axis (pppl
  *worse* than naive, no strict hits) — plausibly past the point the SDEdit
  cleanup can undo without erasing the clamp's signal along with the distortion.

**The rescue-metric fix produces a sane, self-consistent number now.**
`pareto_rescue_fraction` = 1% of 316 comparable points — which is exactly what
falls out of the raw counts above (all naive strict scores are 0, so a
matched-naturalness naive comparator's `best_naive` is always 0; only the 4
corrected points with `concept_score>0` can clear that bar, giving ≈4/316 ≈ 1%).
Contrast with exp41_pilot's vacuous 29% (§3) — this 1% is a real signal, just a
small one, not an artifact of the old fallback.

Off-manifold diagnostic (Part 0) still behaves correctly with the new features
and alpha grid — loss rises monotonically (baseline 0.869 → 1.591 → 2.677 →
4.600 → 8.049 → 17.645 @ alpha 1/2/3/4/6), confirming the clamp mechanism itself
is unaffected by the feature-selection change.

**Verdict:** the F1-based feature-selection fix is real and load-bearing — it
moved peak-residue identity from "clearly wrong" (0% His) to "plausibly right"
(9.1% His, Cys the top residue) and took the concept-hit rate off an
unambiguous 0/590 to a small but nonzero, correction-favoring 4/640. It did
**not** reproduce arXiv:2502.09135's headline hit rates (1.7%/13%) — this SAE's
zinc-finger features are evidently weaker/less separable than theirs, or C2H2 in
`facebook/esm2_t6_8M_UR50D`'s layer-4 residual stream is a harder target than in
whatever model/layer that paper used. Whether this effect size is worth building
the paper's steering claim on is a judgment call, not something this run
settles — the direction is right, the magnitude is not yet compelling.

## 7. Suggested next steps (not yet run)

- **n is still small for the events that matter.** 1–2 strict hits per cell
  means every ratio above is one generation away from doubling or vanishing.
  If the effect is worth chasing further, the next lever is raising
  `n_sequences` at just the alphas that showed hits (2/3/4), not a uniform
  sweep.
- **Alpha=6 both underperforms and adds cost** — worth dropping from the next
  grid in favor of finer resolution between 2 and 4, where all the action is.
- The peak-residue enrichment (23% C+H) is suggestive but not decisive; a
  proper validity check would look at whether the *specific spacing* the
  clamp's high-activation positions fall into resembles C2H2's Cys-X(2-4)-Cys
  pattern, not just residue identity in isolation — not implemented here.
