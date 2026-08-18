# On-manifold steering — experiment history

Tracks PLAN.md §6.6/W4 (on-manifold steering) and the risk-register Gate G2 note
("if G2 fails → drop steering, ship probing-only paper" — **G2 has not been run
yet**; these experiments are running ahead of that gate, not instead of it).

This is the **changelog**: every run, in order, with what was tried, what came
out, and why. For a short current-state description of what the pipeline
actually does (no history, no results), see `docs/steering_methodology.md`
instead — that's the doc to read first if you just need to understand the
experiment, not how it got here.

Implementation: `scripts/run_steering.py`, launched via
`scripts/cluster/submit_steering.sh`.

## 1. Exp 4.1 (original, single-experiment version)

The original version of this experiment (before the Exp A/B split in §8) only
tested SAE-feature clamping on the zinc-finger concept. Testbed: SAE feature(s)
for `Zinc finger_C2H2-type` (InterPLM's released esm2-8m/layer-4 SAE,
`Elana/InterPLM-esm2-8m` via `load_sae_from_hf`), detected by the spec's C2H2
regex `C.{2,4}C.{12}H.{3,5}H` (+ a looser variant). Methodology: Part 0
off-manifold diagnostic, Part 1 masked-Gibbs generation loop with SAE clamp +
optional GLP SDEdit correction, Part 2 per-sequence evaluation, Part 3 Pareto +
rescue sanity check — see `docs/steering_methodology.md` for what each part
does; this section is the results log only.

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
  (e.g. alpha=4: pppl 16.97 corrected vs. 17.82 naive), consistent with the
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

## 7. Suggested next steps for exp41c (superseded by §12 for the current state)

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

## 8. `run_steering.py` v2 — Exp A/B split

`run_steering.py` was rewritten around `glprot_steering_spec.md` v2 into two
separate experiments sharing Part 0 (off-manifold diagnostic) and Part 3
(Pareto/three-regime/rescue) infrastructure but otherwise deliberately not
mechanically identical — see `docs/steering_methodology.md` for the
current-state description, or the module docstring for the exact spec:

- **Exp A** (`--experiment A`, out-dir `steering_sae_zinc[_variant]`) — SAE-feature
  intervention on the zinc-finger features from §2-6 above, now with a choice of
  `--sae-intervention clamp` (hard-set, the old behavior) or `scale_shift`
  (`z* = a·z + b`, Villegas Garcia & Ansuini Eq. 7-8).
- **Exp B** (`--experiment B`, out-dir `steering_diffmean_hydro`) — new: diff-of-means
  steering on hydrophobicity (GRAVY), following Huang et al. (arXiv:2509.07983)
  §4.1.1. Real held-out low-GRAVY sequences as init (not random-AA), 10%-mask
  partial refinement, paired naive-vs-corrected design by `init_sequence_id`.
  Needs `scripts/build_property_contrast.py` (auto-built on first run; was
  `build_gravy_contrast.py` before the charge/instability properties were added).

Both experiments' outputs now live under
`runs/uniref50_500m_4096_d24_steering/steering_<name>/` (this run replaced the
older `uniref50_500m_4096_d24_steering` config path used for §2-6's runs — check
`launches.jsonl` in that run dir for the exact job IDs/args behind any number
below).

## 9. Exp A scale-shift — results: worse than plain clamp, not a partial win

Job 44578361 (`steering_sae_zinc_scaleshift`): same F1-selected features as
§6's `exp41c` (6158, 4421 — the pair that got 4/640 strict C2H2 hits under plain
clamp), now via `--sae-intervention scale_shift --shift-b 10`,
`alphas=2,5,10,20,30`.

**Off-manifold diagnostic saturates immediately and never moves again:**

| alpha | baseline | 2 | 5 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| mean diffusion loss | 0.869 | 43.95 | 43.95 | 44.01 | 44.18 | 44.16 |

`diagnostic_A_monotonic: false` — a ~50x jump at the *lowest* alpha tested,
flat for the next order of magnitude. The additive `b=10` dominates; `a`
barely matters once it's added, so the whole alpha sweep is exploring one
already-destroyed region rather than a graduated one.

**Zero strict C2H2 hits across all 1200 sequences** (600 naive + 600
corrected; loose-regex hits 0-4/100 per cell, no trend, no correction
advantage) — worse than `exp41c`'s 4/640 with the *identical* features.
Sequences aren't degenerate by the §10 criteria below (homopolymer run ~2,
unique-AA ~16 in every cell) — this is a real null, not a collapse artifact:
the intervention lands in a region that still decodes to fluent-looking
protein but is nowhere GLP has ever seen, and never produces the concept.

**Verdict: `scale_shift` with `b=10` is a strictly worse intervention choice
than `clamp` for this target**, not an alternative worth keeping alongside it.
`b=10` was never validated against the feature's natural activation scale
before this run — see §12.

## 10. Exp B diff-of-means/GRAVY — results, and two rounds of a naturalness-metric artifact

Jobs 44578271/44578326 (`steering_diffmean_hydro`): 15 held-out low-GRAVY
sequences, `alphas=0,0.05,0.25,0.5,1,2,5,20`, naive vs. GLP-corrected, paired by
`init_sequence_id`.

**Off-manifold diagnostic behaves correctly** — loss rises monotonically with
alpha (0.869 baseline -> 0.863 @0.05 -> 0.869 @0.25 -> 0.885 @0.5 -> 0.953 @1 ->
1.140 @2 -> 1.737 @5 -> 3.181 @20).

**First pass (raw CSV) looked like a clean win for GLP-correction at every
alpha, but the Pareto plot told a different story:** naive's mean trajectory
sat up-and-left of (i.e. dominating) GLP-corrected's, driven entirely by
naive's alpha=20 point (mean pppl 3.5, mean ΔGRAVY 3.9). That point turned out
to be sequence collapse: 13/15 of naive's alpha=20 rows are dominated by a
single homopolymer run (mean max-homopolymer-run 13.3 vs. ~3 everywhere else,
mean unique-AA-count 12.9 vs. ~19 everywhere else). Pseudo-perplexity scores
a repetitive sequence as artificially *more* natural — it's trivially
self-predictable, not actually fluent — so this one degenerate cell was
dominating both the Pareto plot and `pareto_B_rescue_fraction` (the naturalness-
matched comparator for nearly every corrected point).

**Second pass (composition-aware filter) found the same failure mode is not
naive-only.** `unique_aa_count`/`max_homopolymer_run` alone miss sequences that
interleave a dominant residue with occasional others rather than running it
solid — caught instead by overall composition (any one amino acid making up
>=30% of the sequence; random 20-AA composition is ~5%). Checking that
revealed **GLP-correction itself, not just naive at high alpha, degenerates
sequences** — and specifically, that at alpha=5 it is *correction* causing the
collapse, not the raw clamp:

| alpha=5, `init_sequence_id=8` | pseudo-perplexity | delta-GRAVY | sequence |
|---|---|---|---|
| naive | 14.8 | 1.78 | `MWTIDVGKAFLAHLKLHVGLAKLLVLKSLPLSCRLLLSKFDEILTFCRNLLLAAEPGVDDYGLKLVEFHVKAVMLFGLVLFCIGLVLGIRSLCDSSAACGVKTRLVAFSLPASLQLAALKQVLRERGLLLL` |
| GLP-corrected | 5.5 | 3.61 | `MLLTLGLLVLAVILLLLLLVKPLLLTELLLLLLLSLVLSPLVFPVSLPGLLAVLLLLVKLLVPLVPVVVSVLLLVLLLVDLLFLLLVLVPVLLLGLDVALVLGVLLVLLLLLLLLLPFLLLLVPLTLVLFL` (55% Leucine, runs of 9+) |

At alpha=5, naive stayed clean (15/15 pass the filter, Leu-fraction 0.12-0.26,
pppl 12-17) while GLP-correction collapsed 13/15 sequences into Leucine
homopolymers (Leu-fraction 0.22-0.55, pppl as low as 5.3). Correction is
supposed to pull off-manifold points *back toward* realistic proteins; here it
is doing the opposite. At alpha=20 both arms collapse (0/15 naive, 1/15
corrected pass) — there the raw clamp itself is already too strong for
anything to survive, a different (and expected) failure mode.

**With the composition filter applied and alphas 5/20 dropped** (fewer than 5
clean samples in one arm — too few to trust a mean; still present in the CSV,
just excluded from Part 3 analysis, see `is_degenerate_sequence` in
`run_steering.py`), the six remaining alphas (0, 0.05, 0.25, 0.5, 1, 2) show a
consistent, real win:

| alpha | n(naive,corr) | naive pppl / dGRAVY | corrected pppl / dGRAVY |
|---|---|---|---|
| 0 | 15,14 | 15.03 / 0.40 | 13.86 / 1.35 |
| 0.05 | 14,14 | 15.01 / 0.12 | 12.19 / 1.39 |
| 0.25 | 14,15 | 13.85 / 0.37 | 12.40 / 1.54 |
| 0.5 | 14,11 | 15.06 / 0.35 | 11.34 / 1.73 |
| 1 | 14,11 | 14.95 / 0.36 | 12.47 / 1.71 |
| 2 | 15,9 | 14.64 / 0.28 | 11.21 / 2.08 |

GLP-corrected beats naive on *both* naturalness (lower pppl) and concept score
(higher dGRAVY) at every one of these six alphas — not a tradeoff. Paired
Wilcoxon (by `init_sequence_id`) is significant at all six (p <= 0.004,
n=9-14 pairs; mean paired delta +0.93 to +1.87 dGRAVY in favor of correction).

`pareto_B_rescue_fraction` (naturalness-matched, corrected beats best
naturalness-matched naive) is now **0.68 (52/77 comparable)** — up from
~0.01-0.02 pre-filter, where the metric was being sabotaged by exactly the
degenerate rows above. Three-regime breakdown (naive diffusion-loss terciles,
cutpoints 0.700/0.764): rescue fraction is 0.50 (safe, n=2 comparable — thin),
0.235 (transition, n=17), 0.818 (broken, n=55) — rescue is strongest in the
most-off-manifold naive bucket, weakest in the middle one.

**Caveat carried over from the original interpretation:** even the alpha=0 arm
(no clamp at all, `diffmean_clamp`'s algebraic no-op) shows correction raising
dGRAVY (paired delta +0.93, p=0.0001) — `corrected=True` always runs the SDEdit
pass regardless of alpha, so part of every alpha's win is SDEdit alone pulling
these deliberately-low-GRAVY held-out sequences toward the training
manifold's typical (higher) GRAVY, not an alpha-dependent steering-rescue
effect specifically. The alpha 0.05-2 deltas (1.2-1.9) exceed this alpha=0
baseline (0.93), so there is a real incremental effect on top, but a citable
version of this result should report deltas net of the alpha=0 control, not
the raw per-alpha numbers above.

## 11. Fixes applied to `run_steering.py`

- **`is_degenerate_sequence(row, min_unique_aa=8, min_homopolymer_run=8,
  max_single_aa_fraction=0.30)`** — excludes collapsed/low-complexity rows from
  Part 3 analysis only (Pareto plot, three-regime breakdown, rescue fraction,
  paired Wilcoxon); the raw row stays in the written CSV. Composition check
  (>=30% one amino acid) added after the homopolymer-run/unique-AA check alone
  was shown (§10) to miss corrected-arm degeneracy that interleaves the
  dominant residue instead of running it solid. Wired in for both experiments
  right before the `is_a` branch that calls `pareto_exp_a`/`pareto_exp_b` /
  `three_regime_breakdown`; excluded-row count logged to `run.log_summary` as
  `n_degenerate_excluded`.
- Applying this to the existing `steering_diffmean_hydro` run's artifacts
  (without re-running the GPU job): `pareto.png` was regenerated with alphas
  5/20 dropped (originals kept as `pareto_unfiltered.png`), and
  `metrics/summary.json` was recomputed on the clean rows (original kept as
  `summary_unfiltered.json`). `steering_sae_zinc_scaleshift`'s artifacts were
  **not** regenerated — only 3/1200 rows are flagged there and the result is
  already a flat 0/1200 strict hits either way, so it doesn't change anything.
- Not yet done: an equivalent redraw for the *plot's readability* itself (mean
  ± SEM trajectories with naive->corrected arrows per alpha, no per-sample
  scatter clutter, excluded points dropped from the figure rather than shown
  grayed-out) was done ad hoc for this conversation but the plotting code in
  `pareto_exp_b`/`pareto_exp_a` itself still draws the old dense-scatter style
  — worth porting the arrow/SEM version back into the script if it's kept.

## 12. Suggested next steps (superseded by §13)

**Exp B (diff-of-means/GRAVY) — where the signal is:**
- **Isolate whether alpha=5's collapse is `glp_correct` itself or the 30-round
  masked-Gibbs loop compounding it.** Run `glp_correct` once on an
  alpha=5-clamped held-out activation (same shape as the Part 0 diagnostic,
  no generation loop) and check composition directly, vs. turning on
  `--log-round-diagnostics` (writes `round_property.csv`, `round_gravy.csv` in
  runs predating the multi-property generalization) for a small alpha=5
  pilot to see which round it tips over in.
- **Fill in the alpha grid between 2 and 5** (e.g. 3, 4) — naive stays clean at
  5 but corrected doesn't, so the breakdown is likely a sharp threshold, not
  gradual; worth finding before calling "alpha 0-2" the paper's operating
  envelope.
- **Re-run `--t-start-pilot` for Exp B specifically** (`t_start=0.5` was picked
  for the old zinc-finger experiment and reused here by default, never
  re-validated for GRAVY/diff-of-means) — a smaller `t_start` adds less noise
  before the reverse pass, which may keep correction from having enough room
  to wander into the Leucine attractor at higher alpha.
- **Report the headline numbers net of the alpha=0 control** (§10's last
  caveat) rather than the raw per-alpha paired deltas, for anything citable.

**Exp A (SAE scale-shift/clamp on zinc-finger) — lower priority, weaker
signal even before this run:**
- `scale_shift`'s `--shift-b 10` was never tuned against the feature's natural
  activation scale before this run destroyed the whole alpha range; worth a
  quick sweep of smaller `b` (or `b=0`, scale-only) before concluding
  `scale_shift` itself (not just this `b`) is the wrong idea.
- §7's still-open items (raise `n_sequences` at the alphas that showed hits
  under plain `clamp`, drop alpha=6, check C2H2 spacing not just residue
  identity) remain open and untouched by this run.

---

## 13. Exp B rebuilt: bidirectional, folded, and re-measured (2026-08-17/18)

The single largest revision of Exp B so far. §10's headline — "GLP-correction
improves naturalness *and* concept strength together" — **does not survive**.
The concept-strength half survives and is now much better evidenced; the
naturalness half was an artifact of the metric. Details below, with the
numbers each claim rests on.

### 13.1 Infrastructure that was broken

Seven bugs, all fixed and committed:

1. **Steering could not run at all** from the `crg562661` checkout — InterPLM's
   `sae/intervention.py` targets an older `nnsight` API (`.input[0][0]`,
   `.save().value`). Fixed as a vendored patch (`patches/`, applied by
   `scripts/apply_interplm_patches.sh`); **re-run it after any
   `git submodule update`** or generation dies with an `expand()` shape error.
2. **Figures never reached `figures/`** — `save_fig` wrote to the run-dir root.
3. **Every Exp B run shared one metrics/figures directory.** `RunLogger`
   redirects on `basename(out_dir)`, which was a fixed leaf, so each run
   silently clobbered the previous one's `summary.json` and plots. Fixed by
   requiring a unique `--exp-name`.
4. `--t-start-pilot` only ran at the mid alpha, so it could not separate the
   correction's own drift from the steering gain — it would have selected the
   *worst* available `t_start`.
5. Part 0 never tested the correction operator on its own.
6. `%`-of-excess-recovered divided by ~zero at small alpha (printed −1294%).
7. The fold stage's rate timer included the model load (mis-sized runs 2×).

### 13.2 The correction has its own concept drift — subtract it

At alpha=0 the two arms differ **only** by whether `glp_correct` ran
(diffmean steering at alpha=0 is an exact identity), yet:

| alpha=0 drift (ΔGRAVY) | up (low-GRAVY starts) | down (high-GRAVY starts) |
|---|---|---|
| naive (Gibbs only) | +0.347 | **−0.282** |
| GLP-corrected | +1.145 | **+0.536** |

Mechanism: SDEdit drives Leu 7.1→18.0% and Val 5.1→13.6% (~2× natural
abundance); L+V alone are 57% of the drift. It is **not** regression to the
mean — the naive arm flips sign from high-GRAVY starts while the corrected arm
stays positive, i.e. the correction carries a **fixed hydrophobic bias**. The
up-only design could not have distinguished these.

That drift was 65–97% of the corrected arm's raw ΔGRAVY at every alpha ≤ 2, so
**never quote the raw corrected-vs-naive gap**; quote `steering_attributable`
(each arm differenced against its own alpha=0 run, per init sequence).
Replicated independently at 5 rounds/T=0.5 (+0.454 up, +0.278 down).

### 13.3 Bidirectional steering (the headline)

Steering the same vector both ways — `--directions up,down`, held-out
high-GRAVY starts carved out of the high quartile before the vector is built —
is far stronger evidence than one rising curve, because a direction-independent
artifact pushes both trajectories the same way and cancels.

Divergence (up − down of the attributable effect, unpaired Mann–Whitney; the
two directions use different sequences so a paired test would be wrong):

| alpha | naive | GLP-corrected |
|---|---|---|
| 0.5 | +0.010 (p=0.98) | **+0.551 (p=1.1e-05)** |
| 1 | +0.112 (p=0.23) | **+0.969 (p=6.6e-10)** |
| 2 | +0.077 (p=0.44) | **+1.581 (p=2.4e-13)** |
| 5 | +1.400 (p=1.4e-10) | **+3.382 (p=2.4e-13)** |

Corrected is significant from alpha=0.5; naive only from alpha=3. Gated and
ungated (no degeneracy filter) agree to ~1%, so the result does not depend on
the quality gate. Symmetry is close to perfect at alpha=2 (up +0.785, down
−0.796).

### 13.4 `t_start` = 0.7 for Exp B, and the alpha=5 "breakdown" was its artifact

A 7-point pilot at n=15 (`--t-start-grid 0.2..0.8`) found 0.5–0.6 to be a
pathological band: highest drift, +11pp composition damage, and — the tell —
the *best* pseudo-perplexity. 0.7 has the strongest and most reliable
attributable gain (+0.43, p=1e-4) at a composition cost indistinguishable from
zero. On the full sweep, 0.5 → 0.7 dropped degenerate output **26% → 7.7%**,
took alpha=5's usable corrected rows from **2 → 14**, and made the
dose-response monotone through alpha=5. §10's "unresolved failure mode of the
correction at alpha=5" was a `t_start` artifact, not a property of steering.

Separately, Part 0 now applies one SDEdit pass to clamped activations with **no
generation**: it recovers **89–95%** of the clamp's excess diffusion loss even
at alpha=20. So SDEdit handles heavily-clamped activations fine; the collapse
was the masked-Gibbs loop compounding.

### 13.5 The naturalness claim fails, and why

ESMFold2 (`scripts/fold_steering_outputs.py`, separate Python 3.12 venv,
resumable) gives pLDDT/pTM. Three findings, in order of importance:

**(a) Both model-based naturalness metrics are gamed.** The sequences the
degeneracy gate rejects (64% Leu+Val) score:

| | pLDDT | pTM | pppl |
|---|---|---|---|
| rejected as non-protein | **81.0** | 0.452 | **5.46** |
| our clean output | 54.7 | 0.277 | 14.53 |
| natural ubiquitin | 80 | 0.74 | — |

Poly-L/V junk beats **real ubiquitin** on pLDDT and scores 3× "better" pppl
than our clean output (p ≤ 1e-6). On the larger bidirectional run pTM is merely
*indistinguishable* from junk (p=0.86), not better. **The methodological trap:**
on kept rows alone pppl correlates correctly with pLDDT (rho=−0.48, p=4e-17),
which reads as mutual validation — but that is computed after the gate removed
the rows that expose the shared bias. **Always check metrics on the rejected
rows too.**

**(b) On-manifold ≠ foldable.** GLP diffusion loss of the *finished* sequences
is 0.72–0.86 against 0.869 for real activations: everything we generate is
**more typical under GLP than real proteins are**, at half the foldability.
The manifold is one of activations, not structures; repeated sampling lands in
the distribution's bland middle; and the decode samples masked positions
independently, so local plausibility survives while joint (structural)
constraints do not. Note also that "on-manifold" measured by GLP's own
diffusion loss is GLP grading its own homework — the fold metrics are the
independent check and they disagree.

**(c) The decode loop, not GLP, costs foldability.** Folding the 80 natural
held-out starting proteins: pTM 0.373 (up set) / 0.629 (down set). The alpha=0
**naive** arm — no steering, no GLP — comes out at 0.275 / 0.319. Both arms
land in the same place (0.264 vs 0.276 pooled, n.s.), so this is the
masked-Gibbs protocol, not the correction.

**Baseline discipline:** compare against the held-out set the run actually
generates from. The pooled 0.501 average is wrong for a single-direction
comparison and makes light editing look damaging when it is not.

### 13.6 The quality gate is calibrated; do not tighten it on complexity

`scripts/analyze_sequence_complexity.py` implements SEG-style windowed Shannon
entropy (Wootton & Federhen; window 12, 2.2-bit trigger).

- **The gate is well calibrated.** Applied to the 80 natural starting proteins
  it rejects 3/80 (3.8%) — comparable to the 2.9% it rejects of our own
  output, so it is not tuned to target our failure mode. One constant was
  wrong: `min_homopolymer_run=8` rejected an ordinary protein with a run of 9,
  so it is now **12** (natural runs: median 3, p95 5.1).
- **Our generations are not low-complexity relative to real proteins**
  (min window entropy 2.059 vs natural median 2.126).
- **Do not adopt SEG's 2.2-bit trigger as a gate** — it would reject **75%** of
  the natural low-GRAVY starting proteins. It is a masking trigger for homology
  search, not a degeneracy test.
- **Never gate on complexity at all:** complexity is *affected by the
  treatment* (corrected arm's min entropy falls 2.09 → 1.73 from alpha=0 to 5,
  since steering toward hydrophobicity draws on a smaller sub-alphabet), so a
  complexity cut removes the successfully-steered sequences and biases the
  effect toward zero (alpha=2 divergence 1.59 → 0.93 at SEG's threshold).

Same lesson as pLDDT: **a quality measure the treatment changes must be an
outcome, never a filter.** Hence `pass_rate_*` in summary.json — "fraction of
generations that are non-degenerate AND above threshold", denominator = every
sequence generated, so degenerate output counts as failure rather than being
quietly dropped. Using pLDDT≥70 as a *filter* would keep 87% of the degenerate
sequences and 12% of the clean ones, taking a set from 8% to 38% junk.

### 13.7 Two defensible operating points

| | divergence @alpha=5 | pTM up / down (pure decode loop) | degenerate |
|---|---|---|---|
| 30 rounds, T=1.0 (`bidir_n40`) | **+3.38** | 0.275 / 0.319 | 2.7% |
| 5 rounds, T=0.5 (`t05_m10_r5`) | +1.56 | **0.332 / 0.435** | 3.4% |
| natural starting proteins | — | 0.373 / 0.629 | 0% |

Half the effect for meaningfully better structure and 6× less compute
(p=0.007 down, p=0.10 up). Neither reaches natural foldability: the naturalness
claim is **improved, not rescued**.

**A failed idea worth recording.** Huang et al.'s Algorithm 1 selects mutation
sites by relatedness (cosine similarity to the steering vector, their Eq. 4),
mutating the T *most opposed* residues; we mask uniformly at random.
Implementing theirs (`--mask-selection relatedness`) doubled the fraction of
protein preserved (12% → 25%) as predicted, **but cost ~6× of the steering
effect** (+0.969 → +0.171) with no significant foldability gain. Likely because
it targets exactly the residues ESM most resists changing, and our steering —
one layer, masked positions only — is too weak to move them, where they steer
every layer across all tokens. Not adopted; random masking is better for this
setup.

### 13.8 Mistakes made and corrected during this session

Recorded because each was caught by *checking* rather than reasoning, and the
same failure mode recurs:

1. Claimed the confound was "structural, not tunable" from 3 noisy points at
   n=8. It is tunable.
2. Nearly concluded pppl was validated by its correlation with pLDDT — computed
   on kept rows only (see §13.5a).
3. Overstated pTM as discriminating; the run's own test showed it is not.
4. Compared pilot results against the pooled 0.501 baseline when the pilots
   generate from the hydrophilic set alone (0.373).
5. **Recommended T=0.5 × 30 rounds as the operating point.** The pilot analyses
   never applied the degeneracy gate, and at that cell T=0.5 mode-collapses:
   8/15 non-proteins, composition distance 38 → 106, min entropy 0.98. Those
   collapsed sequences score *high* pTM, so without the gate it looked like the
   best cell in the grid.

The through-line: **degeneracy must be counted, never dropped**, in every
analysis including the quick ones.

### 13.9 Still open

- Exp A has never been re-run with any of the above.
- ~~The correction's own foldability cost~~ — **closed** by §14.5b: -0.012 pTM
  [95% CI -0.029, +0.005], p=0.43 over 235 paired comparisons. Not established,
  and bounded to be small.
- We steer one layer (hook parity with the SAE forces this) and apply the edit
  only at mutated positions; Huang et al. do both across all layers/tokens.
  That is the untested deviation, and the likely reason relatedness selection
  failed here.
- Neither operating point reaches natural foldability.

---

## §14 Extending to charge and instability (2026-08-18)

Banerjee et al. steer three biochemical properties — hydrophobicity, charge and
instability. Only hydrophobicity had been run here. This section records the
design of the extension; **no charge or instability run has been executed yet**,
so nothing below is a result.

### 14.1 What changed in the code

`glp_plm/eval/steering_properties.py` is now the single registry of steerable
properties. `scripts/build_gravy_contrast.py` became
`scripts/build_property_contrast.py --property {gravy,charge,instability}`, and
`run_steering.py` grew `--property`. Each property gets its own out-dir leaf
(`steering_diffmean_{hydro,charge,instability}_<NAME>`), contrast file
(`<property>_contrast.json`) and CSV (`expB_diffmean_<property>.csv`); the GRAVY
names are unchanged, so the archived runs stay readable.

### 14.2 The four methodological decisions

**1. Charge is steered per residue, not net.** Net charge is a *sum* over
residues, so ranking proteins by it ranks them substantially by length, and a
diff-of-means vector built on that contrast would encode "long" as much as
"cationic". GRAVY is already a per-residue mean and the instability index is
already length-normalized (10/L times a dipeptide sum), so charge was the only
one that needed fixing. Net charge is still recorded per sequence; since
masked-Gibbs never changes length the two are proportional within an init, which
is the unit every effect estimate is paired on.

**2. The length confound is measured on every build.** Even with a normalized
score, a residual length difference between the two contrast sets can survive.
`build_property_contrast.py` reports the gap in pool-SD units and warns above
0.25; `--match-length` forces the two sets onto a shared length distribution by
a stratified draw. Left off by default so the GRAVY vector is not silently
changed — **decide it from the reported gap, per property**.

**3. Every generation is scored on every property.** The CSV carries
`delta_gravy`, `delta_charge`, `delta_instability` and `delta_net_charge`
regardless of which was steered. This is what makes
`scripts/analyze_property_specificity.py` possible: a 3×3 matrix of steered vs
measured, in natural-SD units. An off-target column that was never written
cannot be recovered without regenerating, which is why it is unconditional.

The matrix is read at **matched on-target effect** by default (`--match-on effect`), not
at a shared alpha: comparing off-target numbers taken at different on-target strengths
penalises whichever property happened to be pushed hardest, which is a property of the
grid rather than of the direction. Alphas where the grid does not bracket the target are
flagged as unmatched rather than quietly used. `--match-on alpha` gives the
matched-perturbation reading instead.

The matrix' second panel subtracts the *natural* coupling. The properties are
correlated in real proteins, and the contrast build now stores that correlation matrix
from the UniRef50 pool. A faithful direction should reproduce the coupling; only the
residual is evidence of an unspecific vector. Reading a raw off-diagonal cell as
"leakage" would mistake protein chemistry for a defect in the method.

**The coupling is now measured, and it is not the one anyone assumes.** On the shared
5,790-sequence UniRef50 pool: GRAVY x instability r = -0.348, charge x instability
r = +0.160, and GRAVY x charge r = **-0.030** — essentially zero. The intuition that
"charged residues are hydrophilic, so charge and GRAVY must anti-correlate" is wrong for
*net* charge per residue, because a protein rich in both Lys and Glu is strongly
hydrophilic with near-zero net charge. That makes hydrophobicity and charge close to
orthogonal axes here, so leakage between those two is genuinely informative rather than
expected; hydrophobicity and instability are the pair where coupling has to be subtracted
before an off-target number means anything. Length correlates |r| <= 0.053 with all three,
confirming the per-residue normalization.

**4. Alpha is not comparable across properties in raw mode.** `h + alpha*v`
rotates `h` by an angle set by `alpha*||v||/||h||`, and the three vectors have
different norms, so a shared alpha grid means three different perturbation sizes.
`--alpha-mode relative` reinterprets alpha as the fraction of `mean||h||` added
along `v/||v||`, using the `mean_act_norm` the contrast build now records. Raw
stays the default (it is what every existing run used); **a cross-property
comparison must use `relative` for all three**, including a re-run of
hydrophobicity.

### 14.3 Why instability is the most informative of the three

GRAVY and charge are both linear in amino-acid composition. The SDEdit
correction's own drift is *also* compositional (Leu/Val enrichment, §13.2), so on
those two targets a compositional artifact and a real steering effect are the
same kind of object, separable only by the alpha=0 differencing and the
bidirectional design.

The instability index is a sum over **dipeptides** (Guruprasad's DIWV table), so
it is not a function of composition alone. A compositional artifact cannot
produce it directly. `analyze_steering_confound.py` now decomposes whichever
property the run steered into its own components — residues for GRAVY/charge,
dipeptides for instability — so "did the gain come from a broad shift or from
stacking a few high-weight dipeptides?" is answerable. DIWV values span −15.9 to
+58.3, so a handful of dipeptides *can* move the index a long way: that check is
not optional for this target.

### 14.4 Expected failure modes, stated in advance

- **Charge should collapse into homopolymers sooner than GRAVY did.** Steering
  hard toward cationic drives poly-K/poly-R, toward anionic poly-E/poly-D — which
  is exactly what `is_degenerate_sequence` rejects. Expect a higher degenerate
  rate; report it as an outcome (`pass_rate_*`), never filter on it (§13.5b).
- **The correction's alpha=0 drift will differ per property.** Leu/Val enrichment
  depletes charged residues, so the drift should push |charge| *down*. It has to
  be measured per property; `steering_attributable` already handles it, but the
  sign and size are not transferable from the GRAVY run.
- Nothing about the decoding-loop foldability cost (§13.6) is property-specific,
  so both new properties inherit it. The naturalness claim is not expected to
  come out differently.

### 14.4a Bugs found in review before the first run

Recorded because all four were the kind that produce a plausible number rather than a
crash:

1. **Composition decomposition weighted residues, not sequences.** `mean_i score(s_i)`
   decomposes over the mean of per-sequence frequencies; pooling all residues into one bag
   weights long proteins more, so the components no longer summed to the effect they
   explained (error +0.0022 GRAVY on a mixed-length set). Invisible whenever the sequences
   share a length, which is why the original single-sequence test passed.
2. **Length-matched draws were length-ordered.** `_length_matched_draw` assembles indices
   bin by bin and the caller splits the draw by position, so the held-out starts got the
   shortest proteins — a length confound between the starts and the vector, introduced by
   the fix for the length confound between the contrast sets. Now shuffled, and both sides
   are capped at the shared per-bin capacity so their length distributions are identical by
   construction rather than approximately.
3. **`--match-on effect` could select alpha=0.** The neutral alpha is the no-steering
   control; leaving it in the candidate set let a property whose grid overshoots the target
   "match" at zero steering, producing an all-zero row that reads as perfect specificity.
4. **The specificity matrix standardized by the held-out starts.** Those are drawn from
   opposite quartiles of the *steered* property, so their pooled SD is bimodal and inflated
   for exactly that property — shrinking the diagonal, i.e. understating on-target
   specificity. Now standardized by the UniRef50 pool SD, recorded at build time.

Also fixed: `analyze_steering_confound.py` used a homopolymer threshold of 8 where
`run_steering.py` uses 12, so it analysed a smaller row set than the summary.json numbers
it prints beside them, contradicting its own docstring.

### 14.5 First results (2026-08-18, jobs 44772569 / 44773025-27)

Contrast build: one shared 5,790-sequence UniRef50 pool, length gaps +0.13 / -0.03 / -0.13
pool SD (all under the 0.25 threshold, so **`--match-length` was not needed** and the full
contrast separation of +2.37..+2.51 pool SD is kept). Three runs at
`--alpha-mode relative`, grid `0,0.032117,...,0.642336` chosen so hydrophobicity maps onto
its canonical raw grid 0/0.25/0.5/1/2/3/5. 1,120 generations each, 30 rounds, T=1.0,
t_start 0.7, n=40 starts per direction.

**The hydrophobicity run is an exact replication.** Divergences +0.248/+0.552/+0.969/
+1.581/+2.508/+3.382 against the canonical +0.25/+0.55/+0.97/+1.58/+2.51/+3.38; alpha=0
drift +1.145/+0.536 against +1.15/+0.54; degenerate 30/1120 = 2.7% against 2.7%. Relative
mode is a pure re-parameterization, as intended.

**The efficacy claim holds for all three properties.** The corrected arm separates the two
directions from low alpha; the naive arm is indistinguishable from zero until high alpha:

| property | corrected divergence @ alpha=0.257 | in pool SD | naive @ same alpha |
|---|---|---|---|
| hydrophobicity | +1.581 GRAVY (p=2.4e-13) | 3.46 | +0.077 (p=0.44) |
| charge | +0.216 charge/res (p=1.9e-14) | 3.61 | +0.055 (p=0.02) |
| instability | +25.0 index (p=7.4e-9) | 1.59 | +5.3 (p=0.19) |

Charge is steered as strongly as hydrophobicity. Instability moves about half as far,
which is the expected direction for a dipeptide (order-dependent) target under a
composition-shifting intervention.

**Prediction that failed:** §14.4 predicted charge would collapse into homopolymers sooner
than GRAVY (poly-K/R, poly-E/D). It had the *lowest* degenerate rate of the three: 1.9%
against 2.7% for GRAVY and 2.1% for instability.

**The specificity matrix** (matched on-target effect ~1 SD, corrected arm; rows steered,
columns measured, natural SD units):

| steered \ measured | GRAVY | charge | instability |
|---|---|---|---|
| gravy | **+1.21** | -0.03 | +0.07 |
| charge | -0.18 | **+0.67** | +0.60 |
| instability | -0.54 | +0.08 | **+0.86** |

Hydrophobicity is almost perfectly specific. The other two leak: charge moves instability
nearly as much as it moves charge, and instability moves GRAVY by -0.54. After subtracting
the natural coupling the residuals are gravy->instability +0.49, charge->instability +0.49,
instability->gravy -0.24, charge->gravy -0.16. Note the sign on the first: nature says
raising GRAVY should *lower* instability (r=-0.35), and our GRAVY steering leaves it
slightly up — i.e. it reaches the target property without the stabilization that naturally
accompanies it. Read as evidence, not proof: the coupling is observational and an
intervention is not obliged to reproduce it.

**The alpha=0 correction drift is ONE fixed compositional bias seen through three lenses —
and the third is a null control.** At alpha=0 the corrected arm's composition shifts
L +5.9pp, V +5.4pp, E -2.8pp, K -2.4pp, R -1.6pp, D -1.6pp (the naive arm moves nothing by
more than 0.7pp). Project that onto each property:

| property | naive up / down | corrected up / down | fixed bias (corrected - naive) |
|---|---|---|---|
| hydrophobicity | +0.347 / -0.282 | +1.145 / +0.536 | **+0.798 / +0.818** (+1.77 SD) |
| charge | +0.055 / -0.039 | +0.063 / -0.071 | +0.008 / -0.032 (-0.20 SD) |
| instability | +14.99 / -15.48 | +4.66 / -27.13 | **-10.33 / -11.65** (-0.70 SD) |

A fixed bias adds the same amount from both starting sets; mean reversion flips sign and
cancels in the difference. GRAVY and instability both show a fixed bias, equal from both
sets to within 3% and 12% respectively — and their signs are what Leu/Val enrichment
predicts (most hydrophobic residues; stabilizing in the DIWV table). Charge shows none,
which is also predicted: the correction sheds positive (K+R, 4.0pp) and negative (E+D,
4.4pp) charge in near-equal measure, so they cancel. The naive arm is clean, near-symmetric
mean reversion on all three.

This is the strongest evidence yet for the §13.2 account, because it includes a case where
the mechanism predicts *no* effect and none is found. The alpha=0 confound is not a quirk
of hydrophobicity; it is a measurable property of the SDEdit operator with a per-property
signature that can be predicted in advance.

### 14.5a The instability gain is compositional — the order test comes out negative

Instability was included because it is the one target a purely compositional artifact
cannot produce *directly* (§14.3). Running the test says the **steering** is compositional
too, which is a different and more interesting finding than the one it was meant to guard
against.

Every sequence splits exactly into a composition-determined part and an order excess,
using the closed form for the expected index over permutations of its own residues
(`scripts/analyze_instability_mechanism.py`; verified against exhaustive enumeration and
50k shuffles). Splitting the steering-attributable divergence at the top alpha:

| arm | total | composition | order excess | order share |
|---|---|---|---|---|
| corrected | +64.93 (p=4.2e-14) | +61.27 (p=2.1e-14) | +3.65 [95% CI −1.10, +8.41] (p=0.10) | 6% [−2%, 13%] |
| naive | +17.01 (p=5.5e-05) | +19.21 (p=8.1e-07) | −2.19 [95% CI −7.54, +3.15] (p=0.68) | — |

So **at most ~13% of the effect is order-specific**, and the point estimate is 6%. This is
a bounded null, not an underpowered one.

The composition change says what actually happened. Steering up enriches R +8.7pp,
P +7.5pp, S +7.5pp and depletes V −5.9, L −5.7, F −2.6, I −2.5; steering down mirrors it.
The "top dipeptides" (RR, RS, SP, SS, SR, RP, PS, PP) are simply every pair formable from
{R, S, P} — they are a readout of composition, not of targeted adjacencies. That is also
why the concentration statistic (5 dipeptides for 50% of the positive contributions) is
not evidence of order targeting: three enriched residues generate ~9 elevated pairs.

Three consequences:

1. **All three properties are steered through the same channel: composition.** That
   explains the effect sizes — hydrophobicity and charge are *purely* compositional and
   move 3.46 and 3.61 pool SD, while instability has a component the intervention cannot
   reach and moves 1.59 SD.
2. **It explains the instability -> GRAVY leak of −0.54.** Enriching R/S/P while depleting
   V/L/I/F *is* lowering hydrophobicity; the off-diagonal is the same composition change
   read on another scale, not a separate failure.
3. It is a real limitation to state: a diff-of-means direction at one layer, applied only
   at masked positions, moves amino-acid composition. Whether steering all layers and all
   tokens (Huang et al.) reaches order-level structure is untested here and is the natural
   follow-up.

### 14.5b Naturalness across the three properties (folded 2026-08-18)

3,360 generations folded with ESMFold2, 0 errors, plus each property's own 80 natural
starting proteins (they differ per property, so the hydrophobicity baselines do not
transfer — `scripts/export_heldout_starts.py`).

**The decoding loop is a fixed point, not a penalty.** This is the headline and it
generalizes §13.6 substantially. The six natural start sets span pTM 0.369-0.650; after
the alpha=0 *naive* pass — no steering, no GLP — all six land between 0.227 and 0.315:

| start set | natural | after the loop | lost |
|---|---|---|---|
| instability, stable | 0.650 | 0.294 | -55% |
| hydrophobicity, hydrophobic | 0.629 | 0.315 | -50% |
| charge, anionic | 0.507 | 0.256 | -50% |
| charge, cationic | 0.438 | 0.257 | -41% |
| hydrophobicity, hydrophilic | 0.373 | 0.274 | -27% |
| instability, unstable | 0.369 | 0.227 | -38% |

Regressing post on pre: `post = 0.171 + 0.200 x start` (r=+0.80), i.e. **only ~20% of a
starting set's foldability advantage survives**, with a fixed point at pTM 0.214. The
correlation between how well a set folds and how much it loses is **r=-0.98**. So
masked-Gibbs decoding at 30 rounds / T=1.0 does not scale foldability down; it drives any
protein toward ~0.21-0.27 regardless of where it started, and the better it started the
more it loses. "The loop roughly halves foldability" was an artifact of the two
hydrophobicity sets happening to sit near 0.4-0.6.

**The correction's own foldability cost is now a bounded null — this open item is closed.**
§13.9 recorded "three marginal signals in the same direction (p=0.076, 0.029, 0.42)".
Pooling 235 paired alpha=0 comparisons across all three properties gives **-0.012 pTM
[95% CI -0.029, +0.005], p=0.43** (per property: -0.024 p=0.145, -0.004 p=0.707, -0.009
p=0.856). The earlier signal does not survive triple the data. Do not repeat the claim.

**Steering strength is close to free, with one marginal exception.** Corrected arm,
alpha=0 vs the top alpha, paired by starting protein:

| property | ΔpTM | p | pass rate (non-degenerate AND pTM>=0.30) |
|---|---|---|---|
| hydrophobicity | -0.004 | 0.70 | 35% -> 28% (Fisher p=0.39) |
| charge | -0.025 | 0.18 | 25% -> 16% (p=0.24) |
| instability | -0.031 | **0.014** | 25% -> 16% (p=0.24) |

Only instability is significant, and it is one of three tests (Holm-adjusted p=0.042 — it
survives, barely). The hydrophobicity result "the effect is bought without further
structural cost" replicates exactly; treat the instability decline as suggestive. No pass
rate moves significantly.

**Degeneracy stays low and the correction slightly reduces it**: naive 2.9% / 2.1% / 2.9%,
corrected 2.5% / 1.6% / 1.2%.

**The naturalness claim still fails, now for all three properties.** No cell of any run
approaches its own natural starting set. The reason remains the decoding loop, not GLP.

### 14.6 Still open

- Whether a lighter decoding budget moves the loop's fixed point (§14.5b) or
  merely gets closer to it more slowly. The 5-round/T=0.5 setting reached pTM
  0.332/0.435 from 0.373/0.629, which is above the 0.214 fixed point — so the
  fixed point is a property of the budget, not of the method, and the natural
  next question is where it sits as a function of coverage.
- Whether steering all layers and all tokens (Huang et al.) reaches the
  order-level component our single-layer, masked-positions-only intervention does
  not (§14.5a). This is now the most interesting open question in the experiment.
- The charge->instability leak (+0.60) is unexplained. instability->gravy (-0.54)
  is explained by §14.5a.
- The natural-correlation table does not include *net* charge, so the
  counterfactual that motivates the per-residue decision ("ranking on net charge
  would have ranked on length, r=?") is asserted rather than measured. One extra
  4-minute pool scan would settle it.
- Exp A has still never been re-run.
