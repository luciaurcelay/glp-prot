# On-manifold steering — methodology (current)

What the steering pipeline does today, in plain terms. No results and no
history of what changed and why — for those see `docs/steering_history.md`
(§13 is the current state). Tracks PLAN.md §6.6/W4.

**Before anything else:** run `scripts/apply_interplm_patches.sh`. InterPLM's
pinned `sae/intervention.py` targets an older `nnsight` API and generation dies
with an `expand()` shape error without the patch. Re-run it after any
`git submodule update` (SETUP.md step 0).

## The idea being tested

"Steering" a protein language model means pushing its internal activations
toward some target concept (e.g. "more hydrophobic") and seeing what sequence
comes out. Pushed too hard, the activations end up somewhere the model never
saw in training — off the manifold of real protein activations.

GLP is a diffusion model of those activations. The thesis: after steering
pushes an activation off-manifold, using GLP to pull it back (a cleanup step)
should reach a stronger concept effect at a given naturalness than steering
alone.

**Two claims, tested separately, and they have come apart.** The *efficacy*
claim (GLP-correction makes steering work) is supported. The *naturalness*
claim (corrected output is more protein-like) is not, and the reason has
nothing to do with GLP — see §"What the generation loop costs" below.

## The two experiments

**Experiment A** — SAE-feature steering on zinc-finger (`--experiment A`).
InterPLM's released SAE on ESM-2-8M layer 4; a zinc-finger feature is pushed up
by `clamp` (hard-set) or `scale_shift` (`z* = a·z + b`). Scored by a strict
C2H2 regex plus ESMFold pLDDT. **Not re-run since the Exp B rebuild** — none of
the corrections below have been applied to it.

**Experiment B** — diff-of-means steering on a whole-sequence property
(`--experiment B --property {gravy,charge,instability}`). A direction `v` is
built from the difference between top- and bottom-quartile real sequences'
activations; steering is `h' = renorm(h + alpha·v)`.

The three properties are Banerjee et al.'s set, and only `gravy` has been run:

| `--property` | scored as | high / low quartile |
|---|---|---|
| `gravy` | Kyte-Doolittle mean hydropathy | hydrophobic / hydrophilic |
| `charge` | net charge at pH 7 **per residue** | cationic / anionic |
| `instability` | Guruprasad instability index | unstable / stable |

Charge is per residue on purpose: net charge is a sum, so quartiles of it are
partly quartiles of length, and the vector would encode length. The other two
scores are already length-normalized.

Every generation is scored on **all three** regardless of which was steered
(`delta_gravy`, `delta_charge`, `delta_instability` are always present), which is
what `scripts/analyze_property_specificity.py` reads. Off-target effects cannot
be recovered later without regenerating, so they are always written.

Exp B is **bidirectional** by default (`--directions auto` → both when the
contrast file supports it). "up" starts from the held-out *low*-quartile
proteins and applies `+alpha·v`; "down" starts from the held-out *high*-quartile
proteins and applies `−alpha·v`. Both held-out sets are carved out of their quartile before
the vector is computed, so neither direction's starting points contaminate it.

Bidirectional matters because it is the only design that separates real
steering from a direction-independent artifact: a real vector pushes the two
trajectories *apart*, an artifact pushes both the same way and cancels in the
difference.

## The shared pipeline

**Part 0 — off-manifold diagnostic (no generation).** Clamp real held-out
activations at increasing `alpha` and measure GLP's diffusion loss; it should
rise. It additionally applies **one** SDEdit pass to those clamped activations
and re-measures, which isolates the correction operator from the generation
loop.

**Part 1 — generation loop** (`masked_gibbs_generate`). From an initial
sequence, repeat for `n_rounds`: pick positions to mutate → forward → apply the
steering intervention to those positions' activations → **optionally**
`glp_correct` → forward again with that override → resample those positions.
Every alpha runs twice, with and without the correction; that pairing is the
comparison.

*Site selection* (`--mask-selection`): `random` (default) masks a uniform
`mask_fraction`; `relatedness` implements Huang et al.'s Algorithm 1 — score
each token by cosine similarity to `v` and mutate the `--n-mutations` most
opposed. Relatedness was tested and **not adopted**: it preserves twice as much
of the protein but costs ~6× of the steering effect here (history §13.7).

**The cleanup step (`glp_correct`) — SDEdit.** Mix the steered activation with
noise up to `t_start`, then run GLP's reverse process back to 0. `t_start`
controls how much of the edit survives. **Default is now experiment-dependent:
0.5 for A (its own validated value), 0.7 for B.** For Exp B, 0.5–0.6 is a
pathological band — most drift, worst composition, and misleadingly good
pseudo-perplexity.

**Part 2 — per-sequence scoring.** Concept score, pseudo-perplexity, GLP
diffusion loss on the finished sequence, degeneracy checks, and
`composition_distance` (L1 to UniProtKB background composition). pLDDT/pTM come
from a **separate stage** (below), not from this loop.

**Part 3 — analysis.** Every per-alpha statistic is computed **within a
direction** — pooling would average a rising trajectory against a falling one
and cancel the effect.

## Measuring the effect: what to quote

**Quote `steering_attributable`, never the raw corrected-vs-naive gap.** At
alpha=0 the arms differ only by whether the correction ran, and the correction
moves ΔGRAVY by itself (+1.15 at 30 rounds); that drift was 65–97% of the
corrected arm's raw effect at alpha ≤ 2. `steering_attributable` differences
each arm against its own alpha=0 run per init sequence, removing both the drift
and per-init variation.

**`divergence` is the headline estimand** for bidirectional runs: up minus
down, reported raw and attributable, with an *unpaired* Mann–Whitney (the two
directions use different sequences).

## Naturalness: which axis, and why

**Do not use pseudo-perplexity or pLDDT as the naturalness axis.** On this
generation setup, the sequences the degeneracy gate rejects score *better* on
both than our clean output — poly-Leu/Val junk beats real ubiquitin on pLDDT.
pTM is the least bad and is what `analyze_steering_naturalness.py` uses by
default, but on the larger run even pTM is merely indistinguishable from junk.

**Check metrics on the rejected rows, not just the kept ones.** On kept rows
alone pppl and pLDDT correlate correctly with each other, which looks like
mutual validation and is shared bias — the gate has already removed the rows
that expose it.

**Use quality thresholds as outcomes, never filters.** `pass_rate_*` reports
the fraction of *all* generations that are non-degenerate **and** above
threshold, so degenerate output counts as failure. Filtering on pLDDT≥70
instead would keep 87% of the degenerate sequences and 12% of the clean ones.

## The degeneracy gate

`is_degenerate_sequence`: ≥30% one residue, ≤8 distinct amino acids, or a
≥12-long homopolymer run. Calibrated against real proteins — it rejects 3/80
(3.8%) of the natural held-out starts, comparable to the 2.9% of our own
output, so it is not tuned to target our failure mode.

**Do not tighten it on sequence complexity.** Complexity is affected by the
treatment (steering toward hydrophobicity draws on a smaller sub-alphabet), so
a complexity cut is post-treatment selection that removes the successfully
steered sequences. SEG's standard 2.2-bit trigger would also reject 75% of the
natural starting proteins.

## What the generation loop costs

Folding the 80 natural starting proteins gives pTM 0.373 (low-GRAVY set) /
0.629 (high-GRAVY set). The alpha=0 **naive** arm — no steering, no GLP — comes
out at 0.275 / 0.319. **The masked-Gibbs loop roughly halves foldability on its
own**, before any steering or correction, and both arms land in the same place.

Two settings trade this off; both are defensible and support different claims:

| | divergence @alpha=5 | pTM up / down | degenerate | cost |
|---|---|---|---|---|
| 30 rounds, T=1.0 | +3.38 | 0.275 / 0.319 | 2.7% | 1× |
| 5 rounds, T=0.5 | +1.56 | 0.332 / 0.435 | 3.4% | 1/6× |

**Do not raise rounds at T=0.5** — it mode-collapses (8/15 non-proteins at 30
rounds), and the collapsed sequences score *high* pTM, so the failure looks
like an improvement in any analysis that drops degenerate rows instead of
counting them.

Always compare against the held-out set the run actually generates from
(0.373 or 0.629), never the pooled average.

## Key knobs

| flag | what it controls |
|---|---|
| `--exp-name` | **required**, unique — metrics/figures redirect is keyed on it |
| `--property` | `gravy` (default), `charge` or `instability` |
| `--alpha-mode` | `raw` (default) or `relative`. **`relative` for any cross-property comparison** — see below |
| `--match-length` | force the contrast sets onto a shared length distribution (build-time only) |
| `--alphas` | steering strengths to sweep |
| `--directions` | `auto` (default), `up`, `down`, or `up,down` |
| `--t-start` | SDEdit noise level. Default 0.5 for A, **0.7 for B** |
| `--mask-fraction` / `--n-rounds` | edit budget. Default 1.0/100 for A, 0.10/30 for B |
| `--temperature` / `--sample-mode` | resampling sharpness (default 1.0 / `sample`) |
| `--mask-selection` / `--n-mutations` | `random` (default) or `relatedness` |
| `--fold` | ESMFold pLDDT in-loop. **Exp A only** — for Exp B use the separate fold stage |
| `--t-start-pilot` / `--generation-pilot` | sweeps that exit before the main run |

## Comparing properties: alpha has to mean the same thing

`h + alpha·v` rotates `h` by an angle set by `alpha·||v||/||h||`, and the three
properties' vectors have different norms — so the same alpha grid is three
different perturbation sizes. `--alpha-mode relative` reinterprets alpha as the
fraction of `mean||h||` added along `v/||v||`, which is comparable by
construction. `raw` remains the default because every existing run used it; a
three-property table must use `relative` throughout, including a re-run of
hydrophobicity.

The contrast build also reports the **length gap** between its two sets in
pool-SD units and warns above 0.25. If it warns, rebuild with `--match-length
--restart` — otherwise the vector partly encodes length rather than the property.

## Running it end to end

```bash
sbatch --account=ehpc394 scripts/cluster/submit_steering.sh B <UNIQUE-NAME> <config> "<alphas>" "" \
  --property charge --alpha-mode relative \
  --ckpt <...>/glp_final.pt --scaler <...>/rep_statistics.pt
sbatch --account=ehpc394 scripts/cluster/submit_fold.sh <run-dir>      # ESMFold2 pLDDT/pTM
$GLP_VENV/bin/python scripts/plot_steering_report.py       --run-dir <run-dir>
$GLP_VENV/bin/python scripts/analyze_steering_naturalness.py --run-dir <run-dir>

# once two or more properties have been run:
$GLP_VENV/bin/python scripts/analyze_property_specificity.py \
  --run-dirs <gravy-run>,<charge-run>,<instability-run> --out-run-dir <run-dir>
```

`--alpha-mode relative` **requires an explicit `--alphas`** — the raw default grid means a
different perturbation in those units, and the point of relative mode is one grid shared
across properties, which cannot be derived from a single property's scale. Run the
contrast builds first and pick the grid from the `alpha_scale` each one reports.

The specificity matrix defaults to `--match-on effect`: each property is read at its own
alpha, chosen so the on-target divergence is closest to `--target-effect` (1 SD). Reading
all three at one alpha instead (`--match-on alpha`) makes whichever property was pushed
hardest look least specific, which is a fact about the grid, not the direction. A property
whose grid does not bracket the target is flagged as unmatched rather than silently used.

The fold stage runs in a **separate Python 3.12 venv** (ESMFold2 needs `esm`,
which the project venv does not have and cannot get — MN5 is offline), so its
launcher deliberately does not source `env_mn5.sh`. ~2.6 s/sequence, resumable.

## Outputs

`runs/<run>/steering_diffmean_{hydro,charge,instability}_<name>/` —
per-sequence CSV (`expB_diffmean_<property>.csv`),
`fold_confidence.csv` sidecar (keyed on `sample_id`, with `seq_sha1` so it
cannot be merged against regenerated sequences), `metrics/summary.json`,
`run_summary.md`, and `figures/`.

Figures describe **what is plotted, not what it means** — interpretation
belongs in the paper caption, and a claim baked into a PNG went stale against
its own panel once already.

| figure | shows |
|---|---|
| `fig_divergence` | the two trajectories separating; the headline |
| `fig_dose_response` | effect and structural confidence vs alpha, shared axis |
| `fig_baseline_drift` | what the correction does with no steering |
| `naturalness_axis_gate_check` | which metrics the rejected sequences fool |
| `pass_rate_*` | confidently-folded yield as an outcome |
| `off_manifold_diagnostic` | GLP loss vs alpha, with one correction pass |
| `generation_tradeoff` | operating points (from `plot_generation_tradeoff.py`) |
| `property_specificity` | steered × measured matrix, and its residual after the natural coupling |
