# Steering solutions — fixing the on-manifold-doesn't-reach-the-output failure

Companion to `docs/steering_history.md` (the diagnosis) and the paper's steering section.
Written 2026-08-23 after establishing, with real data, that the GLP loss-restoration premise
holds for an activation **in isolation** but **fails for the sequences steering produces** —
for **both** steering mechanisms.

## The root cause, stated precisely

Two **coupled** failures with one origin:

1. **The correction can't survive the round-trip.** GLP is a prior over a *continuous*
   activation manifold, applied to an *intermediate* activation. The objective lives in
   *discrete* sequence space, and the map activation → tokens → re-embed is **lossy and
   non-invertible**. GLP places `h'` on the manifold; the ESM head uses `h'` only to *pick
   tokens*; re-embedding those tokens yields whatever ESM computes for that sequence, with no
   constraint that it be GLP-typical. Confirmed **decoder-agnostic**: fails for both the
   diff-of-means loop (mask 10% / round) and the structural loop (regenerate the whole
   sequence, `mask_fraction=1.0`).
2. **Even on-manifold ≠ foldable.** GLP loss is a proxy that does not track pTM.

Supporting numbers (finished generations re-embedded; real-protein loss = 0.869):
diff-of-means corrected 0.76–0.78 vs naive 0.74–0.75 (all 3 properties);
structural corrected 0.74–0.81 vs naive 0.73–0.78 (coiled coil / beta strand / zinc finger
`p<1e-17`; helix null). Corrected is **not lower — marginally higher**; both arms below real.

**Honest reframe:** activation-space typicality applied at an *intermediate* step is the wrong
lever for discrete generation. The fixes either make the discretization respect the prior, or
optimize the thing we actually care about (foldability).

## The solution ladder (prioritized)

### Tier 1 — turn GLP from an in-loop *corrector* into an output-space *guide / verifier*
Highest ROI; directly targets the diagnosis. This is the "diffusion-model-as-plug-and-play-prior"
framing (Graikos et al., NeurIPS 2022): use the model's score as a regularizer over the
*output*, not a one-shot projection on an intermediate.

- **(a) Best-of-n / rejection on the FINAL sequence.** Generate n steered candidates, re-embed,
  keep those on-manifold *and* folded. Trivial, robust, optimizes the quantity we measured.
  **Gate on pTM**, using GLP loss only as a cheap pre-filter (GLP loss alone is gameable). Also
  answers a scientific question immediately: is any on-manifold+foldable sequence *reachable*
  by steering at all?
- **(b) Reward-guided test-time refinement** — the principled version of (a).
  Reward-Guided Iterative Refinement at Test-Time (protein/DNA, arXiv:2502.14944); ProteinGuide
  (arXiv:2505.04823).

### Tier 2 — guide the decoder so the correction is never discarded
The token sampling is unaware of the prior; make it aware **at the logits**.

- **(c) Logit-space guidance.** Plug-and-Play Guidance for Discrete Diffusion via
  Gradient-Informed Logit Correction (arXiv:2606.06303) — almost exactly our failure mode:
  correct the *head's logits* with a (Jacobian-free) prior gradient so discretization respects
  the prior; benchmarked on protein sequences. Most on-point recent method.
- **(d) Split-Gibbs posterior sampling** (arXiv:2503.01161) — alternate a continuous prior step
  (GLP) with a discrete data-consistency step via a coupling variable; the rigorous version of
  "iterate to a self-consistent fixed point."

### Tier 3 — a GLP-native self-consistency ablation (cheap, do first)
The current loop corrects an intermediate and stops. Instead iterate
**decode → re-embed → GLP-correct → decode** until the *re-embedded* activation of the output
stops moving. This is alternating projection onto (decodable sequences ∩ GLP manifold) —
POCS / PnP-ADMM. No new training; directly optimizes the broken quantity. If the two sets do
**not** intersect (no sequence is decodable-consistent *and* GLP-typical *and* foldable), that
is itself a clean, publishable result.

### Tier 4 — fix the decoder's structural blindness (the other, foldability bottleneck)
Independent-position resampling destroys coevolution regardless of guidance; even perfect
guidance can't fix a structure-blind decoder. Replace it with one that generates *jointly
coherent* sequences:

- **Confidence-based unmasking** (MaskGIT-style: freeze high-confidence tokens, resample only
  low-confidence — the "confidence-frozen remasking" already flagged as the first decoder fix).
- A purpose-built **discrete-diffusion PLM decoder** (DPLM / ESM3-style masked discrete
  diffusion), which the field has moved to for exactly this reason.
- **Steer all layers / all tokens** (Huang et al. protocol; the paper's untested deviation):
  correcting one layer at masked positions leaves the decoder maximal freedom to wander
  off-manifold.

### Tier 5 — the meta-point
Don't try to make the *intermediate* correction survive — that is fighting the discretization.
Either move GLP to where its signal is real (score/verify the *output*, Tier 1) or make the
*decode step itself* prior-aware (Tier 2); in both cases validate against **foldability**, not
GLP loss. **Tiers 2 + 4 combined** attack both coupled failures at once, because a prior-aware,
structure-respecting decoder is exactly what produces sequences whose re-embedded activations
are both on-manifold and foldable.

## What to do first (concrete order)
1. **Best-of-n against pTM** (Tier 1a) — one afternoon; tells you whether the target is
   reachable, and if so gives a headline positive result (GLP-as-verifier).
2. **Self-consistency loop** (Tier 3) — cheap, GLP-native probe of the round-trip.
3. **Logit-space guidance** (Tier 2c) — the principled fix for the diagnosed discretization loss.
4. **In parallel, a confidence-based / discrete-diffusion decoder** (Tier 4) — addresses the
   foldability fixed point, which Tiers 1–3 don't.

## References
- Diffusion models as plug-and-play priors — Graikos et al., NeurIPS 2022.
  https://proceedings.neurips.cc/paper_files/paper/2022/file/5e6cec2a9520708381fe520246018e8b-Paper-Conference.pdf
- Plug-and-Play Guidance for Discrete Diffusion via Gradient-Informed Logit Correction —
  arXiv:2606.06303. https://arxiv.org/abs/2606.06303
- Split Gibbs Discrete Diffusion Posterior Sampling — arXiv:2503.01161.
  https://arxiv.org/pdf/2503.01161
- Reward-Guided Iterative Refinement at Test-Time (protein/DNA) — arXiv:2502.14944.
  https://arxiv.org/pdf/2502.14944
- ProteinGuide: on-the-fly property guidance — arXiv:2505.04823. https://arxiv.org/pdf/2505.04823
- Steering Generative Models with Experimental Data for Protein Fitness — arXiv:2505.15093.
  https://arxiv.org/pdf/2505.15093
- Controlling Repetition in PLMs / UCCS activation steering — arXiv:2602.00782.
  https://arxiv.org/html/2602.00782v1
- MaskGIT — Chang et al., CVPR 2022 (cited in the paper as `chang2022maskgit`).
