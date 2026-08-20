#!/usr/bin/env python
"""On-manifold steering (PLAN §11, W4) — glprot_steering_spec.md v2.

Two experiments testing whether GLP correction rescues off-manifold steering. Each
follows its own source methodology faithfully; they are deliberately **not**
mechanically identical:

  Exp A (`--experiment A`, default out-dir `steering_sae_zinc`): SAE-feature clamp on
    a zinc-finger feature, following `plm-sae` (arXiv:2502.09135) §3.4 — random-AA
    init, full-position resample every round, 100 rounds, keep the round that
    maximizes the clamped feature's own activation (`z_k`) EXCEPT the no-clamp
    anchor arm, which keeps the last round (selection caveat, spec Part 2A).
    Concept score: binary C2H2-pattern hit, aggregated to a hit rate per alpha.
    Needs ESMFold (pLDDT, secondary naturalness metric).

  Exp B (`--experiment B`, default out-dir `steering_diffmean_hydro`): DiffMean /
    contrast-vector steering on hydrophobicity (GRAVY), following Huang et al.
    (arXiv:2509.07983) §4.1.1 — real held-out low-GRAVY sequence init, 10%-mask
    partial refinement, 20-50 rounds, always keep the last round. Concept score:
    continuous ΔGRAVY (final - init). No folding. Needs
    `scripts/build_property_contrast.py` to have been run first (auto-built here if
    missing — see `--contrast`). `--property` picks which of hydrophobicity/charge/
    instability is steered.

Both share: Part 0 (off-manifold diagnostic — does steering raise GLP diffusion loss,
worse at higher alpha?), Part 1 (`glp_correct`, the SDEdit activation-space cleanup),
and Part 3 analysis (Pareto plot, three-regime breakdown, rescue sanity check). See
`docs/PLAN.md` §6.6 and the pasted spec for the full protocol.

`plm-sae` (github.com/edithvillegas/plm-sae, the codebase §3.4 names) was checked
directly (not assumed absent): its `interpretability/` scripts only correlate SAE
activations with Swiss-Prot annotation spans (no sequence generation, no
masking/refinement loop, no C2H2 or any other pattern detector — structural or
otherwise). Its own steering pseudocode's "C2H2 pattern" is exactly the sequence regex
below, not a folded-structure geometric check; folding here supplies pLDDT (naturalness),
not concept detection. It also ships its own SAE (`evillegasgarcia/sae_esm2_6_l3`,
same hook layer, different architecture/training run) — not used here; this experiment
clamps InterPLM's SAE, per the spec's own Setup section.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import _bootstrap  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F

from glp_plm.config import load_config
from glp_plm.data import swissprot_labels as sl
from glp_plm.data.activation_extractor import ActivationExtractor
from glp_plm.diffusion import flow_matching as fm
from glp_plm.diffusion.sampler import sample_on_manifold
from glp_plm.diffusion.scaler import Scaler
from glp_plm.eval import baselines as bl
from glp_plm.eval import probing_1d as pr
from glp_plm.eval import probing_f1 as pf
from glp_plm.eval.probing_1d import balanced_indices
from glp_plm.eval.steering_properties import (PROPERTIES, SCORE_ALL_KEYS, get_property,
                                              score_all)
from glp_plm.experiments import RunLogger
from glp_plm.experiments import style
from glp_plm.model import GLPDenoiser
from glp_plm.utils import get_device, resolve_config_data_dir, resolve_storage_dir, set_seed

import re

AA20 = "ACDEFGHIKLMNPQRSTVWY"
# Spec's C2H2 zinc-finger pattern: Cys-X(2-4)-Cys-X(12)-His-X(3-5)-His.
C2H2_RE = re.compile(r"C.{2,4}C.{12}H.{3,5}H")
# Looser variant (wider spacer tolerances) to check whether the strict spec pattern is
# undercounting real zinc-finger-like structure the strict regex just misses by a residue
# or two, vs. the motif genuinely not being there.
C2H2_RE_LOOSE = re.compile(r"C.{1,6}C.{9,15}H.{1,7}H")


# --------------------------------------------------------------------------- #
#  Steering primitives (SAE clamp / diff-of-means) — both raw-activation-in/out
# --------------------------------------------------------------------------- #
def sae_clamp(sae, x: torch.Tensor, feat_idxs, alpha: float) -> torch.Tensor:
    """Encode `x`, clamp feature(s) `feat_idxs` (int or list of ints — clamping more
    than one simultaneously is how arXiv:2502.09135 gets from 1.7% to 13% zinc-finger
    hits, see `select_zinc_finger_features`) to `alpha * activation_rescale_factor`
    each, decode back to raw activation space. `x` is raw (un-standardized).

    Adds back the SAE's own reconstruction error (`x_norm - decode(encode(x_norm))`)
    on top of the clamped decode, i.e. `x* = fdec(f*) + (x - fdec(f))` — the
    recon+error convention `docs/INTERPLM_MAP.md`'s steering row points at, and the
    one arXiv:2502.09135 (§3.4 eq. 7-8) uses. Without it, everything the SAE
    dictionary fails to reconstruct (not just the target feature) gets zeroed out of
    the activation on every steered position — extra off-manifold distortion beyond
    what the clamp itself needs."""
    idxs = [feat_idxs] if isinstance(feat_idxs, int) else list(feat_idxs)
    x_norm, orig_norms = sae._normalize_input_and_get_norms(x)
    feats = sae.encode(x_norm).clone()
    error = x_norm - sae.decode(feats)
    for idx in idxs:
        feats[..., idx] = alpha * sae.activation_rescale_factor[idx]
    x_hat = sae.decode(feats) + error
    return sae._unnormalize_output(x_hat, orig_norms)


def sae_scale_shift(sae, x: torch.Tensor, feat_idxs, a: float, b: float) -> torch.Tensor:
    """z*_k = a*z_k + b (Villegas Garcia & Ansuini, arXiv:2502.09135 Eq. 7-8): scale-and-
    shift the feature's OWN naturally-arising activation, as opposed to `sae_clamp`'s hard
    set to an absolute target value regardless of what the model actually produced. Same
    recon-error add-back convention (Eq. 8) as `sae_clamp`.

    `a=1, b=0` is the algebraic identity (`feats` unmodified -> `x_hat = decode(feats) +
    (x_norm - decode(feats)) = x_norm` exactly, same cancellation `diffmean_clamp`'s
    alpha=0 relies on) — the natural no-intervention anchor for this formula, unlike
    `sae_clamp` where alpha=0 would ablate the feature (see `make_steer_fn`)."""
    idxs = [feat_idxs] if isinstance(feat_idxs, int) else list(feat_idxs)
    x_norm, orig_norms = sae._normalize_input_and_get_norms(x)
    feats = sae.encode(x_norm).clone()
    error = x_norm - sae.decode(feats)
    for idx in idxs:
        feats[..., idx] = a * feats[..., idx] + b
    x_hat = sae.decode(feats) + error
    return sae._unnormalize_output(x_hat, orig_norms)


def diffmean_clamp(v: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
    """h_tilde = h + alpha*v, renormalized to ||h|| (arXiv:2509.07983 Eq. 1: 'the edited
    activations are rescaled to have the same norm as h before being passed to the next
    layer'). `x`, `v`, and the output are all raw (un-standardized) activation space,
    same convention as `sae_clamp`. Unlike `sae_clamp`, alpha=0.0 is naturally the
    null-baseline (x_new == x exactly, renorm is a no-op) — no special-casing needed."""
    orig_norm = x.norm(dim=-1, keepdim=True)
    x_new = x + alpha * v.to(x.device)
    new_norm = x_new.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return x_new * (orig_norm / new_norm)


def select_zinc_finger_features(sae, processed: Path, layer: int, concept: str,
                                top_n: int = 1, n_shards: Optional[int] = None,
                                max_pos: int = 3000, neg_per_pos: float = 1.0,
                                shortlist_k: int = 64, seed: int = 0):
    """Top-`top_n` SAE features for `concept`, ranked by InterPLM's own validated
    criterion — domain-adjusted F1, selected on valid / confirmed on test
    (`glp_plm/eval/probing_f1.py`, a faithful port of
    `interplm/analysis/concepts/calculate_f1.py`, `docs/INTERPLM_MAP.md:16`) — not
    plain AUC. AUC is rank-only and can't tell a feature that's precise to the C2H2
    motif apart from one merely correlated with the *annotated span* (e.g. the
    Lys/Arg/Glu-rich linker most C2H2 arrays are strung together with): measured on
    an earlier steering pass, the AUC-only pick (feature 7033, val AUC 0.952) peaked
    on Arg/Glu/Lys in 79% of 90 generations and on His in 0% — a linker-composition
    proxy, not a genuine Cys-X(2-4)-Cys-X(12)-His-X(3-5)-His detector; clamping it
    can't produce a motif it was never detecting. F1 (precision-at-a-real-threshold,
    recall per domain instance) is the criterion that would have caught this.

    AUC is still used for a cheap `shortlist_k`-candidate shortlist on a balanced
    subsample (`pr.topk_filter`, same filter/budget run_probing.py's own F1 path
    uses) before the exact, prevalence-correct F1 pass, which needs the full split
    and would be too slow to run over all 10240 features from a cold start.

    arXiv:2502.09135 §4.1 measures a single clamped latent finding the C2H2 motif in
    only 3/180 generations (1.7%) vs. 24/180 (13%) when clamping its two most
    prominent zinc-finger latents together — `top_n=2` is there to close that gap,
    not just a knob."""
    acts_val, labels_val, names = sl.load_split(processed, "valid", layer, n_shards=n_shards)
    assert concept in names, f"{concept!r} not in eval store concepts (have e.g. {names[:5]}...)"
    acts_test, labels_test, _ = sl.load_split(processed, "test", layer, n_shards=n_shards)
    y_val_full = labels_val[concept].to_numpy().astype(int)
    y_test_full = labels_test[concept].to_numpy().astype(int)

    # Shortlist on a balanced subsample (mirrors run_probing.py's f1 path exactly:
    # the valid split is ~5M residues, and a dense (N, 10240) float32 SAE-feature
    # bank over all of them is ~207GB — an OOM kill at ~236GB RSS the hard way).
    sub_rows = balanced_indices((y_val_full > 0).astype(int), max_pos, neg_per_pos, seed=seed)
    sub_feats = bl.sae_bank(sae, acts_val[sub_rows].float()).numpy()
    shortlist = pr.topk_filter(sub_feats, (y_val_full[sub_rows] > 0).astype(int), shortlist_k)
    del sub_feats

    def bank(split: str, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        acts = acts_val if split == "valid" else acts_test
        return bl.sae_bank(sae, acts[rows].float()).numpy()[:, cols]

    # Repeated single-best F1 selection (InterPLM's own protocol, called top_n times,
    # dropping the previous winner each round) rather than reimplementing its grid —
    # `run_concept_f1` already does select-on-valid/confirm-on-test end to end.
    feat_idxs, val_f1s, test_f1s = [], [], []
    remaining = shortlist
    for _ in range(top_n):
        res = pf.run_concept_f1(bank, acts_val.shape[0], acts_test.shape[0],
                                y_val_full, y_test_full, remaining, concept, "sae")
        feat_idxs.append(res.feature_index)
        val_f1s.append(res.val_f1)
        test_f1s.append(res.test_f1)
        remaining = remaining[remaining != res.feature_index]
    return feat_idxs, val_f1s, test_f1s


# --------------------------------------------------------------------------- #
#  GLP: per-sample diffusion loss (Part 0) and SDEdit correction (Part 1)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def diffusion_loss(model, scaler: Scaler, acts_raw: torch.Tensor, n_repeats: int = 4,
                   device: str = "cpu", generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Per-sample flow-matching loss, averaged over `n_repeats` random (t, eps) draws
    (spec Part 0). Built from the same primitives as glp_plm.diffusion.flow_matching's
    scalar `flow_matching_loss`, just with reduction='none' + repeat-averaging."""
    z0 = scaler.standardize(acts_raw.float().to(device))
    losses = []
    for _ in range(n_repeats):
        t = fm.sample_t(z0.shape[0], device=device, generator=generator)
        eps = torch.randn(z0.shape, device=device, generator=generator, dtype=z0.dtype)
        z_t = fm.interpolate(z0, eps, t)
        target = fm.velocity_target(z0, eps)
        pred = model(z_t, t)
        losses.append(F.mse_loss(pred, target, reduction="none").mean(-1))
    return torch.stack(losses).mean(0).cpu()


@torch.no_grad()
def glp_correct(model, scaler: Scaler, acts_raw: torch.Tensor, t_start: float = 0.5,
                num_steps: int = 20, device: str = "cpu",
                generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """SDEdit cleanup: standardize -> sample_on_manifold -> destandardize. `acts_raw`
    is raw (un-standardized), same convention as `sae_clamp`'s input/output."""
    z0 = scaler.standardize(acts_raw.float().to(device))
    z_clean = sample_on_manifold(model, z0, num_steps=num_steps, t_start=t_start, generator=generator)
    return scaler.destandardize(z_clean).cpu()


# --------------------------------------------------------------------------- #
#  Part 0 — off-manifold diagnostic (shared by both experiments)
# --------------------------------------------------------------------------- #
def run_off_manifold_diagnostic(model, scaler, clamp_fn: Callable[[torch.Tensor, float], torch.Tensor],
                                alphas, baseline_acts, out_dir: Path, run: RunLogger, device: str,
                                n_repeats: int, seed: int, experiment: str,
                                correct_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
    """`clamp_fn(acts_raw, alpha) -> acts_raw` is steering-method-agnostic (SAE-clamp or
    diff-of-means) — Part 0 only needs the raw-activation-in/raw-activation-out contract.
    `experiment` ("A"/"B") is recorded as a CSV column per the spec's output schema, even
    though the two experiments already write to separate out-dirs."""
    g = torch.Generator(device=device).manual_seed(seed)
    loss_baseline = diffusion_loss(model, scaler, baseline_acts, n_repeats=n_repeats,
                                   device=device, generator=g)

    rows = [{"experiment": experiment, "alpha": "baseline", "corrected": False,
             "sample_idx": i, "diffusion_loss": float(v)} for i, v in enumerate(loss_baseline)]
    means = {"baseline": float(loss_baseline.mean())}
    means_corr: Dict[Any, float] = {}
    for alpha in alphas:
        steered = clamp_fn(baseline_acts.to(device), alpha)
        loss_a = diffusion_loss(model, scaler, steered, n_repeats=n_repeats,
                                device=device, generator=g)
        rows += [{"experiment": experiment, "alpha": alpha, "corrected": False,
                  "sample_idx": i, "diffusion_loss": float(v)} for i, v in enumerate(loss_a)]
        means[alpha] = float(loss_a.mean())
        line = (f"[diagnostic-{experiment}] alpha={alpha:<6} mean diffusion loss = "
                f"{means[alpha]:.5f}  (baseline {means['baseline']:.5f})")
        if correct_fn is not None:
            # One SDEdit pass on the clamped activations, NO generation loop. Isolates the
            # correction operator from the masked-Gibbs iteration: if the loss comes back to
            # ~baseline here but sequences still collapse at this alpha during generation,
            # the collapse is the Gibbs loop compounding, not SDEdit failing on a single
            # clamped activation (and vice versa).
            loss_c = diffusion_loss(model, scaler, correct_fn(steered), n_repeats=n_repeats,
                                    device=device, generator=g)
            rows += [{"experiment": experiment, "alpha": alpha, "corrected": True,
                      "sample_idx": i, "diffusion_loss": float(v)} for i, v in enumerate(loss_c)]
            means_corr[alpha] = float(loss_c.mean())
            line += f" -> corrected {means_corr[alpha]:.5f}"
            # Fraction of the clamp's excess loss one correction pass removes. Only meaningful
            # once the clamp actually moved the loss: at small alpha the excess is ~0 (or
            # negative — clamping can sit marginally *below* baseline), and dividing by it
            # produces garbage like "-1294% recovered". Report the residual-vs-baseline there
            # instead, which stays interpretable. Values >100% are real and mean the correction
            # pushed the activation to a *higher*-density region than the unsteered original.
            excess = means[alpha] - means["baseline"]
            if excess > 0.02 * means["baseline"]:
                line += (f" (recovers {(means[alpha] - means_corr[alpha]) / excess:+.1%} "
                         "of the excess)")
            else:
                line += (f" (clamp barely moved the loss: excess {excess:+.5f}; corrected sits "
                         f"{means_corr[alpha] - means['baseline']:+.5f} vs baseline)")
        print(line, flush=True)

    csv_path = out_dir / "off_manifold_diagnostic.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment", "alpha", "corrected", "sample_idx",
                                          "diffusion_loss"])
        w.writeheader()
        w.writerows(rows)

    monotonic = all(means[alphas[i]] <= means[alphas[i + 1]] + 1e-9 for i in range(len(alphas) - 1))
    if not monotonic:
        print(f"[diagnostic-{experiment}] WARNING: loss did not rise monotonically with alpha — "
              "reconsider the off-manifold framing before proceeding (spec Part 0 step 5).")
    run.log_summary(**{f"diagnostic_{experiment}_means": means,
                       f"diagnostic_{experiment}_monotonic": monotonic})
    if means_corr:
        run.log_summary(**{
            f"diagnostic_{experiment}_means_corrected": means_corr,
            # only where the clamp moved the loss enough for the ratio to mean anything
            f"diagnostic_{experiment}_excess_recovered": {
                str(a): ((means[a] - means_corr[a]) / (means[a] - means["baseline"]))
                for a in means_corr
                if means[a] - means["baseline"] > 0.02 * means["baseline"]},
        })

    style.apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.5))
    xs = [str(a) for a in alphas]
    sems = [float(np.std([r["diffusion_loss"] for r in rows if r["alpha"] == a]) / np.sqrt(
        max(sum(1 for r in rows if r["alpha"] == a), 1))) for a in alphas]
    ax.errorbar(xs, [means[a] for a in alphas], yerr=[1.96 * s for s in sems],
               marker="o", color=style.steering_arm_color(False), label="steered")
    if means_corr:
        sems_c = [float(np.std([r["diffusion_loss"] for r in rows
                                if r["alpha"] == a and r["corrected"]]) / np.sqrt(
            max(sum(1 for r in rows if r["alpha"] == a and r["corrected"]), 1))) for a in alphas]
        ax.errorbar(xs, [means_corr[a] for a in alphas], yerr=[1.96 * s for s in sems_c],
                   marker="o", color=style.steering_arm_color(True),
                   label="steered + one correction pass (no generation)")
    ax.axhline(means["baseline"], color=style.BASELINE, linestyle="--", label="baseline (unsteered)")
    ax.set_xlabel("steering strength alpha")
    ax.set_ylabel("GLP diffusion loss (mean ± 95% CI)")
    ax.set_title("GLP diffusion loss vs. steering strength")
    ax.legend()
    fig_path = style.save_fig(fig, run.figures_dir / "off_manifold_diagnostic.png")
    print(f"[diagnostic-{experiment}] wrote {csv_path} + {fig_path} (monotonic={monotonic})")


def _stub_ipython_if_broken() -> None:
    """MN5's PYTHON/3.11.5 module ships a `_sqlite3` extension built against
    `sqlite3_deserialize`, which RHEL9's system libsqlite3 (`sqlite-libs-3.34.1`,
    built without SQLITE_ENABLE_DESERIALIZE) doesn't export — so plain `import
    sqlite3` fails there. `nnsight` only imports IPython for `get_ipython()`, to
    detect an interactive/notebook session (always False in a batch job); the
    real IPython package pulls in its `history` submodule, which needs sqlite3.
    Stub just that one function instead of depending on a cluster library fix —
    a no-op everywhere the real import already works fine (e.g. local dev).

    matplotlib's figure-canvas init also probes `IPython.version_info` (to
    decide whether it needs to patch an interactive-backend GUI loop, N/A in
    a batch job) — give the stub a `version_info` past its `>= (8, 24)` gate
    so it takes the same no-op branch as "IPython not installed"."""
    import sys

    if "IPython" in sys.modules:
        return
    try:
        import IPython  # noqa: F401
    except Exception:
        import types
        stub = types.ModuleType("IPython")
        stub.get_ipython = lambda: None
        stub.version_info = (8, 24, 0, "")
        sys.modules["IPython"] = stub


# --------------------------------------------------------------------------- #
#  Part 1/2 — shared masked-Gibbs generation loop with per-round steering
# --------------------------------------------------------------------------- #
class ESMSteerer:
    """ESM2 forward pass with selective hidden-state override at the hook layer.

    Wraps InterPLM's own nnsight-based steering entrypoint
    (interplm.sae.intervention.get_esm_output_with_intervention) rather than a
    hand-rolled forward hook — it already handles the layer-i/i+1 input/output
    namespace clash correctly (see that module's docstring)."""

    def __init__(self, model_name: str, layer: int, device: str):
        _stub_ipython_if_broken()
        from nnsight import NNsight
        from transformers import AutoTokenizer, EsmForMaskedLM

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmForMaskedLM.from_pretrained(model_name).to(device).eval()
        # nnsight 0.7's Envoy.__init__ (NNsight's base class) takes no `device` kwarg
        # (InterPLM's fidelity.py reference targets nnsight>=0.5.7's older API, which
        # did) — the model is already placed on `device` above, nothing else needed.
        self.nn_model = NNsight(self.model)
        self.layer = layer
        self.device = device
        self.valid_aa_ids = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(a) for a in AA20], dtype=torch.long
        )

    def forward(self, token_ids: torch.Tensor, attn_mask: torch.Tensor,
               override: Optional[torch.Tensor] = None):
        from interplm.sae.intervention import get_esm_output_with_intervention

        return get_esm_output_with_intervention(
            self.model, self.nn_model, token_ids, attn_mask, self.layer, override
        )


def random_aa_init_tokens(tok, length: int, generator: torch.Generator) -> torch.Tensor:
    """Uniform-random canonical-AA init (spec Part 2A step 1: "full-length sequence of
    randomly sampled amino acids, not [MASK]") -> (1, length+2) with cls/eos."""
    aa_ids = torch.tensor([tok.convert_tokens_to_ids(a) for a in AA20], dtype=torch.long)
    idx = torch.randint(0, len(AA20), (length,), generator=generator)
    tokens = torch.full((1, length + 2), tok.pad_token_id, dtype=torch.long)
    tokens[0, 0], tokens[0, -1] = tok.cls_token_id, tok.eos_token_id
    tokens[0, 1:length + 1] = aa_ids[idx]
    return tokens


def seq_to_init_tokens(tok, seq: str) -> torch.Tensor:
    """Tokenize a real sequence (spec Part 2B step 1: a held-out low-GRAVY starting
    point) -> (1, len(seq)+2) with cls/eos."""
    ids = [tok.cls_token_id] + [tok.convert_tokens_to_ids(a) for a in seq] + [tok.eos_token_id]
    return torch.tensor([ids], dtype=torch.long)


def _sample_positions(logits_slice: torch.Tensor, valid_ids: torch.Tensor,
                      mode: str, temperature: float,
                      generator: Optional[torch.Generator]) -> torch.Tensor:
    """logits_slice: (k, vocab) on CPU. Restricted to the 20 canonical AA tokens."""
    masked = torch.full_like(logits_slice, float("-inf"))
    masked[:, valid_ids] = logits_slice[:, valid_ids]
    if mode == "argmax":
        return masked.argmax(dim=-1)
    probs = torch.softmax(masked / temperature, dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


def masked_gibbs_generate(steerer: ESMSteerer, init_tokens: torch.Tensor,
                          steer_fn: Callable[[torch.Tensor], torch.Tensor],
                          correct_fn: Callable[[torch.Tensor], torch.Tensor],
                          mask_fraction: float, n_rounds: int, stabilize_tol: float,
                          sample_mode: str, temperature: float,
                          generator: torch.Generator, device: str,
                          selection: str = "random", n_mutations: Optional[int] = None,
                          score_vec: Optional[torch.Tensor] = None,
                          diag_fn: Optional[Callable[[int, torch.Tensor, torch.Tensor], None]] = None,
                          select_fn: Optional[Callable[[int, str], float]] = None,
                          log_fn: Optional[Callable[[int, str], None]] = None
                          ) -> str:
    """Shared masked-Gibbs refinement loop (spec Part 2A/2B): starting from
    `init_tokens` (already containing cls/eos, shape (1, L+2)), run `n_rounds` of
    (mask a `mask_fraction`-random subset of the L interior positions -> forward ->
    steer+correct those positions' layer-4 activations (UPDATE-TOKENS) -> forward
    again with the override -> resample). `mask_fraction=1.0` makes every round a
    full-sequence resample (Exp A, plm-sae §3.4); a smaller fraction makes it an
    incremental partial refinement (Exp B, Huang et al. §4.1.1). There is no
    unsteered warm-up round: every one of `n_rounds` is steered — `init_tokens`
    itself supplies the starting sequence directly (random-AA for A, a real held-out
    sequence for B).

    `diag_fn(round_idx, clamped, corrected)`, if given, is called each round with the
    steered activations before SDEdit and the same positions after — lets a caller
    measure whether the correction is doing anything at the point it's applied.

    `select_fn(round_idx, seq) -> float`, if given, is called after every round
    (round_idx=-1 for `init_tokens` itself) with the sequence decoded so far, scoring
    it from the sequence's own natural (unclamped) properties. The round with the
    highest score is returned instead of unconditionally the last one — Exp A's
    `z_k`-argmax selection (arXiv:2502.09135 §3.4); pass `select_fn=None` for Exp B
    (always last round, and for Exp A's no-clamp anchor arm — spec's selection
    caveat: applying the argmax criterion to an *unsteered* run just cherry-picks the
    best-looking round out of `n_rounds`, inflating the anchor).

    `log_fn(round_idx, seq)`, if given, is called every round purely for
    diagnostic/pilot logging (e.g. per-round GRAVY trajectory for Exp B's round-count
    pilot) — unlike `select_fn` it has no effect on which round is returned."""
    tok = steerer.tokenizer
    tokens = init_tokens.clone().to(device)
    L = tokens.shape[1] - 2
    attn = torch.ones_like(tokens)

    def decode() -> str:
        ids = tokens[0, 1:L + 1].tolist()
        return "".join(tok.convert_ids_to_tokens(i) for i in ids)

    best_seq, best_score = None, float("-inf")
    if select_fn is not None:
        best_seq = decode()
        best_score = select_fn(-1, best_seq)

    # relatedness and confidence both rank positions and take a fixed edit budget; honor
    # --n-mutations if set, else default to the same round(mask_fraction * L) the random loop
    # uses so the modes are compared at an identical number of edits per round.
    k = n_mutations if (selection in ("relatedness", "confidence") and n_mutations) else max(1, round(mask_fraction * L))
    k = min(max(1, k), L)
    for round_idx in range(n_rounds):
        if selection == "relatedness" and score_vec is not None:
            # Huang et al. Algorithm 1 lines 4-6: score every token by cosine similarity of its
            # representation to the steering vector (their Eq. 4) and mutate the T LOWEST —
            # the residues most opposed to the target property.
            #
            # This is not a refinement, it is the difference between targeted mutagenesis and
            # rewriting the protein. Uniform random masking touches every position eventually
            # (~3x per position over 30 rounds at 10%), so it destroys residues that are
            # already right — measured, that costs the fold: the starting proteins fold at
            # pTM 0.501 and five rounds of random masking already drops them to 0.256, with no
            # steering and no correction involved. Relatedness ranking never touches a position
            # that already agrees with the property.
            with torch.no_grad():
                _, h_clean = steerer.forward(tokens, attn, override=None)
            reps = h_clean[0, 1:L + 1, :].float()
            sims = F.cosine_similarity(reps, score_vec.to(reps.device).float().expand_as(reps),
                                       dim=-1)
            perm = torch.argsort(sims)[:k].cpu()
        elif selection == "confidence":
            # Confidence-frozen remasking (MaskGIT / LLaDA / Dream mask-predict rule, applied
            # here to refinement): freeze the residues the MLM already endorses and remask only
            # the k it finds most surprising. Confidence at a position is the model's probability
            # of the CURRENT residue given its context, p(x_i | x_context), computed on the
            # unmasked sequence — a low value means "given the neighbours, this residue looks
            # wrong", exactly the residues worth resampling; a high value means the residue is
            # well-determined and rewriting it only risks the fold. Same edit budget k as random,
            # so this is a controlled swap of WHICH positions move, not how many. Unlike
            # relatedness this needs no steering vector, so it is defined at alpha=0 too — which
            # is the arm where the random loop's foldability collapse was measured (pTM
            # 0.501 -> 0.256 in five rounds with no steering). Deterministic bottom-k for v1;
            # running-max persistence and stochastic (Gumbel) selection are the noted v2 knobs.
            with torch.no_grad():
                logits_clean, _ = steerer.forward(tokens, attn, override=None)
            lg = logits_clean[0, 1:L + 1, :].float()
            valid = steerer.valid_aa_ids.to(lg.device)
            masked_lg = torch.full_like(lg, float("-inf"))
            masked_lg[:, valid] = lg[:, valid]
            probs = torch.softmax(masked_lg, dim=-1)
            cur = tokens[0, 1:L + 1].to(probs.device)
            conf = probs.gather(1, cur.unsqueeze(1)).squeeze(1)
            perm = torch.argsort(conf)[:k].cpu()
        else:
            perm = torch.randperm(L, generator=generator)[:k]
        pos = (perm + 1).to(device)
        prev = tokens[0, pos].clone()
        tokens[0, pos] = tok.mask_token_id

        _, hidden = steerer.forward(tokens, attn, override=None)
        acts = hidden[0, pos, :]
        clamped = steer_fn(acts.to(device)).to(hidden.dtype).to(device)
        corrected = correct_fn(clamped).to(hidden.dtype).to(device)
        if diag_fn is not None:
            diag_fn(round_idx, clamped, corrected)
        override = hidden.clone()
        override[0, pos, :] = corrected

        logits, _ = steerer.forward(tokens, attn, override=override)
        new_tok = _sample_positions(logits[0, pos].detach().cpu(), steerer.valid_aa_ids,
                                    sample_mode, temperature, generator)
        tokens[0, pos] = new_tok.to(device)

        cur_seq = None
        if select_fn is not None or log_fn is not None:
            cur_seq = decode()
        if log_fn is not None:
            log_fn(round_idx, cur_seq)
        if select_fn is not None:
            score = select_fn(round_idx, cur_seq)
            if score > best_score:
                best_seq, best_score = cur_seq, score

        changed = (tokens[0, pos] != prev).float().mean().item()
        if changed < stabilize_tol:
            break

    return best_seq if select_fn is not None else decode()


# --------------------------------------------------------------------------- #
#  Evaluation primitives
# --------------------------------------------------------------------------- #
def c2h2_concept_score(seq: str, pattern: re.Pattern = C2H2_RE) -> float:
    return float(len(pattern.findall(seq)))


def max_homopolymer_run(seq: str) -> int:
    """Longest run of a single repeated residue — cheap red flag for degenerate/
    repetitive generations, which pseudo-perplexity alone can't distinguish from
    genuine fluency (a trivial repeat is easy for the MLM to predict, so it scores
    as "natural" by pppl while being biologically meaningless)."""
    if not seq:
        return 0
    best = cur = 1
    for i in range(1, len(seq)):
        cur = cur + 1 if seq[i] == seq[i - 1] else 1
        best = max(best, cur)
    return best


# UniProtKB residue-composition statistics (%), the reference distribution for
# `composition_distance`. Kept as a plain constant (not derived from the eval store) so the
# naturalness axis doesn't shift between runs.
AA_BACKGROUND = {"A": 8.25, "R": 5.53, "N": 4.06, "D": 5.45, "C": 1.38, "Q": 3.93,
                 "E": 6.75, "G": 7.07, "H": 2.27, "I": 5.96, "L": 9.66, "K": 5.84,
                 "M": 2.42, "F": 3.86, "P": 4.70, "S": 6.56, "T": 5.34, "W": 1.08,
                 "Y": 2.92, "V": 6.87}


def composition_distance(seq: str) -> float:
    """L1 distance (percentage points, 0-200) between `seq`'s amino-acid composition and
    the UniProtKB background. Higher = less protein-like composition.

    A composition-aware companion to pseudo-perplexity, which on its own points the wrong
    way here: measured on Exp B's alpha=0 GLP-corrected arm, pppl *improves* (15.0 -> 13.9,
    i.e. reads as more natural) while the correction drives Leu to ~18% and Val to ~14%
    — roughly 2x their natural abundance — and mechanically adds +1.35 GRAVY with no
    steering applied. Those sequences pass `is_degenerate_sequence` (no long homopolymer
    run, >8 unique AAs), so that filter doesn't catch this either. Reported per row and
    summarized per (alpha, arm) so the drift is visible instead of being credited to
    steering.

    Read it COMPARATIVELY (naive vs corrected, across alpha), not against zero: a finite
    sequence can't match a 20-way background exactly, so this has a length-dependent
    sampling-noise floor (~40pp at the L~145 of Exp B's held-out inits). On that first run
    the naive arm sat flat at 41-48pp for every alpha while the corrected arm rose
    monotonically — 55pp at alpha=0 (no steering at all), 65pp at 0.5, 69pp at 2, 82pp at
    20 — i.e. the correction, not the steering, is what degrades composition."""
    if not seq:
        return float("nan")
    n = len(seq)
    counts = Counter(seq)
    return float(sum(abs(100.0 * counts.get(a, 0) / n - bg) for a, bg in AA_BACKGROUND.items()))


@torch.no_grad()
def sae_feature_activation(sae, acts_raw: torch.Tensor, feat_idxs, device: str):
    """Mean/max SAE activation of `feat_idxs` over all positions of a (raw,
    un-standardized) activation tensor — a softer, direct check of whether the
    target concept's own SAE feature(s) actually fired in the finished sequence,
    independent of the strict/loose C2H2 regex matching the raw amino acids."""
    idxs = [feat_idxs] if isinstance(feat_idxs, int) else list(feat_idxs)
    x_norm, _ = sae._normalize_input_and_get_norms(acts_raw.to(device))
    feats = sae.encode(x_norm)[..., idxs]
    return float(feats.mean()), float(feats.max())


@torch.no_grad()
def sae_feature_peak_position(sae, acts_raw: torch.Tensor, feat_idxs, device: str) -> int:
    """Residue index where `feat_idxs` activates most strongly (max across features,
    argmax across positions) — a validity check InterPLM's own dashboard uses to
    characterize what a feature detects: if 7033/3684 are genuine C2H2 detectors,
    their peak position should land on Cys/His far more than the ~10% (2/20
    canonical AAs) base rate."""
    idxs = [feat_idxs] if isinstance(feat_idxs, int) else list(feat_idxs)
    x_norm, _ = sae._normalize_input_and_get_norms(acts_raw.to(device))
    feats = sae.encode(x_norm)[..., idxs]
    if feats.dim() == 1:
        feats = feats.unsqueeze(-1)
    return int(feats.max(dim=-1).values.argmax().item())


@torch.no_grad()
def diffmean_projection(v: torch.Tensor, acts_raw: torch.Tensor, device: str):
    """Mean/max cosine similarity of `v` (the diff-of-means steering vector) against
    each position's raw activation (arXiv:2509.07983 Eq. 4's relatedness score) — a
    direct, SAE-independent check of whether the target direction is actually
    present in the finished sequence's own (unclamped, re-embedded) activations."""
    x = acts_raw.to(device).float()
    sim = F.cosine_similarity(x, v.to(device).float().expand_as(x), dim=-1)
    return float(sim.mean()), float(sim.max())


@torch.no_grad()
def diffmean_peak_position(v: torch.Tensor, acts_raw: torch.Tensor, device: str) -> int:
    x = acts_raw.to(device).float()
    sim = F.cosine_similarity(x, v.to(device).float().expand_as(x), dim=-1)
    return int(sim.argmax().item())


def make_steer_fn(method: str, alpha: float, sae=None, feat_idxs=None, v: Optional[torch.Tensor] = None,
                  intervention: str = "clamp", b: float = 0.0):
    if method == "sae":
        if intervention == "scale_shift":
            # Villegas Garcia & Ansuini's own formula (arXiv:2502.09135 Eq. 7) — `alpha`
            # here is their `a` (scale). No special-casing needed: a=1,b=0 (the caller's
            # job to pass in for the anchor condition) cancels algebraically to identity,
            # same as diffmean_clamp's alpha=0.
            return lambda x: sae_scale_shift(sae, x, feat_idxs, alpha, b)
        # clamp mode: alpha=0.0 is the null-baseline condition (Exp A's no-clamp anchor
        # arm): sae_clamp(..., alpha=0) would still overwrite the feature (suppressing
        # it), a real off-manifold intervention, not "unsteered" — so it's special-cased
        # to a no-op here. diffmean_clamp needs no such special-casing (alpha=0 -> x_new
        # == x exactly).
        if alpha == 0.0:
            return lambda a: a
        return lambda a: sae_clamp(sae, a, feat_idxs, alpha)
    return lambda a: diffmean_clamp(v, a, alpha)


def property_columns(init_seq: Optional[str], final_seq: str) -> Dict[str, float]:
    """`<p>_init`, `<p>_final`, `delta_<p>` for EVERY steerable property, not just the one
    being steered (glp_plm/eval/steering_properties.py, design point 2).

    This is what makes the specificity matrix possible after the fact: an off-target
    column that was never written cannot be recovered without regenerating. The steered
    property's columns are duplicated as `prop_*`/`delta_prop` by the caller, so
    `delta_gravy` keeps meaning Δgravy in every run, including charge and instability ones.
    """
    fin = score_all(final_seq)
    ini = score_all(init_seq) if init_seq else {}
    cols: Dict[str, float] = {}
    for k in SCORE_ALL_KEYS:
        cols[f"{k}_init"] = ini.get(k)
        cols[f"{k}_final"] = fin.get(k)
        cols[f"delta_{k}"] = (fin[k] - ini[k]) if (k in fin and k in ini) else None
    return cols


#: Column names `property_columns` produces, in CSV order.
PROPERTY_COLUMNS: List[str] = [
    name for k in SCORE_ALL_KEYS for name in (f"{k}_init", f"{k}_final", f"delta_{k}")]


def target_signal(method: str, acts_raw: torch.Tensor, device: str, sae=None, feat_idxs=None,
                  v: Optional[torch.Tensor] = None) -> Tuple[float, float]:
    if method == "sae":
        return sae_feature_activation(sae, acts_raw, feat_idxs, device)
    return diffmean_projection(v, acts_raw, device)


def target_peak_position(method: str, acts_raw: torch.Tensor, device: str, sae=None, feat_idxs=None,
                         v: Optional[torch.Tensor] = None) -> int:
    if method == "sae":
        return sae_feature_peak_position(sae, acts_raw, feat_idxs, device)
    return diffmean_peak_position(v, acts_raw, device)


@torch.no_grad()
def pseudo_perplexity(steerer: ESMSteerer, seq: str, device: str, batch_size: int = 32) -> float:
    """Standard MLM pseudo-perplexity: mask each residue one at a time, average NLL,
    exponentiate. Lower = more natural."""
    tok = steerer.tokenizer
    base_ids = [tok.cls_token_id] + [tok.convert_tokens_to_ids(a) for a in seq] + [tok.eos_token_id]
    base = torch.tensor(base_ids, dtype=torch.long)
    L = len(seq)
    nlls = []
    for start in range(0, L, batch_size):
        pos = list(range(start, min(start + batch_size, L)))
        batch = base.unsqueeze(0).repeat(len(pos), 1).clone()
        for row, p in enumerate(pos):
            batch[row, p + 1] = tok.mask_token_id
        attn = torch.ones_like(batch)
        batch, attn = batch.to(device), attn.to(device)
        logits = steerer.model(input_ids=batch, attention_mask=attn).logits
        for row, p in enumerate(pos):
            logp = torch.log_softmax(logits[row, p + 1], dim=-1)[base_ids[p + 1]]
            nlls.append(-logp.item())
    return float(np.exp(np.mean(nlls)))


def fold_confidence(seq: str, device: str,
                    _cache: dict = {}) -> Tuple[Optional[float], Optional[float]]:
    """Best-effort ESMFold confidence for `seq`, as (pLDDT on 0-100, pTM on 0-1).

    The field-standard naturalness axis for generated sequences, and the one Exp B was
    missing: pseudo-perplexity asks "is this sequence statistically typical", pLDDT/pTM ask
    "does it specify a structure" — an independent signal, which matters here because the
    correction's Leu/Val enrichment *improves* pppl while pushing composition away from
    natural (see `composition_distance`). Hydrophobic-enriched sequences are also
    aggregation-prone, so folding confidence is a real test of whether the steered
    sequences are plausible proteins at all rather than pppl-flattering junk.

    Both are reported because they fail differently: pLDDT is per-residue local confidence
    (averaged over CA atoms here), so a high mean can hide "confident fragments, incoherent
    global fold" — which pTM, a single global TM-score estimate, is meant to catch.

    Scales verified against the installed transformers (5.14.1), not assumed:
      * pLDDT — `categorical_lddt` -> `EsmCategoricalMixture(..., start=0, end=1).mean()`,
        a softmax-weighted average of bin centres on a 0-1 grid, so the output is in [0,1]
        and IS multiplied by 100 here for conventional reporting. (The previous comment
        claimed it already arrived 0-100, which would have made this a 10x inflation; it
        does not.)
      * pTM — openfold's `compute_tm`, which reduces over residues to a scalar already in
        TM-score units [0,1], so it is recorded as-is.

    Heavy: ESMFold is ~16-24GB GPU mem (PLAN.md D4 flags it Tier-2/feasibility-gated) and
    adds a fold per generated sequence, hence opt-in via --fold."""
    try:
        if "model" not in _cache:
            from transformers import AutoTokenizer, EsmForProteinFolding
            _cache["tok"] = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
            _cache["model"] = EsmForProteinFolding.from_pretrained(
                "facebook/esmfold_v1", low_cpu_mem_usage=True).to(device).eval()
        tok, model = _cache["tok"], _cache["model"]
        inputs = tok([seq], return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model(**inputs)
        # per-atom pLDDT, (1, L, 37); CA = atom index 1
        plddt = float(out["plddt"][0, :, 1].mean()) * 100.0
        ptm_val = getattr(out, "ptm", None)
        return plddt, (float(ptm_val) if ptm_val is not None else None)
    except Exception as e:  # opt-in path; don't take down the run over a folding failure
        print(f"[steering] WARNING: --fold failed for one sequence ({e}); recording None")
        return None, None


# --------------------------------------------------------------------------- #
#  Part 3 — analysis: Pareto plots, three-regime breakdown, rescue sanity check
# --------------------------------------------------------------------------- #
def is_degenerate_sequence(row: dict, min_unique_aa: int = 8, min_homopolymer_run: int = 12,
                           max_single_aa_fraction: float = 0.30) -> bool:
    """Flags collapsed/low-complexity generations (observed in Exp B naive
    diff-of-means steering at high alpha: sequences collapse into long homopolymer
    runs). Pseudo-perplexity scores these as artificially *more* natural (a
    repetitive sequence is trivially self-predictable), which silently dominates
    the naturalness-vs-concept Pareto comparison and the rescue-fraction
    comparator pool in Part 3. Excluded from analysis only — the raw row stays in
    the written CSV for inspection.

    `max_homopolymer_run`/`unique_aa_count` alone miss sequences that interleave a
    dominant residue with occasional others (e.g. long L-runs broken up every few
    residues) without ever hitting a single long run or collapsing unique-AA count
    — caught here by overall single-residue composition instead (>=30% of the
    sequence being one amino acid; random 20-AA composition is ~5%).

    Calibrated against real proteins rather than by eye (scripts/analyze_sequence_complexity.py
    reproduces this). Applied to the 80 natural held-out UniRef50 proteins the run starts from,
    this rejects 3/80 (3.8%) — comparable to the 2.9% it rejects of our own generations, i.e.
    it is not a filter tuned to target our failure mode specifically. Two of those three are
    genuinely low-complexity natural proteins (66% and 68% single-residue). The third was a
    false positive at the old `min_homopolymer_run=8`: an otherwise ordinary protein with a
    run of 9. Natural runs here have median 3 and p95 5.1, so 12 still sits far into the tail
    while no longer rejecting that sequence.

    Do NOT tighten this on local complexity. Complexity is affected by the treatment — in the
    corrected arm the minimum window entropy falls from 2.09 at alpha=0 to 1.73 at alpha=5,
    because steering toward hydrophobicity draws on a smaller sub-alphabet — so a complexity
    cut removes the successfully-steered sequences and biases the effect toward zero. Measured:
    the alpha=2 divergence goes 1.59 (no gate) -> 1.58 (this gate) -> 0.93 at SEG's 2.2-bit
    trigger, which would also reject 75% of the natural starting proteins."""
    seq = row.get("sequence")
    if seq and (max(Counter(seq).values()) / len(seq)) >= max_single_aa_fraction:
        return True
    return (int(float(row["unique_aa_count"])) <= min_unique_aa
            or int(float(row["max_homopolymer_run"])) >= min_homopolymer_run)


def _rescue_fraction(naive: List[dict], corrected: List[dict]) -> Tuple[float, int]:
    """Spec Part 3 sanity check: at matched-or-better naturalness, GLP-corrected
    concept score should beat the best naive concept score achieved at that (or
    worse) naturalness. A corrected point with no naturalness-matched naive
    comparator is excluded rather than counted as a free win (the old `default=-1.0`
    fallback made `concept_score=0 > -1.0` pass trivially)."""
    rescued = comparable = 0
    for c in corrected:
        matched = [n for n in naive if n["naturalness_score"] <= c["naturalness_score"]]
        if not matched:
            continue
        comparable += 1
        if c["concept_score"] > max(n["concept_score"] for n in matched):
            rescued += 1
    frac = rescued / comparable if comparable else float("nan")
    return frac, comparable


def three_regime_breakdown(rows: List[dict], naive_rows: List[dict]) -> Dict[str, dict]:
    """Bin `rows` into safe/transition/broken by diffusion-loss terciles computed
    from the NAIVE population only (so regime boundaries reflect how off-manifold
    naive steering gets, not shifted by GLP's own correction pulling losses back
    down) — spec Part 3: "bin samples by diffusion loss ... check where GLP
    correction helps most"."""
    naive_losses = np.array([r["diffusion_loss"] for r in naive_rows])
    if len(naive_losses) < 3:
        return {}
    lo, hi = np.percentile(naive_losses, [33.33, 66.67])

    def regime(loss: float) -> str:
        return "safe" if loss <= lo else ("transition" if loss <= hi else "broken")

    out = {"cutpoints": [float(lo), float(hi)]}
    for reg in ("safe", "transition", "broken"):
        naive_r = [r for r in naive_rows if regime(r["diffusion_loss"]) == reg]
        corr_r = [r for r in rows if r.get("corrected") and regime(r["diffusion_loss"]) == reg]
        rescue_frac, n_comp = _rescue_fraction(naive_r, corr_r) if naive_r and corr_r else (float("nan"), 0)
        out[reg] = {
            "n_naive": len(naive_r), "n_corrected": len(corr_r),
            "naive_concept_mean": float(np.mean([r["concept_score"] for r in naive_r])) if naive_r else None,
            "corrected_concept_mean": float(np.mean([r["concept_score"] for r in corr_r])) if corr_r else None,
            "rescue_fraction": rescue_frac, "rescue_comparable": n_comp,
        }
    return out


def _wilcoxon_p(vals: List[float]) -> Optional[float]:
    """Signed-rank p for `vals` vs 0, or None when it isn't computable (too few pairs /
    all-zero differences) — kept explicit so summary.json never carries a fake p."""
    from scipy.stats import wilcoxon

    if len(vals) < 5 or all(v == 0 for v in vals):
        return None
    try:
        return float(wilcoxon(vals).pvalue)
    except ValueError:
        return None


def steering_attributable_by_alpha(rows: List[dict], alphas: List[float],
                                   neutral_alpha: float = 0.0) -> Dict[str, dict]:
    """Exp B: concept score with each arm's OWN no-steering run subtracted, per init.

    Motivation (measured on the first Exp B run): at alpha=0 — no steering at all, so the
    two arms differ *only* by whether `glp_correct` ran — the GLP-corrected arm already
    gains +1.35 ΔGRAVY vs the naive arm's +0.40, because SDEdit enriches Leu/Val to ~2x
    natural abundance (see `composition_distance`). That drift is 65-97% of the corrected
    arm's raw ΔGRAVY at every alpha <= 2, so the raw naive-vs-corrected gap
    (`paired_wilcoxon_by_alpha`) credits the correction's compositional bias to steering.

    Differencing within an arm against its own alpha==neutral_alpha run, paired by
    `init_sequence_id`, removes both that drift and per-init variation, leaving the
    steering-attributable effect. On that first run it cut the corrected-vs-naive gap
    roughly in half (+0.93..+1.87 raw -> +0.28..+1.08) but *sharpened* the finding: naive
    steering is flat-to-slightly-negative in the safe range (pooled -0.12) while corrected
    scales monotonically with alpha (+0.23 -> +1.00, pooled +0.44, p=1e-4).

    Returns {} when the grid has no neutral-alpha arm to difference against."""
    base: Dict[Tuple[bool, str], float] = {
        (r["corrected"], r["init_sequence_id"]): r["concept_score"]
        for r in rows if r["alpha"] == neutral_alpha
    }
    if not base:
        print(f"[exp-b] WARNING: no alpha=={neutral_alpha} arm in this grid — skipping the "
              "steering-attributable breakdown (raw numbers include the correction's own "
              "no-steering drift; add 0 to --alphas to separate them).")
        return {}

    out: Dict[str, dict] = {}
    for a in alphas:
        if a == neutral_alpha:
            continue
        arms: Dict[bool, Dict[str, float]] = {}
        for corrected in (False, True):
            arms[corrected] = {
                r["init_sequence_id"]: r["concept_score"] - base[(corrected, r["init_sequence_id"])]
                for r in rows
                if r["alpha"] == a and r["corrected"] == corrected
                and (corrected, r["init_sequence_id"]) in base
            }
        entry: Dict[str, dict] = {}
        for corrected, name in ((False, "naive"), (True, "corrected")):
            v = list(arms[corrected].values())
            if v:
                entry[name] = {"mean": float(np.mean(v)), "n": len(v), "wilcoxon_p": _wilcoxon_p(v)}
        common = sorted(set(arms[False]) & set(arms[True]))
        if len(common) >= 2:
            gap = [arms[True][i] - arms[False][i] for i in common]
            entry["gap"] = {"mean": float(np.mean(gap)), "n": len(gap), "wilcoxon_p": _wilcoxon_p(gap)}
        if entry:
            out[str(a)] = entry
    return out


def divergence_by_alpha(rows: List[dict], alphas: List[float],
                        neutral_alpha: float = 0.0) -> Dict[str, dict]:
    """Bidirectional steering: how far the up and down trajectories separate at each alpha.

    Steering the SAME vector both ways (+alpha*v from low-GRAVY starts, -alpha*v from
    high-GRAVY starts) is a much stronger test than one rising curve, because a real
    steering direction must push the two apart while any direction-independent artefact
    pushes them the same way and cancels in the difference. Two flavours are reported:

      raw          mean(delta up) - mean(delta down). Cancels drift only to the extent the
                   drift is direction-independent — which is exactly what the alpha=0 arms
                   let you check rather than assume.
      attributable each direction first differenced against its OWN alpha=0 run per init
                   (see `steering_attributable_by_alpha`), then subtracted. Robust even when
                   the two directions' drifts differ (e.g. if the correction mean-reverts,
                   pushing GRAVY up from low starts but down from high ones).

    Positive = up finished more hydrophobic than down, i.e. the intended separation. Should
    grow monotonically with alpha, and grow faster for the GLP-corrected arm than the naive
    one if the correction is what makes steering effective."""
    up = [r for r in rows if r.get("direction", "up") == "up"]
    down = [r for r in rows if r.get("direction", "up") == "down"]
    if not up or not down:
        return {}
    attr = {"up": steering_attributable_by_alpha(up, alphas, neutral_alpha),
            "down": steering_attributable_by_alpha(down, alphas, neutral_alpha)}

    out: Dict[str, dict] = {}
    for a in alphas:
        entry: Dict[str, dict] = {}
        for corrected, arm in ((False, "naive"), (True, "corrected")):
            cell = {}
            u_raw = [r["concept_score"] for r in up
                     if r["alpha"] == a and r["corrected"] == corrected]
            d_raw = [r["concept_score"] for r in down
                     if r["alpha"] == a and r["corrected"] == corrected]
            if u_raw and d_raw:
                cell["raw"] = {"up": float(np.mean(u_raw)), "down": float(np.mean(d_raw)),
                               "divergence": float(np.mean(u_raw) - np.mean(d_raw)),
                               "n_up": len(u_raw), "n_down": len(d_raw)}
            u_a = attr["up"].get(str(a), {}).get(arm)
            d_a = attr["down"].get(str(a), {}).get(arm)
            if u_a and d_a:
                cell["attributable"] = {
                    "up": u_a["mean"], "down": d_a["mean"],
                    "divergence": u_a["mean"] - d_a["mean"],
                    "n_up": u_a["n"], "n_down": d_a["n"]}
            if cell:
                entry[arm] = cell
        if entry:
            out[str(a)] = entry
    return out


def composition_by_alpha(rows: List[dict], alphas: List[float]) -> Dict[str, dict]:
    """Mean `composition_distance` per (alpha, arm) — makes the correction's compositional
    drift visible next to pseudo-perplexity, which moves the opposite way (see
    `composition_distance`)."""
    out: Dict[str, dict] = {}
    for a in alphas:
        entry = {}
        for corrected, name in ((False, "naive"), (True, "corrected")):
            v = [composition_distance(r["sequence"]) for r in rows
                 if r["alpha"] == a and r["corrected"] == corrected and r.get("sequence")]
            if v:
                entry[name] = float(np.mean(v))
        if entry:
            out[str(a)] = entry
    return out


def fold_by_alpha(rows: List[dict], alphas: List[float]) -> Dict[str, dict]:
    """Mean ESMFold pLDDT / pTM per (alpha, arm), for rows where --fold produced a value.

    The structural counterpart to `composition_by_alpha`: reported next to
    pseudo-perplexity so the naturalness claim rests on an axis the correction's Leu/Val
    enrichment can't flatter. Skips rows with a missing/failed fold rather than treating
    them as zeros."""
    def num(r, key):
        v = r.get(key)
        if v in (None, "", "None"):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    out: Dict[str, dict] = {}
    for a in alphas:
        entry = {}
        for corrected, name in ((False, "naive"), (True, "corrected")):
            cell = [r for r in rows if r["alpha"] == a and r["corrected"] == corrected]
            for key in ("plddt", "ptm"):
                v = [x for x in (num(r, key) for r in cell) if x is not None]
                if v:
                    entry[f"{name}_{key}"] = float(np.mean(v))
                    entry[f"{name}_{key}_n"] = len(v)
        if entry:
            out[str(a)] = entry
    return out


def paired_wilcoxon_by_alpha(rows: List[dict], alphas: List[float]) -> Dict[str, dict]:
    """Exp B only: paired naive-vs-corrected concept-score (ΔGRAVY) comparison per
    alpha, paired by `init_sequence_id` (each held-out sequence run once per
    (alpha, arm) — spec: "reusing the same starting sequences across every alpha
    makes this a paired design ... recovers most of the statistical power")."""
    from scipy.stats import wilcoxon

    out = {}
    for a in alphas:
        naive = {r["init_sequence_id"]: r["concept_score"] for r in rows
                 if r["alpha"] == a and not r["corrected"]}
        corr = {r["init_sequence_id"]: r["concept_score"] for r in rows
                if r["alpha"] == a and r["corrected"]}
        common = sorted(set(naive) & set(corr))
        if len(common) < 2:
            continue
        d = [corr[i] - naive[i] for i in common]
        try:
            _, p = wilcoxon(d)
        except ValueError:
            p = float("nan")
        out[str(a)] = {"n_pairs": len(common), "mean_delta": float(np.mean(d)), "wilcoxon_p": float(p)}
    return out


def pareto_exp_a(rows: List[dict], out_dir: Path, run: RunLogger, alphas: List[float],
                 neutral_alpha: float = 0.0):
    """Aggregated per-alpha (spec: binary concept score -> no per-sample points),
    anchored at the no-steering arm (alpha==neutral_alpha, corrected=False — 0.0 for
    --sae-intervention=clamp, 1.0 for scale_shift)."""
    steered_alphas = [a for a in alphas if a != neutral_alpha]
    anchor = [r for r in rows if r["alpha"] == neutral_alpha and not r["corrected"]]
    naive = [r for r in rows if not r["corrected"] and r["alpha"] != neutral_alpha]
    corrected = [r for r in rows if r["corrected"] and r["alpha"] != neutral_alpha]

    def agg(subset, alphas_):
        pts = []
        for a in alphas_:
            cell = [r for r in subset if r["alpha"] == a]
            if not cell:
                continue
            pts.append({"alpha": a,
                       "naturalness_score": float(np.mean([r["naturalness_score"] for r in cell])),
                       "concept_score": float(np.mean([r["concept_score"] for r in cell]))})
        return pts

    naive_agg = agg(naive, steered_alphas)
    corr_agg = agg(corrected, steered_alphas)
    anchor_pt = agg(anchor, [neutral_alpha])

    style.apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    if naive_agg:
        ax.plot([p["naturalness_score"] for p in naive_agg], [p["concept_score"] for p in naive_agg],
               "o-", color=style.method_color("sae"), label="naive steering (SAE only)")
    if corr_agg:
        ax.plot([p["naturalness_score"] for p in corr_agg], [p["concept_score"] for p in corr_agg],
               "o-", color=style.method_color("glp"), label="GLP-corrected")
    if anchor_pt:
        ax.scatter([anchor_pt[0]["naturalness_score"]], [anchor_pt[0]["concept_score"]],
                  marker="*", s=220, color=style.BASELINE, edgecolors=style.INK, linewidths=0.6,
                  zorder=5, label="unsteered anchor")
    ax.set_xlabel("naturalness (mean pseudo-perplexity; lower = more natural)")
    ax.set_ylabel("motif hit rate (fraction with ≥1 C2H2 match)")
    ax.set_title("Zinc-finger motif hit rate vs. naturalness")
    ax.legend()
    style.save_fig(fig, run.figures_dir / "pareto.png")

    frac, n_comp = _rescue_fraction(naive_agg, corr_agg)
    _log_rescue(run, "A", frac, n_comp, len(corr_agg))


def pareto_exp_b(rows: List[dict], out_dir: Path, run: RunLogger, suffix: str = ""):
    """Per-sample scatter + mean trajectory overlay (spec explicitly allows this for
    Exp B's continuous ΔGRAVY score, unlike Exp A's aggregate-only)."""
    naive = [r for r in rows if not r["corrected"]]
    corrected = [r for r in rows if r["corrected"]]

    style.apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    if naive:
        ax.scatter([r["naturalness_score"] for r in naive], [r["concept_score"] for r in naive],
                  c=style.steering_arm_color(False), alpha=0.35, s=18, label="naive (per-sample)")
    if corrected:
        ax.scatter([r["naturalness_score"] for r in corrected], [r["concept_score"] for r in corrected],
                  c=style.steering_arm_color(True), alpha=0.35, s=18,
                  label="GLP-corrected (per-sample)")

    def mean_traj(subset):
        alphas = sorted(set(r["alpha"] for r in subset))
        return [(float(np.mean([r["naturalness_score"] for r in subset if r["alpha"] == a])),
                float(np.mean([r["concept_score"] for r in subset if r["alpha"] == a]))) for a in alphas]

    for subset, color, label in ((naive, style.steering_arm_color(False), "naive (mean)"),
                                 (corrected, style.steering_arm_color(True),
                                  "GLP-corrected (mean)")):
        pts = mean_traj(subset)
        if pts:
            xs, ys = zip(*sorted(pts))
            ax.plot(xs, ys, "o-", color=color, linewidth=2, label=label)
    ax.set_xlabel("naturalness (pseudo-perplexity; lower = more natural)")
    ax.set_ylabel("ΔGRAVY (final - init)")
    label = f" ({suffix.lstrip('_')} direction)" if suffix else ""
    ax.set_title(f"Hydrophobicity shift vs. naturalness{label}")
    ax.legend()
    # Deliberately NOT written to disk. Its x-axis is pseudo-perplexity, which on this
    # generation setup rates the degenerate poly-Leu/Val sequences as the most natural output
    # in the run — so the figure ranks its points by a quantity that is inverted where it
    # matters. The rescue fraction below is still logged (spec Part 3), and the figures that
    # carry the claims are built by scripts/plot_steering_report.py against ESMFold2 pTM.
    plt.close(fig)

    frac, n_comp = _rescue_fraction(naive, corrected)
    _log_rescue(run, f"B{suffix}", frac, n_comp, len(corrected))


def _log_rescue(run: RunLogger, experiment: str, frac: float, comparable: int, n_corrected: int):
    if comparable == 0:
        print(f"[exp-{experiment}] WARNING: no GLP-corrected point has a naturalness-matched naive "
              "comparator — rescue fraction is undefined (spec Part 3).")
    elif frac < 0.5:
        print(f"[exp-{experiment}] WARNING: rescue sanity check only holds for {frac:.0%} of "
              f"{comparable} comparable GLP-corrected points — the 'rescue' framing needs "
              "revisiting (spec Part 3).")
    else:
        print(f"[exp-{experiment}] rescue sanity check holds for {frac:.0%} of {comparable} "
              "comparable GLP-corrected points.")
    run.log_summary(**{f"pareto_{experiment}_rescue_fraction": frac,
                       f"pareto_{experiment}_rescue_comparable": comparable,
                       f"pareto_{experiment}_rescue_n": n_corrected})


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/configs_lucia/esm8m_500m_4096_d24.yaml")
    ap.add_argument("--ckpt", help="GLP checkpoint (default <output_dir>/checkpoints/glp_final.pt)")
    ap.add_argument("--scaler", help="rep_statistics.pt (default <output_dir>/checkpoints/rep_statistics.pt)")
    ap.add_argument("--eval-dir", help="…/swissprot/processed. Default: config data.eval_dir, "
                                        "else <output_dir>/swissprot/processed")
    ap.add_argument("--n-shards-per-split", type=int, help="bound eval-store reads (smoke tests)")

    ap.add_argument("--experiment", choices=["A", "B"], required=True,
                    help="'A': SAE-feature clamp on zinc-finger (plm-sae §3.4, random-AA "
                         "init, full resample, 100 rounds, z_k-argmax selection). 'B': "
                         "diff-of-means on hydrophobicity/GRAVY (Huang et al. §4.1.1, "
                         "held-out real init, 10%% mask, 20-50 rounds, last-round "
                         "selection). Deliberately NOT mechanically identical — see "
                         "glprot_steering_spec.md.")

    # Exp A specific
    ap.add_argument("--zinc-finger-concept", default="Zinc finger_C2H2-type")
    ap.add_argument("--zinc-finger-feature",
                    help="[A only] override/skip auto-selection; comma-separated for multiple "
                         "simultaneous features (e.g. '7033,4210')")
    ap.add_argument("--top-n-features", type=int, default=1,
                    help="[A only] auto-select this many top features by domain-adjusted F1 "
                         "when --zinc-finger-feature is not given (arXiv:2502.09135 finds "
                         "top_n=2 much stronger than 1)")
    ap.add_argument("--length-range", default="25,40",
                    help="[A only] min,max sequence length for the random-AA init (spec: ~25-40aa)")
    ap.add_argument("--sae-intervention", choices=["clamp", "scale_shift"], default="clamp",
                    help="[A only] 'clamp': feats[idx] = alpha*rescale_factor, a hard set to "
                         "an absolute target value (this repo's original design). "
                         "'scale_shift': z*_k = alpha*z_k + b (Villegas Garcia & Ansuini's own "
                         "formula, arXiv:2502.09135 Eq. 7) — scales/shifts the feature's OWN "
                         "naturally-arising activation instead of overriding it. Changes what "
                         "--alphas defaults to and means (their 'a', not a multiple of max "
                         "activation) — see --shift-b.")
    ap.add_argument("--shift-b", type=float, default=10.0,
                    help="[A, scale_shift only] the 'b' shift term (arXiv:2502.09135 Eq. 7 "
                         "tests b in [0.1,1,10,50,100,200]; 10 is a representative single "
                         "value — not swept, to keep the sequence budget comparable to the "
                         "clamp-mode run). Applied to every alpha EXCEPT the anchor (alpha=1, "
                         "which always uses b=0 — the algebraic identity, not this value).")

    # Exp B specific
    ap.add_argument("--property", default="gravy", choices=sorted(PROPERTIES),
                    help="[B only] which whole-sequence property to steer. 'gravy' "
                         "(hydrophobicity), 'charge' (net charge per residue at pH 7) and "
                         "'instability' (Guruprasad index) are the three Banerjee et al. "
                         "steer. Selects the contrast file, the reported effect, and the "
                         "default out-dir leaf; every generation is scored on ALL of them "
                         "regardless, so off-target effects are always recoverable.")
    ap.add_argument("--contrast", "--gravy-contrast", dest="contrast",
                    help="[B only] path to <property>_contrast.json (default "
                         "<out-dir>/<property>_contrast.json — built automatically via "
                         "build_property_contrast.py if missing)")
    ap.add_argument("--alpha-mode", choices=["raw", "relative"], default="raw",
                    help="[B only] how --alphas is interpreted. 'raw' (default, and what "
                         "every existing run used): h + alpha*v, so alpha is in units of "
                         "1/||v|| and is NOT comparable between properties whose vectors "
                         "have different norms. 'relative': h + alpha*mean||h||*v/||v||, so "
                         "alpha is the fraction of the activation norm added along the "
                         "direction — the same rotation for every property, which is what a "
                         "cross-property comparison needs.")
    ap.add_argument("--match-length", action="store_true",
                    help="[B only] pass through to the contrast builder when it has to "
                         "build: force the two contrast sets onto a shared length "
                         "distribution. Only affects a build, not an existing contrast file.")
    ap.add_argument("--restart-contrast", action="store_true",
                    help="[B only] rebuild the contrast file even if it exists. Separate "
                         "from --restart (which regenerates this run's SEQUENCES) because "
                         "rebuilding costs a full UniRef50 rescan and, more importantly, "
                         "rebuilds with whatever flags THIS invocation carries — silently "
                         "replacing a contrast that was built separately with e.g. "
                         "--match-length.")

    ap.add_argument("--alphas", default=None,
                    help="steering strengths. Exp A default '0.5,1,2,4,8' (multiples of the "
                         "feature's max activation) — 0.0 (the mandatory no-clamp Pareto "
                         "anchor, a categorical condition per spec, not part of the swept "
                         "grid) is auto-added if not already present, so it always runs. "
                         "Exp B default '0,0.05,0.25,0.5,1,2,5,20' (arXiv:2509.07983's own "
                         "log2 grid; 0 is a valid neutral baseline here, no special-casing "
                         "needed).")
    ap.add_argument("--n-samples-diagnostic", type=int, default=2000)
    ap.add_argument("--n-repeats", type=int, default=4, help="diffusion-loss t/eps repeats")

    ap.add_argument("--n-sequences", type=int, default=None,
                    help="generated sequences per (alpha, corrected). Default: 100 for A "
                         "(spec's floor); ignored for B (always all held-out sequences, "
                         "for the paired design)")
    ap.add_argument("--mask-fraction", type=float, default=None,
                    help="fraction of positions re-masked per round. Default: 1.0 for A "
                         "(full resample, plm-sae §3.4), 0.10 for B (Huang et al. §4.1.1)")
    ap.add_argument("--n-rounds", type=int, default=None,
                    help="Default: 100 for A (spec, fixed), 30 for B (spec's 20-50 pilot "
                         "range midpoint)")
    ap.add_argument("--stabilize-tol", type=float, default=0.0,
                    help="early-stop if the re-sampled fraction that actually changed drops "
                         "below this. Default 0.0 effectively disables early stopping — the "
                         "spec wants fixed round counts for both experiments, not "
                         "convergence-gated ones; raise this to re-enable early stop.")
    ap.add_argument("--generation-pilot", action="store_true",
                    help="[B] sweep round count x sampling temperature at the anchor and a mid "
                         "alpha, and exit. Diagnoses the foldability loss: the alpha=0 NAIVE "
                         "arm (no steering, no correction) already drops pTM from the starting "
                         "proteins' 0.50 to 0.26, so the masked-Gibbs decode loop — not "
                         "steering and not GLP — is what costs foldability. Writes "
                         "generation_pilot.csv in the run's own schema so submit_fold.sh can "
                         "fold it directly.")
    ap.add_argument("--rounds-grid", default="5,10,20,30",
                    help="[--generation-pilot] round counts to sweep")
    ap.add_argument("--mask-fraction-grid", default="",
                    help="[--generation-pilot] mask fractions to sweep (empty = just "
                         "--mask-fraction). Needed to explore the light-editing regime: the "
                         "gentlest setting tested so far still rewrites 37%% of the protein "
                         "(5 rounds x 10%%), and foldability is already collapsed there, so "
                         "whether ANY edit budget preserves the fold is unanswered below that")
    ap.add_argument("--temperature-grid", default="1.0,0.7,0.5",
                    help="[--generation-pilot] sampling temperatures to sweep. Repeatedly "
                         "sampling at T=1.0 from an 8M model is the suspected drift source: "
                         "30 rounds at 10%% masking resample each position ~3x")
    ap.add_argument("--mask-selection", choices=["random", "relatedness", "confidence"],
                    default="random",
                    help="[B] how each round picks positions to mutate. 'random': a uniform "
                         "mask_fraction subset (what this repo has always done). "
                         "'relatedness': Huang et al. Algorithm 1 lines 5-6 — score every "
                         "token by cosine similarity to the steering vector (their Eq. 4) and "
                         "mutate the --n-mutations LOWEST, i.e. the residues most opposed to "
                         "the target property. 'confidence': MaskGIT/LLaDA confidence-frozen "
                         "remasking — remask the k residues with the LOWEST model endorsement "
                         "p(x_i|context) and freeze the rest; needs no steering vector so it is "
                         "defined at alpha=0, the arm where the fold collapse was measured. "
                         "Random masking rewrites every position ~3x over 30 rounds and destroys "
                         "the fold (starting proteins pTM 0.501 -> 0.256 after five rounds, with "
                         "no steering at all); relatedness and confidence both leave "
                         "already-good residues alone.")
    ap.add_argument("--n-mutations", type=int,
                    help="[--mask-selection relatedness] mutation sites per round (Huang et "
                         "al.'s T). Defaults to round(mask_fraction * L) so the edit budget "
                         "matches the random baseline unless set.")
    ap.add_argument("--sample-mode", choices=["sample", "argmax"], default="sample")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--pppl-batch-size", type=int, default=32)

    ap.add_argument("--t-start", type=float, default=None,
                    help="GLP SDEdit start noise level. Default: 0.5 for A (the spec's value, "
                         "chosen for the zinc-finger setting) and 0.7 for B (measured — see "
                         "the note where this default is applied)")
    ap.add_argument("--t-start-pilot", action="store_true",
                    help="run only the t_start in {0.3,0.5,0.7} pilot (spec Part 1) and exit. "
                         "Each t_start runs at both the no-steering anchor and the mid alpha "
                         "so the correction's own drift is separated from the steering gain "
                         "(see the pilot block's comment) — pick on attributable gain, not "
                         "raw concept score.")
    ap.add_argument("--t-start-grid", default="0.3,0.5,0.7",
                    help="[--t-start-pilot] t_start values to sweep (spec Part 1 uses "
                         "0.3,0.5,0.7). Cost is len(grid) x --n-pilot x 2 generations.")
    ap.add_argument("--n-pilot", type=int, default=8,
                    help="[--t-start-pilot] sequences per (t_start, alpha) cell, capped at "
                         "--n-sequences. Default 8 is a smoke-test size — the pilot is cheap "
                         "(~2s/generation for Exp B), so raise it before drawing a conclusion: "
                         "at n=8 the cells' sem on drift is +-0.14..0.30 GRAVY.")
    ap.add_argument("--sde-steps", type=int, default=20)
    ap.add_argument("--log-round-diagnostics", action="store_true",
                    help="log per-round diffusion loss before/after SDEdit (corrected=True "
                         "rows only) to <out-dir>/round_diagnostics.csv, plus a per-round "
                         "trajectory log (round_feature_activation.csv for A, round_property.csv "
                         "for B — the latter is how you check Exp B's round-count pilot: does "
                         "ΔGRAVY plateau by round 20-50?)")
    ap.add_argument("--diag-n-repeats", type=int, default=1,
                    help="t/eps repeats for --log-round-diagnostics' diffusion-loss calls "
                         "(kept low by default — this runs every round, not once per sequence)")

    ap.add_argument("--fold", action=argparse.BooleanOptionalAction, default=None,
                    help="ESMFold pLDDT + pTM. Default: True for A (spec: 'Needs folding: "
                         "Yes'), False for B (spec: 'Needs folding: No'). Heavy (16-24GB GPU "
                         "mem, one fold per generated sequence) — use --no-fold for cheap A "
                         "smoke tests, and --fold on B when you need the structural "
                         "naturalness axis rather than pseudo-perplexity (which the "
                         "correction's Leu/Val enrichment flatters — see "
                         "composition_distance). NOTE: for Exp B this flag was previously a "
                         "silent no-op (the only fold call sat inside the Exp A branch and "
                         "plddt was not a B column); it now works for both.")
    ap.add_argument("--skip-diagnostic", action="store_true")
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--restart", action="store_true", help="ignore any existing partial CSVs")
    ap.add_argument("--exp-name", help="REQUIRED (unless --out-dir is given): a unique tag "
                                       "for this steering run. The out-dir leaf becomes "
                                       "steering_sae_zinc_<exp-name> (A) / steering_diffmean_hydro_"
                                       "<exp-name> (B). RunLogger redirects metrics/figures to "
                                       "$GLP_RUNS_ROOT/<leaf>/ keyed ONLY on that leaf name, so a "
                                       "non-unique tag silently clobbers another run's "
                                       "metrics/figures — pick a fresh one per run.")
    ap.add_argument("--out-dir", help="explicit out-dir path, overrides --exp-name. Its basename "
                                      "is still the RunLogger redirect key, so keep it unique too.")
    ap.add_argument("--directions", default="auto",
                    help="[B only] steering directions to run. 'up' starts from the held-out "
                         "LOW-GRAVY sequences and applies +alpha*v (raise hydrophobicity); "
                         "'down' starts from the held-out HIGH-GRAVY set and applies -alpha*v "
                         "(lower it). Running both is bidirectional steering: the two "
                         "trajectories should diverge monotonically, which is far stronger "
                         "evidence than one rising curve, and their difference cancels any "
                         "direction-independent drift. 'down' needs `heldout_high` in "
                         "the contrast file. Default 'auto' runs both when that key is "
                         "present and falls back to up-only (with a warning) when it is not, "
                         "so a contrast file built before bidirectional steering still runs "
                         "its original single direction rather than erroring; naming a "
                         "direction explicitly is honoured strictly. Doubles generation cost. "
                         "Exp A is up-only.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    is_a = args.experiment == "A"
    method = "sae" if is_a else "diffmean"

    cfg = load_config(args.config)
    set_seed(args.seed if args.seed is not None else cfg.seed)
    device = get_device(cfg.device)
    out = Path(cfg.output_dir)
    ckpt_dir = resolve_storage_dir(out, "checkpoints", "GLP_RUNS_ROOT", symlink=False)
    if args.eval_dir:
        processed = Path(args.eval_dir)
    elif cfg.data.eval_dir:
        processed = resolve_config_data_dir(cfg.data.eval_dir)
    else:
        processed = resolve_storage_dir(out, "swissprot", "GLP_DATA_ROOT", symlink=False) / "processed"

    prop = None if is_a else get_property(args.property)
    # "hydro" rather than "gravy" so the GRAVY leaf is unchanged from before the other two
    # properties existed (runs/steering_crg/steering_diffmean_hydro_* stay findable).
    leaf = "steering_sae_zinc" if is_a else f"steering_diffmean_{prop.leaf}"
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.exp_name:
        out_dir = out / f"{leaf}_{args.exp_name}"
    else:
        # No default: RunLogger redirects metrics/figures to $GLP_RUNS_ROOT/<leaf>/ keyed
        # only on the out-dir basename, so a fixed default leaf (steering_diffmean_hydro)
        # made every Exp B run collide on one shared metrics/figures dir. Force a unique
        # name instead of silently clobbering.
        ap.error(f"steering runs must be uniquely named: pass --exp-name <tag> (leaf becomes "
                 f"{leaf}_<tag>) or --out-dir <path>. Without it, metrics/figures would redirect "
                 f"to a shared $GLP_RUNS_ROOT/{leaf}/ dir and clobber other runs.")
    out_dir.mkdir(parents=True, exist_ok=True)
    run = RunLogger(out_dir)
    run.log_config(cfg)

    scaler = Scaler.load(args.scaler or ckpt_dir / "rep_statistics.pt").to(device)
    model = GLPDenoiser(cfg.model).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt or ckpt_dir / "glp_final.pt", map_location=device))

    sae = feat_idxs = diffmean_vec = heldout = None
    if is_a:
        from interplm.sae.inference import load_sae_from_hf
        plm = "esm2-8m" if "8M" in cfg.data.model_name else "esm2-650m"
        sae = load_sae_from_hf(plm, cfg.data.layer).to(device)

        feat_path = out_dir / "zinc_finger_feature.json"
        if args.zinc_finger_feature is not None:
            feat_idxs = [int(x) for x in args.zinc_finger_feature.split(",") if x.strip()]
            feat_val_f1s = feat_test_f1s = [None] * len(feat_idxs)
        elif feat_path.exists() and not args.restart:
            d = json.loads(feat_path.read_text())
            feat_idxs = d["feature_indices"]
            feat_val_f1s = d.get("val_f1s", [None] * len(feat_idxs))
            feat_test_f1s = d.get("test_f1s", [None] * len(feat_idxs))
        else:
            feat_idxs, feat_val_f1s, feat_test_f1s = select_zinc_finger_features(
                sae, processed, cfg.data.layer, args.zinc_finger_concept,
                top_n=args.top_n_features, n_shards=args.n_shards_per_split)
            feat_path.write_text(json.dumps({
                "feature_indices": feat_idxs, "val_f1s": feat_val_f1s, "test_f1s": feat_test_f1s,
                "concept": args.zinc_finger_concept, "selection_metric": "domain_adjusted_f1",
                "activation_rescale_factors": [float(sae.activation_rescale_factor[i]) for i in feat_idxs],
            }, indent=2))
        f1_str = ", ".join(f"{i}=(val {v:.3f}/test {t:.3f})" if v is not None else f"{i}=?"
                           for i, v, t in zip(feat_idxs, feat_val_f1s, feat_test_f1s))
        print(f"[steering] zinc-finger features = {feat_idxs} (domain-adjusted F1: {f1_str})")
        if args.sae_intervention == "scale_shift":
            default_alphas = "2,5,10,20,30"  # Villegas Garcia & Ansuini's own tested `a` grid
            neutral_alpha = 1.0  # a=1,b=0 is the algebraic identity for this formula
        else:
            default_alphas = "0.5,1,2,4,8"
            neutral_alpha = 0.0  # sae_clamp's null-baseline special-case (see make_steer_fn)
        len_lo, len_hi = (int(x) for x in args.length_range.split(","))
    else:
        from build_property_contrast import build_property_contrast
        contrast_path = (Path(args.contrast) if args.contrast
                         else out_dir / f"{args.property}_contrast.json")
        # NOT args.restart: --restart means "regenerate this run's sequences". Letting it
        # also rebuild the contrast would (a) rescan 12 GB of UniRef50 and (b) silently
        # rebuild with THIS invocation's flags — so a --restart of a run whose contrast was
        # built separately with --match-length would quietly replace the vector with an
        # unmatched one, and the resumed run would no longer be the run it claims to be.
        build_property_contrast(cfg, contrast_path, prop_key=args.property, seed=args.seed,
                                restart=args.restart_contrast, device=device,
                                match_length=args.match_length)
        gd = json.loads(contrast_path.read_text())
        # Pre-generalization files are GRAVY-only and use `gravy_*` keys; read either.
        file_prop = gd.get("property", "gravy")
        if file_prop != args.property:
            ap.error(f"{contrast_path} was built for property {file_prop!r} but "
                     f"--property is {args.property!r}. Point --contrast at the right file "
                     "or pass --restart to rebuild it in place.")
        high_mean = gd.get("prop_high_mean", gd.get("gravy_high_mean"))
        low_mean = gd.get("prop_low_in_mean", gd.get("gravy_low_in_mean"))
        diffmean_vec = torch.tensor(gd["vector"], dtype=torch.float32)
        heldout = gd["heldout"]
        heldout_high = gd.get("heldout_high") or []
        print(f"[steering] {args.property} contrast: high={high_mean:.4f} vs "
              f"low={low_mean:.4f} (||v||={gd['vector_norm']:.4f}), "
              f"{len(heldout)} held-out starting sequences")
        if gd.get("length_gap_pool_sd") is not None and not gd.get("match_length"):
            print(f"[steering] contrast-set length gap {gd['length_gap_pool_sd']:+.2f} pool SD "
                  "(rebuild with --match-length --restart if this is large)")

        # --alpha-mode is implemented by rescaling the vector ONCE, so every call site
        # (pilots included) inherits it and `alpha` keeps its plain meaning in the CSV.
        # Only diffmean_clamp is affected: the projection columns are cosine similarities,
        # which are scale-invariant.
        alpha_scale = 1.0
        if args.alpha_mode == "relative":
            mean_act_norm = gd.get("mean_act_norm")
            if not mean_act_norm or not np.isfinite(mean_act_norm):
                ap.error(f"--alpha-mode relative needs `mean_act_norm` in {contrast_path}, "
                         "which predates it — rebuild the contrast with --restart.")
            alpha_scale = float(mean_act_norm) / float(gd["vector_norm"])
            diffmean_vec = diffmean_vec * alpha_scale
            print(f"[steering] alpha-mode=relative: alpha is now the fraction of mean||h||="
                  f"{mean_act_norm:.4f} added along v/||v|| (raw-mode alpha x {alpha_scale:.3f})")
            if not args.alphas:
                # The raw default grid ('0,...,5,20') was chosen for GRAVY in RAW units; in
                # relative units the same numbers mean an enormous perturbation (alpha=1
                # adds a vector as long as ||h|| itself). And the point of relative mode is a
                # grid SHARED across properties, which cannot be derived from one property's
                # own scale — so it has to be stated explicitly.
                raw_default = [float(a) for a in "0,0.05,0.25,0.5,1,2,5,20".split(",")]
                equiv = ",".join(f"{a / alpha_scale:g}" for a in raw_default)
                ap.error(
                    "--alpha-mode relative requires an explicit --alphas: the raw default "
                    "grid means a different perturbation in these units. For THIS property "
                    f"the raw default grid is equivalent to --alphas {equiv}. Pick one grid "
                    "and pass it to every property, or the runs are not comparable.")
        # The contrast can live outside the run dir (a shared multi-property build passes
        # --contrast), so record where it came from AND copy the two fields the analyses
        # need out of it — otherwise a downstream script that only sees the run dir cannot
        # standardize its numbers or subtract the natural coupling.
        run.log_summary(steered_property=args.property, alpha_mode=args.alpha_mode,
                        mask_selection=args.mask_selection,
                        alpha_scale=alpha_scale, vector_norm=float(gd["vector_norm"]),
                        contrast_path=str(contrast_path.resolve()),
                        contrast_length_gap_pool_sd=gd.get("length_gap_pool_sd"),
                        contrast_prop_gap_pool_sd=gd.get("prop_gap_pool_sd"),
                        contrast_match_length=bool(gd.get("match_length")),
                        contrast_pool_property_stats=gd.get("pool_property_stats"),
                        contrast_natural_correlations=gd.get("natural_correlations"))
        default_alphas = "0,0.05,0.25,0.5,1,2,5,20"
        neutral_alpha = 0.0  # diffmean_clamp's alpha=0 -> x_new == x exactly, no special-casing
        if args.directions.strip() == "auto":
            directions = ["up", "down"] if heldout_high else ["up"]
            if not heldout_high:
                print("[steering] WARNING: this contrast file has no `heldout_high`, so "
                      "only the 'up' direction will run (it predates bidirectional steering). "
                      "Bidirectional is the stronger design — two trajectories that must "
                      "diverge, whose difference cancels direction-independent drift — so "
                      "rebuild the contrast with --restart to enable it.")
        else:
            directions = [d.strip() for d in args.directions.split(",") if d.strip()]
            bad = [d for d in directions if d not in ("up", "down")]
            if bad:
                ap.error(f"--directions must be 'up' and/or 'down' (or 'auto'), got {bad}")
            if "down" in directions and not heldout_high:
                ap.error(f"--directions includes 'down' but {contrast_path} has no `heldout_high` "
                         "(it predates bidirectional steering). Rebuild it with "
                         "`--contrast <path> --restart`, or pass --directions up.")
        heldout_sets = {"up": heldout, "down": heldout_high}
        if args.n_sequences is not None and args.n_sequences != len(heldout):
            print(f"[steering] --n-sequences ignored for experiment B — using all held-out "
                  "sequences per direction (paired design)")
        args.n_sequences = len(heldout)
        print("[steering] directions=" + ",".join(
            f"{d}(n={len(heldout_sets[d])})" for d in directions))

    if args.n_sequences is None:
        args.n_sequences = 100
    if args.mask_fraction is None:
        args.mask_fraction = 1.0 if is_a else 0.10
    if args.n_rounds is None:
        args.n_rounds = 100 if is_a else 30
    if args.fold is None:
        args.fold = is_a
    if args.t_start is None:
        # Exp B: 0.7, not the 0.5 inherited from the zinc-finger experiment. A 7-point
        # pilot at n=15 per cell (--t-start-pilot --t-start-grid 0.2..0.8) found 0.5-0.6 to
        # be a pathological band: highest correction drift (1.55/1.69 ΔGRAVY with NO
        # steering), composition cost of +11pp that is clearly nonzero, and — the tell —
        # the *best* pseudo-perplexity, because their Leu/Val enrichment is easy for the MLM
        # to predict. 0.7 gives the strongest AND most reliable steering-attributable gain in
        # the grid (+0.43, p=0.0001) with a composition cost indistinguishable from zero.
        # Confirmed on the full sweep: switching 0.5 -> 0.7 dropped degenerate generations
        # from 62/240 (26%) to 23/300 (7.7%), took alpha=5's usable corrected rows from 2 to
        # 14, and made the dose-response monotone through alpha=5 — i.e. the "breakdown at
        # alpha=5" was an artefact of t_start, not a property of steering strength.
        # Exp A keeps 0.5: none of the above was measured for the SAE-clamp/zinc-finger path.
        args.t_start = 0.5 if is_a else 0.7

    alphas = [float(a) for a in (args.alphas or default_alphas).split(",") if a.strip()]
    if is_a and neutral_alpha not in alphas:
        # The no-steering anchor (spec's mandatory Pareto anchor, "do NOT use alpha=0
        # [as steering strength], instead run a separate arm with no clamping at all")
        # rides whichever alpha value make_steer_fn treats as an algebraic no-op for
        # the active --sae-intervention (0.0 for clamp, 1.0+b=0 for scale_shift) — but
        # the default grid doesn't include it, so it has to be added explicitly or the
        # anchor arm silently never runs.
        alphas = [neutral_alpha] + alphas

    if is_a:
        directions = ["up"]   # bidirectional is a diff-of-means (Exp B) construct

    steerer = ESMSteerer(cfg.data.model_name, cfg.data.layer, device)

    def b_for(alpha: float) -> float:
        """[A, scale_shift only] the anchor (alpha==neutral_alpha) always uses b=0 (the
        algebraic identity); every other alpha uses the fixed --shift-b."""
        return 0.0 if alpha == neutral_alpha else args.shift_b

    def build_init(i: int, generator: torch.Generator, direction: str = "up"):
        """`direction` selects which held-out set the start comes from: 'up' -> the
        low-score set (steered +v), 'down' -> the high-score set (steered -v). Exp A ignores
        it (random-AA inits).

        Returns (tokens, length, held-out id, starting property score, starting sequence).
        `score` is the generic contrast key, `gravy` the pre-generalization one that
        archived contrast files still use. The starting sequence comes back so every
        property can be differenced init->final, not just the steered one."""
        if is_a:
            length = int(np.random.default_rng(args.seed * 7919 + i).integers(len_lo, len_hi + 1))
            return (random_aa_init_tokens(steerer.tokenizer, length, generator), length,
                    None, None, None)
        ho = heldout_sets[direction][i]
        init_score = ho["score"] if "score" in ho else ho["gravy"]
        return (seq_to_init_tokens(steerer.tokenizer, ho["sequence"]), len(ho["sequence"]),
                ho["id"], init_score, ho["sequence"])

    if args.t_start_pilot:
        extractor = ActivationExtractor(cfg.data.model_name, cfg.data.layer, device=device)
        mid_alpha = alphas[len(alphas) // 2]
        n_pilot = min(args.n_pilot, args.n_sequences)
        t_start_grid = [float(t) for t in args.t_start_grid.split(",") if t.strip()]
        # Each t_start is run at BOTH the no-steering anchor and mid_alpha, on the same
        # seeds (so the two share a masking schedule and differ only by the steer_fn).
        # Scoring a t_start on its steered concept score alone would reward exactly the
        # pathology this pilot needs to avoid: a high t_start injects more noise, so the
        # correction rewrites more of the sequence freely, and on Exp B that free rewriting
        # is itself strongly concept-directed (+1.35 ΔGRAVY at alpha=0 via Leu/Val
        # enrichment, at t_start=0.5). The anchor arm makes that drift measurable, so a
        # t_start can be picked on the steering-ATTRIBUTABLE gain and its composition cost.
        pilot_alphas = [neutral_alpha, mid_alpha] if mid_alpha != neutral_alpha else [neutral_alpha]
        pilot_rows = []
        for t_start in t_start_grid:
            for i in range(n_pilot):
                for alpha in pilot_alphas:
                    seed_i = args.seed * 1000 + i
                    g = torch.Generator().manual_seed(seed_i)
                    g_dev = torch.Generator(device=device).manual_seed(seed_i)
                    init_tokens, length, init_id, init_score, _ = build_init(i, g)
                    seq = masked_gibbs_generate(
                        steerer, init_tokens,
                        steer_fn=make_steer_fn(method, alpha, sae=sae, feat_idxs=feat_idxs, v=diffmean_vec,
                                               intervention=args.sae_intervention, b=b_for(alpha)),
                        correct_fn=lambda a: glp_correct(model, scaler, a, t_start=t_start,
                                                         num_steps=args.sde_steps, device=device, generator=g_dev),
                        mask_fraction=args.mask_fraction, n_rounds=args.n_rounds,
                        stabilize_tol=args.stabilize_tol, sample_mode=args.sample_mode,
                        temperature=args.temperature, generator=g, device=device,
                        selection=args.mask_selection, n_mutations=args.n_mutations,
                        score_vec=None if is_a else diffmean_vec)
                    acts = torch.from_numpy(extractor.extract_single(seq)).float()
                    dloss = float(diffusion_loss(model, scaler, acts, n_repeats=args.n_repeats,
                                                 device=device, generator=g_dev).mean())
                    concept = c2h2_concept_score(seq) if is_a else prop.score(seq) - init_score
                    pilot_rows.append({"t_start": t_start, "alpha": alpha, "sample_idx": i,
                                      "length": length, "concept_score": concept,
                                      "pseudo_perplexity": pseudo_perplexity(steerer, seq, device),
                                      "diffusion_loss": dloss,
                                      "composition_distance": composition_distance(seq)})
                    print(f"[t_start-pilot] t_start={t_start} alpha={alpha} sample={i} "
                          f"concept={concept:+.4f} pppl={pilot_rows[-1]['pseudo_perplexity']:.2f} "
                          f"dloss={dloss:.4f} comp={pilot_rows[-1]['composition_distance']:.1f}pp",
                          flush=True)

        # per-t_start read-out: drift vs steering-attributable gain vs composition cost
        print(f"\n[t_start-pilot] anchor alpha={neutral_alpha}, steered alpha={mid_alpha}, "
              f"n={n_pilot} per cell")
        for t_start in t_start_grid:
            base = {r["sample_idx"]: r for r in pilot_rows
                    if r["t_start"] == t_start and r["alpha"] == neutral_alpha}
            steered = {r["sample_idx"]: r for r in pilot_rows
                       if r["t_start"] == t_start and r["alpha"] == mid_alpha}
            if not base:
                continue
            drift = float(np.mean([r["concept_score"] for r in base.values()]))
            comp_b = float(np.mean([r["composition_distance"] for r in base.values()]))
            common = sorted(set(base) & set(steered))
            attr = (float(np.mean([steered[i]["concept_score"] - base[i]["concept_score"]
                                   for i in common])) if common else float("nan"))
            comp_s = (float(np.mean([steered[i]["composition_distance"] for i in common]))
                      if common else float("nan"))
            pppl_s = (float(np.mean([steered[i]["pseudo_perplexity"] for i in common]))
                      if common else float("nan"))
            print(f"[t_start-pilot] t_start={t_start}: drift(no steering)={drift:+.4f}  "
                  f"attributable={attr:+.4f}  composition={comp_b:.1f}->{comp_s:.1f}pp  "
                  f"pppl={pppl_s:.2f}")
        print("[t_start-pilot] pick the t_start with the best attributable gain per unit of "
              "drift/composition cost — NOT the largest raw concept score.")
        pilot_csv = out_dir / "t_start_pilot.csv"
        with open(pilot_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pilot_rows[0].keys()))
            w.writeheader(); w.writerows(pilot_rows)
        print(f"[t_start-pilot] wrote {pilot_csv} — pick a t_start from this, then re-run "
              f"without --t-start-pilot (--t-start <chosen>).")
        return

    if args.generation_pilot:
        if is_a:
            ap.error("--generation-pilot is an Exp B diagnostic (it needs the real held-out "
                     "starting proteins to measure what the decode loop costs)")
        extractor = ActivationExtractor(cfg.data.model_name, cfg.data.layer, device=device)
        mid_alpha = alphas[len(alphas) // 2]
        pilot_alphas = sorted({neutral_alpha, mid_alpha})
        rounds_grid = [int(x) for x in args.rounds_grid.split(",") if x.strip()]
        temp_grid = [float(x) for x in args.temperature_grid.split(",") if x.strip()]
        mf_grid = ([float(x) for x in args.mask_fraction_grid.split(",") if x.strip()]
                   or [args.mask_fraction])
        n_pilot = min(args.n_pilot, len(heldout_sets["up"]))
        print(f"[gen-pilot] rounds={rounds_grid} temps={temp_grid} alphas={pilot_alphas} "
              f"arms=naive,corrected n={n_pilot} (direction=up)", flush=True)
        rows_p, sid = [], 0
        for n_rounds in rounds_grid:
          for mask_frac in mf_grid:
            for temp in temp_grid:
                for alpha in pilot_alphas:
                    for corrected in (False, True):
                        for i in range(n_pilot):
                            seed_i = args.seed * 1000 + i
                            g = torch.Generator().manual_seed(seed_i)
                            g_dev = torch.Generator(device=device).manual_seed(seed_i)
                            init_tokens, length, init_id, init_score, init_seq = build_init(
                                i, g, "up")
                            seq = masked_gibbs_generate(
                                steerer, init_tokens,
                                steer_fn=make_steer_fn(method, alpha, v=diffmean_vec),
                                correct_fn=(lambda a: glp_correct(
                                    model, scaler, a, t_start=args.t_start,
                                    num_steps=args.sde_steps, device=device, generator=g_dev))
                                if corrected else (lambda a: a),
                                mask_fraction=mask_frac, n_rounds=n_rounds,
                                stabilize_tol=args.stabilize_tol, sample_mode=args.sample_mode,
                                temperature=temp, generator=g, device=device,
                                selection=args.mask_selection, n_mutations=args.n_mutations,
                                score_vec=diffmean_vec)
                            rows_p.append({
                                "sample_id": sid, "init_sequence_id": init_id,
                                "n_rounds": n_rounds, "temperature": temp, "alpha": alpha,
                                "mask_fraction": mask_frac, "selection": args.mask_selection,
                                "corrected": corrected, "direction": "up", "length": length,
                                "property": args.property,
                                "prop_init": init_score, "prop_final": prop.score(seq),
                                "delta_prop": prop.score(seq) - init_score,
                                **property_columns(init_seq, seq),
                                "naturalness_score": pseudo_perplexity(
                                    steerer, seq, device, batch_size=args.pppl_batch_size),
                                "max_homopolymer_run": max_homopolymer_run(seq),
                                "unique_aa_count": len(set(seq)),
                                "composition_distance": composition_distance(seq),
                                "sequence": seq})
                            sid += 1
                        print(f"[gen-pilot] rounds={n_rounds} mask={mask_frac} T={temp} "
                              f"alpha={alpha} corrected={corrected} done ({sid} total)",
                              flush=True)
        out_csv = out_dir / "generation_pilot.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_p[0].keys()))
            w.writeheader(); w.writerows(rows_p)
        print(f"[gen-pilot] wrote {out_csv} ({len(rows_p)} sequences). Fold it with:\n"
              f"  sbatch scripts/cluster/submit_fold.sh <run-dir> --input {out_csv} "
              f"--out {out_dir / 'generation_pilot_fold.csv'}")
        return

    if not args.skip_diagnostic:
        acts_test, _, _ = sl.load_split(processed, "test", cfg.data.layer, n_shards=args.n_shards_per_split)
        g = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(acts_test.shape[0], generator=g)[:args.n_samples_diagnostic]
        baseline_acts = acts_test[idx].float()
        # neutral_alpha (null-baseline condition, see make_steer_fn) is redundant here —
        # the diagnostic already has its own unclamped "baseline" row.
        diag_alphas = [a for a in alphas if a != neutral_alpha]
        if is_a and args.sae_intervention == "scale_shift":
            diag_clamp_fn = lambda a, al: sae_scale_shift(sae, a, feat_idxs, al, args.shift_b)
        elif is_a:
            diag_clamp_fn = lambda a, al: sae_clamp(sae, a, feat_idxs, al)
        else:
            diag_clamp_fn = lambda a, al: diffmean_clamp(diffmean_vec, a, al)
        # `correct_fn` here is deliberately generation-free: one SDEdit pass straight onto the
        # clamped activations, so Part 0 also answers "does the correction operator itself
        # recover a heavily-clamped activation, or does the collapse need the Gibbs loop?"
        g_diag = torch.Generator(device=device).manual_seed(args.seed)
        run_off_manifold_diagnostic(model, scaler, diag_clamp_fn, diag_alphas, baseline_acts,
                                    out_dir, run, device, args.n_repeats, args.seed, args.experiment,
                                    correct_fn=lambda a: glp_correct(
                                        model, scaler, a, t_start=args.t_start,
                                        num_steps=args.sde_steps, device=device, generator=g_diag))

    if not args.skip_generate:
        extractor = ActivationExtractor(cfg.data.model_name, cfg.data.layer, device=device)

        # "gravy" rather than the property key for the hydrophobicity run, so the archived
        # expB_diffmean_gravy.csv name is unchanged; charge/instability get their own.
        csv_name = "expA_sae_zincfinger.csv" if is_a else f"expB_diffmean_{args.property}.csv"
        csv_path = out_dir / csv_name
        existing = []
        if csv_path.exists() and not args.restart:
            with open(csv_path) as f:
                existing = list(csv.DictReader(f))
        done_counts = {}
        for r in existing:
            # legacy CSVs (pre-bidirectional) have no direction column -> they are all "up"
            key = (r.get("direction") or "up", r["alpha"], r["corrected"])
            done_counts[key] = done_counts.get(key, 0) + 1

        if is_a:
            fieldnames = ["sample_id", "alpha", "corrected", "length", "motif_hit",
                         "concept_count", "concept_count_loose", "naturalness_score",
                         "diffusion_loss", "max_homopolymer_run", "unique_aa_count",
                         "composition_distance",
                         "target_feature_activation_mean", "target_feature_activation_max",
                         "target_feature_peak_residue", "plddt", "ptm", "sequence"]
        else:
            fieldnames = (["sample_id", "init_sequence_id", "alpha", "corrected", "length",
                          "property", "prop_init", "prop_final", "delta_prop"]
                          + PROPERTY_COLUMNS +
                          ["naturalness_score",
                           "diffusion_loss", "max_homopolymer_run", "unique_aa_count",
                           "composition_distance",
                           "target_projection_mean", "target_projection_max",
                           "plddt", "ptm", "direction", "sequence"])

        mode = "a" if existing else "w"
        if existing and set(existing[0].keys()) != set(fieldnames):
            # Appending rows in a different schema to an existing CSV produces a file whose
            # header does not describe its own later rows — silent corruption that only
            # shows up as nonsense in the analysis. This fires for runs started before the
            # multi-property columns existed.
            added = sorted(set(fieldnames) - set(existing[0]))
            dropped = sorted(set(existing[0]) - set(fieldnames))
            ap.error(
                f"{csv_path} was written with a different column set (added: {added or '-'}; "
                f"dropped: {dropped or '-'}), so resuming would append rows the header does "
                "not describe. Pass --restart to regenerate it, or point --out-dir/--exp-name "
                "at a fresh run.")
        rows_for_analysis = []
        for r in existing:
            # legacy CSVs (pre-generalization) carry only delta_gravy, which was the
            # steered property's delta by construction there.
            concept_score = ((1.0 if r["motif_hit"] == "True" else 0.0) if is_a
                             else float(r.get("delta_prop") or r["delta_gravy"]))
            rows_for_analysis.append({
                **r, "alpha": float(r["alpha"]), "corrected": r["corrected"] == "True",
                "direction": r.get("direction") or "up",
                "naturalness_score": float(r["naturalness_score"]),
                "diffusion_loss": float(r["diffusion_loss"]), "concept_score": concept_score,
            })
        sample_id = len(existing)

        diag_path = out_dir / "round_diagnostics.csv"
        diag_fieldnames = ["sample_id", "alpha", "corrected", "round",
                           "diffusion_loss_pre", "diffusion_loss_post"]
        diag_file = diag_writer = None
        if args.log_round_diagnostics:
            diag_mode = "a" if diag_path.exists() and not args.restart else "w"
            diag_file = open(diag_path, diag_mode, newline="")
            diag_writer = csv.DictWriter(diag_file, fieldnames=diag_fieldnames)
            if diag_mode == "w":
                diag_writer.writeheader()

        roundlog_name = "round_feature_activation.csv" if is_a else "round_property.csv"
        roundlog_path = out_dir / roundlog_name
        roundlog_fieldnames = (["sample_id", "alpha", "corrected", "round",
                                "feature_activation_mean", "feature_activation_max"] if is_a
                               else ["sample_id", "alpha", "corrected", "round",
                                     "property", "prop_current"])
        roundlog_file = roundlog_writer = None
        if args.log_round_diagnostics:
            rl_mode = "a" if roundlog_path.exists() and not args.restart else "w"
            roundlog_file = open(roundlog_path, rl_mode, newline="")
            roundlog_writer = csv.DictWriter(roundlog_file, fieldnames=roundlog_fieldnames)
            if rl_mode == "w":
                roundlog_writer.writeheader()

        with open(csv_path, mode, newline="") as fcsv:
            w = csv.DictWriter(fcsv, fieldnames=fieldnames)
            if mode == "w":
                w.writeheader()
            for direction in directions:
                # -v for the down direction; `alpha` stays the magnitude in the CSV so both
                # directions share an x-axis and `direction` carries the sign.
                sign = 1.0 if direction == "up" else -1.0
                n_seq_dir = args.n_sequences if is_a else len(heldout_sets[direction])
                for alpha in alphas:
                    for corrected in (False, True):
                        key = (direction, str(alpha), str(corrected))
                        n_done = done_counts.get(key, 0)
                        for i in range(n_done, n_seq_dir):
                            seed_i = args.seed * 1_000_003 + sample_id
                            g = torch.Generator().manual_seed(seed_i)
                            g_dev = torch.Generator(device=device).manual_seed(seed_i)

                            (init_tokens, length, init_sequence_id, init_score,
                             init_seq) = build_init(i, g, direction)
                            # spec's selection caveat: Exp A's no-steering anchor (alpha==
                            # neutral_alpha) keeps the LAST round (select_fn=None), not the
                            # z_k-argmax criterion, to avoid cherry-picking an unsteered run's
                            # best-looking round. Exp B never selects (always last round).
                            use_select = is_a and alpha != neutral_alpha

                            # sign flips the diff-of-means vector for the down direction;
                            # alpha=0 is identity either way, which is what makes the
                            # alpha=0 down arm a clean read on the correction's own drift
                            # from a HIGH-GRAVY start (a fixed hydrophobic bias would still
                            # push up there; mean-reversion would push down).
                            steer_fn = make_steer_fn(method, sign * alpha, sae=sae,
                                                     feat_idxs=feat_idxs, v=diffmean_vec,
                                                     intervention=args.sae_intervention,
                                                     b=b_for(alpha) if is_a else 0.0)

                            def correct_fn(a, _corrected=corrected, _g=g_dev):
                                if not _corrected:
                                    return a
                                return glp_correct(model, scaler, a, t_start=args.t_start,
                                                  num_steps=args.sde_steps, device=device, generator=_g)

                            diag_fn = None
                            if diag_writer is not None and corrected:
                                def diag_fn(round_idx, clamped, corrected_acts,
                                           _sample_id=sample_id, _alpha=alpha, _g=g_dev):
                                    loss_pre = float(diffusion_loss(
                                        model, scaler, clamped.detach(), n_repeats=args.diag_n_repeats,
                                        device=device, generator=_g).mean())
                                    loss_post = float(diffusion_loss(
                                        model, scaler, corrected_acts.detach(), n_repeats=args.diag_n_repeats,
                                        device=device, generator=_g).mean())
                                    diag_writer.writerow({
                                        "sample_id": _sample_id, "alpha": _alpha, "corrected": True,
                                        "round": round_idx, "diffusion_loss_pre": loss_pre,
                                        "diffusion_loss_post": loss_post})
                                    diag_file.flush()

                            select_fn = None
                            if use_select:
                                def select_fn(round_idx, cur_seq, _sample_id=sample_id, _alpha=alpha,
                                             _corrected=corrected):
                                    acts = torch.from_numpy(extractor.extract_single(cur_seq)).float()
                                    mean_act, max_act = target_signal(method, acts, device, sae=sae,
                                                                      feat_idxs=feat_idxs, v=diffmean_vec)
                                    if roundlog_writer is not None:
                                        roundlog_writer.writerow({
                                            "sample_id": _sample_id, "alpha": _alpha, "corrected": _corrected,
                                            "round": round_idx, "feature_activation_mean": mean_act,
                                            "feature_activation_max": max_act})
                                        roundlog_file.flush()
                                    # max, not mean: the target feature is sparse (active at a
                                    # handful of positions at most), so a length-diluted mean
                                    # can't tell a round with a strong local hit from none at all.
                                    return max_act

                            log_fn = None
                            if not is_a and roundlog_writer is not None:
                                def log_fn(round_idx, cur_seq, _sample_id=sample_id, _alpha=alpha,
                                          _corrected=corrected):
                                    roundlog_writer.writerow({
                                        "sample_id": _sample_id, "alpha": _alpha, "corrected": _corrected,
                                        "round": round_idx, "property": args.property,
                                        "prop_current": prop.score(cur_seq)})
                                    roundlog_file.flush()

                            seq = masked_gibbs_generate(
                                steerer, init_tokens, steer_fn, correct_fn, args.mask_fraction,
                                args.n_rounds, args.stabilize_tol, args.sample_mode,
                                args.temperature, g, device, diag_fn=diag_fn, select_fn=select_fn,
                                log_fn=log_fn, selection=args.mask_selection,
                                n_mutations=args.n_mutations,
                                # sign-adjust so relatedness masks positions opposed to THIS
                                # direction's target: for 'down' (sign=-1) the residues most
                                # opposed to lowering the property are the ones most aligned
                                # with +v (highest property), so the vector must flip too —
                                # cosine is scale-invariant, so the alpha_scale on diffmean_vec
                                # is irrelevant here and only the sign matters. Inert for
                                # random/confidence (they ignore score_vec).
                                score_vec=None if is_a else sign * diffmean_vec)

                            pppl = pseudo_perplexity(steerer, seq, device, batch_size=args.pppl_batch_size)
                            acts_final = torch.from_numpy(extractor.extract_single(seq)).float()
                            dloss = float(diffusion_loss(model, scaler, acts_final, n_repeats=args.n_repeats,
                                                         device=device, generator=g_dev).mean())

                            if is_a:
                                concept_count = c2h2_concept_score(seq)
                                concept_count_loose = c2h2_concept_score(seq, C2H2_RE_LOOSE)
                                motif_hit = concept_count > 0
                                feat_act_mean, feat_act_max = target_signal(method, acts_final, device,
                                                                            sae=sae, feat_idxs=feat_idxs)
                                peak_pos = target_peak_position(method, acts_final, device, sae=sae,
                                                                feat_idxs=feat_idxs)
                                peak_residue = seq[peak_pos] if 0 <= peak_pos < len(seq) else None
                                plddt, ptm = fold_confidence(seq, device) if args.fold else (None, None)
                                row = {"sample_id": sample_id, "alpha": alpha, "corrected": corrected,
                                      "length": length, "motif_hit": motif_hit,
                                      "concept_count": concept_count, "concept_count_loose": concept_count_loose,
                                      "naturalness_score": pppl, "diffusion_loss": dloss,
                                      "max_homopolymer_run": max_homopolymer_run(seq),
                                      "unique_aa_count": len(set(seq)),
                                      "composition_distance": composition_distance(seq),
                                      "target_feature_activation_mean": feat_act_mean,
                                      "target_feature_activation_max": feat_act_max,
                                      "target_feature_peak_residue": peak_residue,
                                      "plddt": plddt, "ptm": ptm, "sequence": seq}
                                concept_score = float(motif_hit)
                                log_line = (f"[exp-a] alpha={alpha} corrected={corrected} sample={i} "
                                           f"motif_hit={motif_hit} count={concept_count}/{concept_count_loose} "
                                           f"pppl={pppl:.2f} dloss={dloss:.4f} plddt={plddt} ptm={ptm}")
                            else:
                                prop_final = prop.score(seq)
                                delta_prop = prop_final - init_score
                                plddt, ptm = fold_confidence(seq, device) if args.fold else (None, None)
                                proj_mean, proj_max = target_signal(method, acts_final, device, v=diffmean_vec)
                                row = {"sample_id": sample_id, "init_sequence_id": init_sequence_id,
                                      "alpha": alpha, "corrected": corrected, "length": length,
                                      "property": args.property, "prop_init": init_score,
                                      "prop_final": prop_final, "delta_prop": delta_prop,
                                      **property_columns(init_seq, seq),
                                      "naturalness_score": pppl,
                                      "diffusion_loss": dloss, "max_homopolymer_run": max_homopolymer_run(seq),
                                      "unique_aa_count": len(set(seq)),
                                      "composition_distance": composition_distance(seq),
                                      "target_projection_mean": proj_mean, "target_projection_max": proj_max,
                                      "plddt": plddt, "ptm": ptm, "direction": direction,
                                      "sequence": seq}
                                concept_score = delta_prop
                                fold_str = (f" plddt={plddt:.1f} ptm={ptm:.3f}"
                                            if plddt is not None and ptm is not None else "")
                                log_line = (f"[exp-b] dir={direction} alpha={alpha} corrected={corrected} "
                                           f"init_id={init_sequence_id} {args.property} "
                                           f"{init_score:{prop.fmt}}->{prop_final:{prop.fmt}} "
                                           f"(d={delta_prop:{prop.fmt}}) pppl={pppl:.2f} "
                                           f"dloss={dloss:.4f}{fold_str}")

                            w.writerow(row)
                            fcsv.flush()
                            rows_for_analysis.append({**row, "concept_score": concept_score,
                                                      "naturalness_score": pppl,
                                                      "diffusion_loss": dloss,
                                                      "direction": direction})
                            sample_id += 1
                            print(log_line, flush=True)

        if diag_file is not None:
            diag_file.close()
            print(f"[steering] wrote {diag_path}")
        if roundlog_file is not None:
            roundlog_file.close()
            print(f"[steering] wrote {roundlog_path}")

        n_before_filter = len(rows_for_analysis)
        degenerate_rows = [r for r in rows_for_analysis if is_degenerate_sequence(r)]
        rows_for_analysis = [r for r in rows_for_analysis if not is_degenerate_sequence(r)]
        if degenerate_rows:
            print(f"[steering] excluded {len(degenerate_rows)}/{n_before_filter} degenerate sequences "
                  "(see is_degenerate_sequence) from Part 3 analysis — still present in "
                  f"{csv_path.name} for inspection.")
            run.log_summary(n_degenerate_excluded=len(degenerate_rows))

        if is_a:
            naive_steered = [r for r in rows_for_analysis if not r["corrected"] and r["alpha"] != neutral_alpha]
            pareto_exp_a(rows_for_analysis, out_dir, run, alphas, neutral_alpha)
            regimes = three_regime_breakdown(rows_for_analysis, naive_steered)
            anchor_rows = [r for r in rows_for_analysis if r["alpha"] == neutral_alpha and not r["corrected"]]
            if anchor_rows:
                anchor_hits = sum(1 for r in anchor_rows if r["concept_score"] > 0)
                print(f"[exp-a] unsteered anchor (no steering): {anchor_hits}/{len(anchor_rows)} strict "
                      "concept hits — the C2H2 regex's own false-positive rate on this model's "
                      "unsteered generation distribution.")
        else:
            # Pooling the directions would average a rising trajectory against a falling one
            # and cancel the effect, so every per-alpha statistic is computed WITHIN a
            # direction; `divergence_by_alpha` is what combines them.
            dirs_present = sorted({r.get("direction", "up") for r in rows_for_analysis},
                                  key=lambda d: 0 if d == "up" else 1)
            attrib_by_dir, comp_by_dir, w_by_dir, regimes_by_dir = {}, {}, {}, {}
            for d in dirs_present:
                sub = [r for r in rows_for_analysis if r.get("direction", "up") == d]
                tag = f"[exp-b/{d}]"
                pareto_exp_b(sub, out_dir, run, suffix="" if d == "up" else f"_{d}")
                regimes_by_dir[d] = three_regime_breakdown(
                    sub, [r for r in sub if not r["corrected"]])
                w_by_dir[d] = paired_wilcoxon_by_alpha(sub, alphas)
                attrib_by_dir[d] = steering_attributable_by_alpha(sub, alphas, neutral_alpha)
                comp_by_dir[d] = composition_by_alpha(sub, alphas)

                for a, st in w_by_dir[d].items():
                    print(f"{tag} paired Wilcoxon alpha={a}: n={st['n_pairs']} "
                          f"mean_delta={st['mean_delta']:+.4f} p={st['wilcoxon_p']:.4f}")
                # The alpha=0 arms are the drift read-out, and with both directions present
                # they also identify its mechanism: a fixed hydrophobic bias stays positive
                # from HIGH-GRAVY starts, whereas mean-reversion toward typical composition
                # flips sign there.
                base_rows = [r for r in sub if r["alpha"] == neutral_alpha]
                for corrected, name in ((False, "naive"), (True, "corrected")):
                    v = [r["concept_score"] for r in base_rows if r["corrected"] == corrected]
                    if v:
                        print(f"{tag} baseline drift (alpha={neutral_alpha}, NO steering) "
                              f"{name}: mean_delta={np.mean(v):+.4f} (n={len(v)})")
                for a, st in attrib_by_dir[d].items():
                    parts = []
                    for name in ("naive", "corrected", "gap"):
                        if name in st:
                            pv = st[name]["wilcoxon_p"]
                            parts.append(f"{name}={st[name]['mean']:+.4f}"
                                         f"(n={st[name]['n']},p="
                                         f"{'n/a' if pv is None else f'{pv:.4f}'})")
                    print(f"{tag} steering-attributable alpha={a}: " + "  ".join(parts))
                for a, st in comp_by_dir[d].items():
                    print(f"{tag} composition distance alpha={a}: "
                          + "  ".join(f"{k}={v:.2f}pp" for k, v in st.items()))

            # single-direction runs keep the flat (unkeyed) schema so older readers still work
            if len(dirs_present) == 1:
                d = dirs_present[0]
                run.log_summary(paired_wilcoxon=w_by_dir[d],
                                steering_attributable=attrib_by_dir[d],
                                composition_distance=comp_by_dir[d])
                regimes = regimes_by_dir[d]
            else:
                run.log_summary(paired_wilcoxon_by_direction=w_by_dir,
                                steering_attributable_by_direction=attrib_by_dir,
                                composition_distance_by_direction=comp_by_dir)
                regimes = regimes_by_dir

            diverg = divergence_by_alpha(rows_for_analysis, alphas, neutral_alpha)
            if diverg:
                run.log_summary(divergence=diverg)
                for a, st in diverg.items():
                    for arm in ("naive", "corrected"):
                        if arm not in st:
                            continue
                        bits = []
                        for flavour in ("raw", "attributable"):
                            if flavour in st[arm]:
                                c = st[arm][flavour]
                                bits.append(f"{flavour}: up={c['up']:+.4f} down={c['down']:+.4f} "
                                            f"diverg={c['divergence']:+.4f} "
                                            f"(n={c['n_up']}/{c['n_down']})")
                        print(f"[exp-b] divergence alpha={a} {arm}: " + "  |  ".join(bits))
            elif len(dirs_present) > 1:
                print("[exp-b] WARNING: both directions present but divergence could not be "
                      "computed (no shared alphas with usable rows).")
        # structural naturalness axis (only populated when --fold ran); split per direction
        # for the same reason the per-alpha stats are — the two trajectories aren't comparable
        # pooled.
        fold_dirs = sorted({r.get("direction", "up") for r in rows_for_analysis},
                           key=lambda d: 0 if d == "up" else 1)
        folds = {d: fold_by_alpha([r for r in rows_for_analysis
                                   if r.get("direction", "up") == d], alphas)
                 for d in fold_dirs}
        folds = {d: v for d, v in folds.items() if v}
        if folds:
            run.log_summary(fold_confidence=(folds[fold_dirs[0]] if len(folds) == 1 else folds))
            for d, per_alpha in folds.items():
                for a, st in per_alpha.items():
                    print(f"[exp-{args.experiment.lower()}/{d}] fold alpha={a}: "
                          + "  ".join(f"{k}={v:.3f}" for k, v in st.items()
                                      if not k.endswith("_n")))
        run.log_summary(three_regime=regimes)
        print(f"[steering] wrote {csv_path}")

    print(f"[steering] done. outputs under {out_dir}")


if __name__ == "__main__":
    main()
