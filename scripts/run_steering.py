#!/usr/bin/env python
"""On-manifold steering, Exp 4.1: SAE zinc-finger clamp + GLP SDEdit cleanup (PLAN §11, W4).

Implements the MVP slice of `glprot_steering_spec.md`:
  Part 0 — off-manifold diagnostic: does SAE clamping raise GLP diffusion loss,
           worsening with clamp strength alpha? (no generation needed)
  Part 1 — shared masked-Gibbs generation/masking loop (steer only the positions
           being re-predicted each round — UPDATE-TOKENS; GLP SDEdit cleanup is
           applied to those same positions before the forward pass continues)
  Part 2 — Exp 4.1 itself: SAE-feature clamp on a zinc-finger feature, naive vs.
           GLP-corrected, evaluated by C2H2-pattern concept score + pseudo-
           perplexity naturalness (+ optional ESMFold pLDDT) + diffusion loss
  Part 4 — Pareto plot (naive vs. corrected) + the "rescue" sanity check

Exp 4.2 (DiffMean/signal-peptide) is not implemented here — Part 1's loop takes
a `steer_fn`/`correct_fn` pair so it drops in later without rework.

`plm-sae` (github.com/edithvillegas/plm-sae, the codebase the spec names for the
generation loop / C2H2 detector) was checked directly (not assumed absent): its
`interpretability/` scripts only correlate SAE activations with Swiss-Prot
annotation spans (no sequence generation, no masking/refinement loop, no C2H2 or
any other pattern detector). It also ships its own SAE
(`evillegasgarcia/sae_esm2_6_l3`, hooked at the same layer-4 point but a
different architecture/training run from InterPLM's) — not used here; per the
spec's own Context section this experiment clamps InterPLM's SAE. The masked-
Gibbs loop and the C2H2 regex below are implemented directly from the spec's own
description/pseudocode, not ported code.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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
from glp_plm.experiments import RunLogger
from glp_plm.experiments import style
from glp_plm.model import GLPDenoiser
from glp_plm.utils import get_device, resolve_config_data_dir, resolve_storage_dir, set_seed

AA20 = "ACDEFGHIKLMNPQRSTVWY"
# Spec's C2H2 zinc-finger pattern: Cys-X(2-4)-Cys-X(12)-His-X(3-5)-His.
C2H2_RE = re.compile(r"C.{2,4}C.{12}H.{3,5}H")
# Looser variant (wider spacer tolerances) to check whether the strict spec pattern is
# undercounting real zinc-finger-like structure the strict regex just misses by a residue
# or two, vs. the motif genuinely not being there.
C2H2_RE_LOOSE = re.compile(r"C.{1,6}C.{9,15}H.{1,7}H")


# --------------------------------------------------------------------------- #
#  SAE clamp (reuses the SAE's own normalize/encode/decode/unnormalize path —
#  the same path `forward(x, unnormalize=True)` uses internally for steering,
#  per InterPLM/interplm/sae/dictionary.py's own docstring)
# --------------------------------------------------------------------------- #
def sae_clamp(sae, x: torch.Tensor, feat_idxs, alpha: float) -> torch.Tensor:
    """Encode `x`, clamp feature(s) `feat_idxs` (int or list of ints — clamping more
    than one simultaneously is how arXiv:2502.09135 gets from 1.7% to 13% zinc-finger
    hits, see `select_zinc_finger_features`) to `alpha * activation_rescale_factor`
    each, decode back to raw activation space. `x` is raw (un-standardized).

    Adds back the SAE's own reconstruction error (`x_norm - decode(encode(x_norm))`)
    on top of the clamped decode, i.e. `x* = fdec(f*) + (x - fdec(f))` — the
    recon+error convention `docs/INTERPLM_MAP.md`'s steering row points at, and the
    one arXiv:2502.09135 (the plm-sae paper this experiment's generation loop is
    modeled after, §3.4 eq. 7-8) uses. Without it, everything the SAE dictionary
    fails to reconstruct (not just the target feature) gets zeroed out of the
    activation on every steered position — extra off-manifold distortion beyond
    what the clamp itself needs."""
    idxs = [feat_idxs] if isinstance(feat_idxs, int) else list(feat_idxs)
    x_norm, orig_norms = sae._normalize_input_and_get_norms(x)
    feats = sae.encode(x_norm).clone()
    error = x_norm - sae.decode(feats)
    for idx in idxs:
        feats[..., idx] = alpha * sae.activation_rescale_factor[idx]
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
    the first steering pass, the AUC-only pick (feature 7033, val AUC 0.952) peaked
    on Arg/Glu/Lys in 79% of 90 generations and on His in 0% — a linker-composition
    proxy, not a genuine Cys-X(2-4)-Cys-X(12)-His-X(3-5)-His detector; clamping it
    can't produce a motif it was never detecting. F1 (precision-at-a-real-threshold,
    recall per domain instance) is the criterion that would have caught this.

    AUC is still used for a cheap `shortlist_k`-candidate shortlist on a balanced
    subsample (`pr.topk_filter`, same filter/budget run_probing.py's own F1 path
    uses) before the exact, prevalence-correct F1 pass, which needs the full split
    and would be too slow to run over all 10240 features from a cold start.

    arXiv:2502.09135 §4.1 (the plm-sae paper this experiment is modeled after) measures
    a single clamped latent finding the C2H2 motif in only 3/180 generations (1.7%)
    vs. 24/180 (13%) when clamping its two most prominent zinc-finger latents together
    — `top_n=2` is there to close that gap, not just a knob."""
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


def compute_diffmean_vector(processed: Path, layer: int, concept: str,
                            n_shards: Optional[int] = None) -> Tuple[torch.Tensor, Dict]:
    """Difference-of-means steering vector (arXiv:2509.07983 Eq. 2): mean raw activation
    over concept-positive-labeled residues minus mean over concept-negative residues, on
    the valid split at natural prevalence. Residue-level (not the paper's whole-protein
    average — this concept is a local motif, and this repo already has residue-level
    labels from the same `sl.load_split` the SAE F1 path uses). No shortlist/threshold
    search needed, unlike `select_zinc_finger_features` — this is a single vector, and
    its direction is canonically positive-toward-the-property by construction."""
    acts_val, labels_val, names = sl.load_split(processed, "valid", layer, n_shards=n_shards)
    assert concept in names, f"{concept!r} not in eval store concepts (have e.g. {names[:5]}...)"
    y = labels_val[concept].to_numpy().astype(bool)
    pos, neg = acts_val[y].float(), acts_val[~y].float()
    v = pos.mean(0) - neg.mean(0)
    return v, {"concept": concept, "n_pos_residues": int(y.sum()), "n_neg_residues": int((~y).sum()),
               "vector_norm": float(v.norm())}


# --------------------------------------------------------------------------- #
#  GLP: per-sample diffusion loss (Part 0) and SDEdit correction (Part 1/GLP correct)
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
#  Part 0 — off-manifold diagnostic
# --------------------------------------------------------------------------- #
def run_off_manifold_diagnostic(model, scaler, clamp_fn: Callable[[torch.Tensor, float], torch.Tensor],
                                alphas, baseline_acts,
                                out_dir: Path, run: RunLogger, device: str,
                                n_repeats: int, seed: int):
    """`clamp_fn(acts_raw, alpha) -> acts_raw` is steering-method-agnostic (SAE-clamp or
    diff-of-means) — Part 0 only needs the raw-activation-in/raw-activation-out contract."""
    g = torch.Generator(device=device).manual_seed(seed)
    loss_baseline = diffusion_loss(model, scaler, baseline_acts, n_repeats=n_repeats,
                                   device=device, generator=g)

    rows = [{"alpha": "baseline", "sample_idx": i, "diffusion_loss": float(v)}
            for i, v in enumerate(loss_baseline)]
    means = {"baseline": float(loss_baseline.mean())}
    for alpha in alphas:
        steered = clamp_fn(baseline_acts.to(device), alpha)
        loss_a = diffusion_loss(model, scaler, steered, n_repeats=n_repeats,
                                device=device, generator=g)
        rows += [{"alpha": alpha, "sample_idx": i, "diffusion_loss": float(v)}
                 for i, v in enumerate(loss_a)]
        means[alpha] = float(loss_a.mean())
        print(f"[diagnostic] alpha={alpha:<5} mean diffusion loss = {means[alpha]:.5f}"
              f"  (baseline {means['baseline']:.5f})", flush=True)

    csv_path = out_dir / "off_manifold_diagnostic.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["alpha", "sample_idx", "diffusion_loss"])
        w.writeheader()
        w.writerows(rows)

    monotonic = all(means[alphas[i]] <= means[alphas[i + 1]] + 1e-9 for i in range(len(alphas) - 1))
    if not monotonic:
        print("[diagnostic] WARNING: loss did not rise monotonically with alpha — "
              "reconsider the off-manifold framing before proceeding (spec Part 0 step 5).")
    run.log_summary(diagnostic_means=means, diagnostic_monotonic=monotonic)

    style.apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.5))
    xs = [str(a) for a in alphas]
    sems = [float(np.std([r["diffusion_loss"] for r in rows if r["alpha"] == a]) / np.sqrt(
        max(sum(1 for r in rows if r["alpha"] == a), 1))) for a in alphas]
    ax.errorbar(xs, [means[a] for a in alphas], yerr=[1.96 * s for s in sems],
               marker="o", color=style.ACCENT, label="SAE-clamped")
    ax.axhline(means["baseline"], color=style.BASELINE, linestyle="--", label="baseline (unsteered)")
    ax.set_xlabel("clamp strength alpha (x feature max activation)")
    ax.set_ylabel("GLP diffusion loss (mean ± 95% CI)")
    ax.set_title("Off-manifold diagnostic: SAE clamp vs. GLP diffusion loss")
    ax.legend()
    style.save_fig(fig, out_dir / "off_manifold_diagnostic.png")
    print(f"[diagnostic] wrote {csv_path} + off_manifold_diagnostic.png (monotonic={monotonic})")


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
#  Part 1 — shared masked-Gibbs generation loop with per-round steering
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


def masked_gibbs_generate(steerer: ESMSteerer, length: int,
                          steer_fn: Callable[[torch.Tensor], torch.Tensor],
                          correct_fn: Callable[[torch.Tensor], torch.Tensor],
                          mask_fraction: float, n_rounds: int, stabilize_tol: float,
                          sample_mode: str, temperature: float,
                          generator: torch.Generator, device: str,
                          diag_fn: Optional[Callable[[int, torch.Tensor, torch.Tensor], None]] = None,
                          select_fn: Optional[Callable[[int, str], float]] = None
                          ) -> str:
    """Spec Part 1: fully-masked init -> unsteered round-0 fill -> `n_rounds` of
    (re-mask a uniform-random subset -> forward -> steer+correct those positions'
    layer-4 activations (UPDATE-TOKENS) -> forward again with the override -> sample).

    `diag_fn(round_idx, clamped, corrected)`, if given, is called each round with the
    SAE-clamped activations before SDEdit and the same positions after — lets a
    caller measure whether the correction is actually doing anything at the point
    it's applied, instead of only before generation (Part 0) or after it's fully
    done (final per-sequence diffusion_loss eval).

    `select_fn(round_idx, seq)`, if given, is called after every round (round_idx=-1
    for the unsteered round-0 fill) with the sequence decoded *so far*, and should
    return a score computed from the sequence's own natural (unclamped) properties —
    e.g. the target SAE feature's activation on it. The round with the highest score
    is returned instead of unconditionally the last one: the per-round clamp only
    biases that round's sampling logits, it doesn't persist as a property of the
    finished protein, so later rounds can drift away from an earlier, better state
    with nothing to recover it (arXiv:2502.09135 §3.4 keeps the max-activation
    iteration out of 100 for the same reason — this is that piece)."""
    tok = steerer.tokenizer
    L = length
    tokens = torch.full((1, L + 2), tok.mask_token_id, dtype=torch.long, device=device)
    tokens[0, 0], tokens[0, -1] = tok.cls_token_id, tok.eos_token_id
    attn = torch.ones(1, L + 2, dtype=torch.long, device=device)

    def decode() -> str:
        ids = tokens[0, 1:L + 1].tolist()
        return "".join(tok.convert_ids_to_tokens(i) for i in ids)

    # round 0: plain fill, no steering (spec step 2)
    logits0, _ = steerer.forward(tokens, attn, override=None)
    new_tok = _sample_positions(logits0[0, 1:L + 1].detach().cpu(), steerer.valid_aa_ids,
                                sample_mode, temperature, generator)
    tokens[0, 1:L + 1] = new_tok.to(device)

    best_seq, best_score = None, float("-inf")
    if select_fn is not None:
        best_seq = decode()
        best_score = select_fn(-1, best_seq)

    k = max(1, round(mask_fraction * L))
    for round_idx in range(n_rounds):
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

        if select_fn is not None:
            cur_seq = decode()
            score = select_fn(round_idx, cur_seq)
            if score > best_score:
                best_seq, best_score = cur_seq, score

        changed = (tokens[0, pos] != prev).float().mean().item()
        if changed < stabilize_tol:
            break

    return best_seq if select_fn is not None else decode()


# --------------------------------------------------------------------------- #
#  Part 2 — Exp 4.1 evaluation (concept / naturalness / diffusion loss)
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
    characterize what a feature detects (color activation-per-position by amino
    acid identity, see `visualize_protein_feature` in their feature-dashboard code):
    if 7033/3684 are genuine C2H2 detectors, their peak position should land on
    Cys/His far more than the ~10% (2/20 canonical AAs) base rate; if it's close to
    that base rate, the AUC that selected them is picking up something else (general
    cysteine content, disulfide context, etc.), not the C2H2 spacing pattern itself."""
    idxs = [feat_idxs] if isinstance(feat_idxs, int) else list(feat_idxs)
    x_norm, _ = sae._normalize_input_and_get_norms(acts_raw.to(device))
    feats = sae.encode(x_norm)[..., idxs]
    if feats.dim() == 1:
        feats = feats.unsqueeze(-1)
    return int(feats.max(dim=-1).values.argmax().item())


@torch.no_grad()
def diffmean_projection(v: torch.Tensor, acts_raw: torch.Tensor, device: str):
    """Mean/max cosine similarity of `v` (the diff-of-means steering vector) against
    each position's raw activation (arXiv:2509.07983 Eq. 4's relatedness score) — the
    DiffMean analog of `sae_feature_activation`: a direct, SAE-independent check of
    whether the target direction is actually present in the finished sequence's own
    (unclamped, re-embedded) activations. Max, not mean, for the same reason
    `sae_feature_activation`'s caller uses max in `select_fn`: a local hit can be
    diluted to noise by a length-averaged mean."""
    x = acts_raw.to(device).float()
    sim = F.cosine_similarity(x, v.to(device).float().expand_as(x), dim=-1)
    return float(sim.mean()), float(sim.max())


@torch.no_grad()
def diffmean_peak_position(v: torch.Tensor, acts_raw: torch.Tensor, device: str) -> int:
    """Residue index of highest cosine similarity to `v` — the DiffMean analog of
    `sae_feature_peak_position`, same validity-check role: does the peak land on
    Cys/His far above base rate, or is the direction picking up something else."""
    x = acts_raw.to(device).float()
    sim = F.cosine_similarity(x, v.to(device).float().expand_as(x), dim=-1)
    return int(sim.argmax().item())


# --------------------------------------------------------------------------- #
#  Method dispatch — SAE-clamp vs. diff-of-means share the rest of the script
#  (Parts 0/1/2/4) through this raw-activation-in/raw-activation-out contract.
# --------------------------------------------------------------------------- #
def make_steer_fn(method: str, alpha: float, sae=None, feat_idxs=None, v: Optional[torch.Tensor] = None):
    if method == "sae":
        # alpha=0.0 is the null-baseline condition (see steer_fn's docstring at the
        # main() call site): sae_clamp(..., alpha=0) would still overwrite the feature
        # (suppressing it), a real off-manifold intervention, not "unsteered" — so it's
        # special-cased to a no-op here. diffmean_clamp needs no such special-casing
        # (alpha=0 -> x_new == x exactly).
        if alpha == 0.0:
            return lambda a: a
        return lambda a: sae_clamp(sae, a, feat_idxs, alpha)
    return lambda a: diffmean_clamp(v, a, alpha)


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


def fold_plddt(seq: str, device: str, _cache: dict = {}) -> Optional[float]:
    """Best-effort ESMFold pLDDT (--fold, opt-in; PLAN.md D4 flags this Tier-2/
    feasibility-gated: 16-24GB GPU mem, not exercised/verified in this session)."""
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
        # per-atom pLDDT, (1, L, 37); CA = atom index 1. Reported on a 0-100 scale upstream.
        return float(out["plddt"][0, :, 1].mean()) * 100.0
    except Exception as e:  # opt-in path; don't take down the run over a folding failure
        print(f"[exp41] WARNING: --fold pLDDT failed for one sequence ({e}); recording None")
        return None


def sample_protein_lengths(processed: Path, split: str, n: int, seed: int,
                           n_shards: Optional[int] = None,
                           min_len: int = 50, max_len: int = 400) -> List[int]:
    import pandas as pd

    shard_ids = sl.split_shards(processed, split, n_shards)
    lengths = []
    for i in shard_ids:
        df = pd.read_csv(processed / f"shard_{i}" / "protein_data.tsv", sep="\t")
        seq_col = next(c for c in df.columns if c.lower() == "sequence")
        lengths.extend(len(s) for s in df[seq_col].astype(str))
    lengths = [l for l in lengths if min_len <= l <= max_len]
    assert lengths, f"no protein lengths in [{min_len},{max_len}] found under {processed}/{split}"
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.choice(lengths, size=n, replace=True)]


# --------------------------------------------------------------------------- #
#  Part 4 — Pareto plot + rescue sanity check
# --------------------------------------------------------------------------- #
def pareto_and_sanity_check(rows: List[dict], out_dir: Path, run: RunLogger):
    naive = [r for r in rows if not r["corrected"]]
    corrected = [r for r in rows if r["corrected"]]

    style.apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    if naive:
        ax.scatter([r["naturalness_score"] for r in naive], [r["concept_score"] for r in naive],
                  c=style.method_color("sae"), label="naive steering (SAE only)", alpha=0.75)
    if corrected:
        ax.scatter([r["naturalness_score"] for r in corrected], [r["concept_score"] for r in corrected],
                  c=style.method_color("glp"), label="GLP-corrected", alpha=0.75)
    ax.set_xlabel("naturalness (pseudo-perplexity; lower = more natural)")
    ax.set_ylabel("concept score (# C2H2 pattern matches)")
    ax.set_title("Exp 4.1 Pareto: zinc-finger concept vs. naturalness")
    ax.legend()
    style.save_fig(fig, out_dir / "pareto.png")

    # Sanity check (spec Part 4.2): at matched-or-better naturalness, GLP-corrected
    # concept score should beat whatever naive steering achieves there. A corrected
    # point with NO naturalness-matched naive comparator is excluded rather than
    # counted as a free win: the old `default=-1.0` fallback made `concept_score=0 >
    # -1.0` pass trivially, so a corrected point that merely beat every naive point's
    # naturalness (concept_score still 0 everywhere) registered as "rescued" — that's
    # what produced the exp41_pilot's vacuous 29% with zero real concept hits.
    rescued = comparable = 0
    for c in corrected:
        matched = [n for n in naive if n["naturalness_score"] <= c["naturalness_score"]]
        if not matched:
            continue
        comparable += 1
        if c["concept_score"] > max(n["concept_score"] for n in matched):
            rescued += 1
    frac = rescued / max(comparable, 1)
    if comparable == 0:
        print("[exp41] WARNING: no GLP-corrected point has a naturalness-matched naive "
              "comparator — rescue fraction is undefined (spec Part 4.2).")
    elif frac < 0.5:
        print(f"[exp41] WARNING: rescue sanity check only holds for {frac:.0%} of "
              f"{comparable} comparable GLP-corrected points — the 'rescue' framing "
              "needs revisiting (spec Part 4.2).")
    else:
        print(f"[exp41] rescue sanity check holds for {frac:.0%} of {comparable} comparable "
              "GLP-corrected points.")
    run.log_summary(pareto_rescue_fraction=frac, pareto_rescue_comparable=comparable,
                    pareto_rescue_n=len(corrected))


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

    ap.add_argument("--steer-method", choices=["sae", "diffmean"], default="sae",
                    help="'sae': InterPLM SAE-feature clamp (exp41/exp41c). 'diffmean': "
                         "difference-of-means activation addition (arXiv:2509.07983 Eq. 1-3) — "
                         "h_tilde = h + alpha*v, v = mean(pos-labeled residues) - "
                         "mean(neg-labeled residues), renormalized to ||h||. Needs no SAE.")
    ap.add_argument("--zinc-finger-concept", default="Zinc finger_C2H2-type")
    ap.add_argument("--zinc-finger-feature",
                    help="[sae only] override/skip auto-selection; comma-separated for multiple "
                         "simultaneous features (e.g. '7033,4210')")
    ap.add_argument("--top-n-features", type=int, default=1,
                    help="[sae only] auto-select this many top features by domain-adjusted F1 "
                         "(InterPLM's own validated criterion, select-on-valid/confirm-on-test — "
                         "see select_zinc_finger_features) when --zinc-finger-feature is not given "
                         "(arXiv:2502.09135 finds top_n=2 much stronger than 1)")

    ap.add_argument("--alphas", default=None,
                    help="clamp strengths. For --steer-method sae: multiples of the feature's max "
                         "activation (default '0,2,3,4' — exp41c (1,2,3,4,6) showed all the "
                         "concept-hit signal at 2/3/4). For diffmean: a literal multiplier on the "
                         "raw mean-difference vector (default '0,0.5,1,2,4' — arXiv:2509.07983's "
                         "own scale, default alpha=1.0, sweet spot 0.5-2, reported collapse at "
                         "alpha=20). 0.0 is always the null-baseline value (see make_steer_fn) — "
                         "NO steering at all, to calibrate the C2H2 regex's own false-positive rate.")
    ap.add_argument("--n-samples-diagnostic", type=int, default=2000)
    ap.add_argument("--n-repeats", type=int, default=4, help="diffusion-loss t/eps repeats")

    ap.add_argument("--n-sequences", type=int, default=300, help="generated sequences per (alpha, "
                    "corrected). Exp 4.1c's n=64 only produced 1-2 strict-regex hits per cell — "
                    "too few to trust; 300 is ~4.7x that, and exp41c's 640-sequence run (5 "
                    "alphas) only took 17m25s of a 2h budget, so there's ample headroom.")
    ap.add_argument("--seq-length-source", choices=["eval", "fixed"], default="eval")
    ap.add_argument("--fixed-length", type=int, default=150)
    ap.add_argument("--mask-fraction", type=float, default=0.15, help="fraction of positions "
                    "re-masked per round. exp41_pilot's 1.0 (full remask every round) can't let "
                    "any locally-built motif persist into the next round; small+persistent is "
                    "the incremental-refinement design Part 1 intends.")
    ap.add_argument("--n-rounds", type=int, default=24, help="more rounds than exp41's 8, since "
                    "each round now perturbs less (mask_fraction=0.15) and ESM-8M forwards are "
                    "cheap")
    ap.add_argument("--stabilize-tol", type=float, default=0.05)
    ap.add_argument("--sample-mode", choices=["sample", "argmax"], default="sample")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--pppl-batch-size", type=int, default=32)

    ap.add_argument("--t-start", type=float, default=0.5, help="GLP SDEdit start noise level")
    ap.add_argument("--t-start-pilot", action="store_true",
                    help="run only the t_start in {0.3,0.5,0.7} pilot (spec Part 1) and exit")
    ap.add_argument("--sde-steps", type=int, default=20)
    ap.add_argument("--log-round-diagnostics", action="store_true",
                    help="log per-round diffusion loss before/after SDEdit (corrected=True rows "
                         "only) to <out-dir>/round_diagnostics.csv — debug aid for checking "
                         "whether the correction is doing anything where it's applied, not just "
                         "before generation (Part 0) or on the finished sequence")
    ap.add_argument("--diag-n-repeats", type=int, default=1,
                    help="t/eps repeats for --log-round-diagnostics' diffusion-loss calls "
                         "(kept low by default — this runs every round, not once per sequence)")

    ap.add_argument("--fold", action="store_true", help="also compute ESMFold pLDDT (heavy, opt-in)")
    ap.add_argument("--skip-diagnostic", action="store_true")
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--restart", action="store_true", help="ignore any existing partial CSVs")
    ap.add_argument("--out-dir", help="default <output_dir>/steering_exp41")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

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

    out_dir = Path(args.out_dir) if args.out_dir else out / "steering_exp41"
    out_dir.mkdir(parents=True, exist_ok=True)
    run = RunLogger(out_dir)
    run.log_config(cfg)

    scaler = Scaler.load(args.scaler or ckpt_dir / "rep_statistics.pt").to(device)
    model = GLPDenoiser(cfg.model).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt or ckpt_dir / "glp_final.pt", map_location=device))

    sae = feat_idxs = diffmean_vec = None
    if args.steer_method == "sae":
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
        default_alphas = "0,2,3,4"
    else:
        vec_path = out_dir / "diffmean_vector.json"
        if vec_path.exists() and not args.restart:
            d = json.loads(vec_path.read_text())
            diffmean_vec = torch.tensor(d["vector"], dtype=torch.float32)
        else:
            diffmean_vec, meta = compute_diffmean_vector(
                processed, cfg.data.layer, args.zinc_finger_concept, n_shards=args.n_shards_per_split)
            vec_path.write_text(json.dumps({"vector": diffmean_vec.tolist(), **meta}, indent=2))
            print(f"[steering] diffmean vector for {args.zinc_finger_concept!r}: "
                  f"{meta['n_pos_residues']} pos / {meta['n_neg_residues']} neg residues, "
                  f"||v||={meta['vector_norm']:.4f}")
        default_alphas = "0,0.5,1,2,4"

    alphas = [float(a) for a in (args.alphas or default_alphas).split(",") if a.strip()]

    steerer = ESMSteerer(cfg.data.model_name, cfg.data.layer, device)

    if args.t_start_pilot:
        extractor = ActivationExtractor(cfg.data.model_name, cfg.data.layer, device=device)
        lengths = sample_protein_lengths(processed, "test", 8, args.seed, n_shards=args.n_shards_per_split)
        pilot_rows = []
        for t_start in (0.3, 0.5, 0.7):
            for i, length in enumerate(lengths):
                # Two generators: masked_gibbs_generate's own randperm/multinomial run on
                # CPU (needs a CPU generator), but glp_correct/diffusion_loss call
                # torch.randn(..., device=device, generator=...) which requires a
                # generator on that SAME device — torch errors on a device mismatch.
                seed_i = args.seed * 1000 + i
                g = torch.Generator().manual_seed(seed_i)
                g_dev = torch.Generator(device=device).manual_seed(seed_i)
                seq = masked_gibbs_generate(
                    steerer, length,
                    steer_fn=make_steer_fn(args.steer_method, alphas[len(alphas) // 2],
                                           sae=sae, feat_idxs=feat_idxs, v=diffmean_vec),
                    correct_fn=lambda a: glp_correct(model, scaler, a, t_start=t_start,
                                                     num_steps=args.sde_steps, device=device, generator=g_dev),
                    mask_fraction=args.mask_fraction, n_rounds=args.n_rounds,
                    stabilize_tol=args.stabilize_tol, sample_mode=args.sample_mode,
                    temperature=args.temperature, generator=g, device=device)
                acts = torch.from_numpy(extractor.extract_single(seq)).float()
                dloss = float(diffusion_loss(model, scaler, acts, n_repeats=args.n_repeats,
                                             device=device, generator=g_dev).mean())
                pilot_rows.append({"t_start": t_start, "sample_idx": i, "length": length,
                                  "concept_score": c2h2_concept_score(seq),
                                  "pseudo_perplexity": pseudo_perplexity(steerer, seq, device),
                                  "diffusion_loss": dloss})
                print(f"[t_start-pilot] t_start={t_start} sample={i} concept={pilot_rows[-1]['concept_score']} "
                      f"pppl={pilot_rows[-1]['pseudo_perplexity']:.2f} dloss={dloss:.4f}")
        pilot_csv = out_dir / "t_start_pilot.csv"
        with open(pilot_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pilot_rows[0].keys()))
            w.writeheader(); w.writerows(pilot_rows)
        print(f"[t_start-pilot] wrote {pilot_csv} — pick a t_start from this, then re-run "
              f"without --t-start-pilot (--t-start <chosen>).")
        return

    if not args.skip_diagnostic:
        acts_test, _, _ = sl.load_split(processed, "test", cfg.data.layer, n_shards=args.n_shards_per_split)
        g = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(acts_test.shape[0], generator=g)[:args.n_samples_diagnostic]
        baseline_acts = acts_test[idx].float()
        # alpha=0.0 (null-baseline condition, see make_steer_fn) is redundant here —
        # the diagnostic already has its own unclamped "baseline" row.
        diag_alphas = [a for a in alphas if a != 0.0]
        diag_clamp_fn = (lambda a, al: sae_clamp(sae, a, feat_idxs, al)) if args.steer_method == "sae" \
            else (lambda a, al: diffmean_clamp(diffmean_vec, a, al))
        run_off_manifold_diagnostic(model, scaler, diag_clamp_fn, diag_alphas, baseline_acts,
                                    out_dir, run, device, args.n_repeats, args.seed)

    if not args.skip_generate:
        extractor = ActivationExtractor(cfg.data.model_name, cfg.data.layer, device=device)
        if args.seq_length_source == "eval":
            lengths = sample_protein_lengths(processed, "test", args.n_sequences, args.seed,
                                            n_shards=args.n_shards_per_split)
        else:
            lengths = [args.fixed_length] * args.n_sequences

        csv_path = out_dir / "exp41_sae_results.csv"
        existing = []
        if csv_path.exists() and not args.restart:
            with open(csv_path) as f:
                existing = list(csv.DictReader(f))
        done_counts = {}
        for r in existing:
            key = (r["alpha"], r["corrected"])
            done_counts[key] = done_counts.get(key, 0) + 1

        fieldnames = ["sample_id", "alpha", "corrected", "length", "concept_score",
                     "concept_score_loose", "naturalness_score", "diffusion_loss",
                     "max_homopolymer_run", "unique_aa_count",
                     "target_feature_activation_mean", "target_feature_activation_max",
                     "target_feature_peak_residue", "plddt", "sequence"]
        mode = "a" if existing else "w"
        rows_for_pareto = [
            {**r, "alpha": float(r["alpha"]), "corrected": r["corrected"] == "True",
             "concept_score": float(r["concept_score"]), "naturalness_score": float(r["naturalness_score"])}
            for r in existing
        ]
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

        select_path = out_dir / "round_feature_activation.csv"
        select_fieldnames = ["sample_id", "alpha", "corrected", "round",
                             "feature_activation_mean", "feature_activation_max"]
        select_file = select_writer = None
        if args.log_round_diagnostics:
            select_mode = "a" if select_path.exists() and not args.restart else "w"
            select_file = open(select_path, select_mode, newline="")
            select_writer = csv.DictWriter(select_file, fieldnames=select_fieldnames)
            if select_mode == "w":
                select_writer.writeheader()

        with open(csv_path, mode, newline="") as fcsv:
            w = csv.DictWriter(fcsv, fieldnames=fieldnames)
            if mode == "w":
                w.writeheader()
            for alpha in alphas:
                for corrected in (False, True):
                    key = (str(alpha), str(corrected))
                    n_done = done_counts.get(key, 0)
                    for i in range(n_done, args.n_sequences):
                        length = lengths[i]
                        # See t_start-pilot block above: masking/token-sampling needs a CPU
                        # generator, glp_correct/diffusion_loss need one on `device`.
                        seed_i = args.seed * 1_000_003 + sample_id
                        g = torch.Generator().manual_seed(seed_i)
                        g_dev = torch.Generator(device=device).manual_seed(seed_i)

                        # alpha=0.0 is the null-baseline condition (§7 suggested next step): NO
                        # steering at all — needed to calibrate the C2H2 regex's own
                        # false-positive rate on this model's natural generation distribution
                        # (exp41c reported 4/640 strict hits with no idea whether that's above
                        # chance). See make_steer_fn's docstring for why sae vs. diffmean handle
                        # alpha=0.0 differently under the hood but agree on behavior.
                        steer_fn = make_steer_fn(args.steer_method, alpha, sae=sae,
                                                 feat_idxs=feat_idxs, v=diffmean_vec)

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

                        def select_fn(round_idx, cur_seq, _sample_id=sample_id, _alpha=alpha,
                                     _corrected=corrected):
                            acts = torch.from_numpy(extractor.extract_single(cur_seq)).float()
                            mean_act, max_act = target_signal(args.steer_method, acts, device,
                                                              sae=sae, feat_idxs=feat_idxs, v=diffmean_vec)
                            if select_writer is not None:
                                select_writer.writerow({
                                    "sample_id": _sample_id, "alpha": _alpha, "corrected": _corrected,
                                    "round": round_idx, "feature_activation_mean": mean_act,
                                    "feature_activation_max": max_act})
                                select_file.flush()
                            # max, not mean: the target feature is sparse (active at a handful
                            # of positions at most, out of ~L), so a length-diluted mean can't
                            # tell a round with a strong local hit from one with none at all.
                            return max_act

                        seq = masked_gibbs_generate(
                            steerer, length, steer_fn, correct_fn, args.mask_fraction,
                            args.n_rounds, args.stabilize_tol, args.sample_mode,
                            args.temperature, g, device, diag_fn=diag_fn, select_fn=select_fn)

                        concept = c2h2_concept_score(seq)
                        concept_loose = c2h2_concept_score(seq, C2H2_RE_LOOSE)
                        pppl = pseudo_perplexity(steerer, seq, device, batch_size=args.pppl_batch_size)
                        acts_final = torch.from_numpy(extractor.extract_single(seq)).float()
                        dloss = float(diffusion_loss(model, scaler, acts_final, n_repeats=args.n_repeats,
                                                     device=device, generator=g_dev).mean())
                        feat_act_mean, feat_act_max = target_signal(
                            args.steer_method, acts_final, device, sae=sae, feat_idxs=feat_idxs, v=diffmean_vec)
                        peak_pos = target_peak_position(
                            args.steer_method, acts_final, device, sae=sae, feat_idxs=feat_idxs, v=diffmean_vec)
                        peak_residue = seq[peak_pos] if 0 <= peak_pos < len(seq) else None
                        plddt = fold_plddt(seq, device) if args.fold else None

                        row = {"sample_id": sample_id, "alpha": alpha, "corrected": corrected,
                              "length": length, "concept_score": concept,
                              "concept_score_loose": concept_loose, "naturalness_score": pppl,
                              "diffusion_loss": dloss, "max_homopolymer_run": max_homopolymer_run(seq),
                              "unique_aa_count": len(set(seq)),
                              "target_feature_activation_mean": feat_act_mean,
                              "target_feature_activation_max": feat_act_max,
                              "target_feature_peak_residue": peak_residue,
                              "plddt": plddt, "sequence": seq}
                        w.writerow(row)
                        fcsv.flush()
                        rows_for_pareto.append(row)
                        sample_id += 1
                        print(f"[exp41] alpha={alpha} corrected={corrected} sample={i} "
                              f"concept={concept} concept_loose={concept_loose} pppl={pppl:.2f} "
                              f"dloss={dloss:.4f} max_run={row['max_homopolymer_run']} "
                              f"feat_act={feat_act_mean:.3f}/{feat_act_max:.3f} "
                              f"peak_residue={peak_residue}", flush=True)

        if diag_file is not None:
            diag_file.close()
            print(f"[exp41] wrote {diag_path}")
        if select_file is not None:
            select_file.close()
            print(f"[exp41] wrote {select_path}")

        # alpha=0.0 rows are the null-baseline control (no clamp at all) — stay in the
        # CSV for the record, but excluded from the Pareto/rescue analysis, which is
        # specifically about the concept-steering tradeoff at nonzero alpha.
        steered_rows = [r for r in rows_for_pareto if r["alpha"] != 0.0]
        pareto_and_sanity_check(steered_rows, out_dir, run)
        n_null = len(rows_for_pareto) - len(steered_rows)
        if n_null:
            null_hits = sum(1 for r in rows_for_pareto if r["alpha"] == 0.0 and r["concept_score"] > 0)
            print(f"[exp41] null-baseline (alpha=0, no clamp): {null_hits}/{n_null} strict concept "
                  "hits — the C2H2 regex's own false-positive rate on this model's unsteered "
                  "generation distribution.")
        print(f"[exp41] wrote {csv_path} + pareto.png")

    print(f"[steering] done. outputs under {out_dir}")


if __name__ == "__main__":
    main()
