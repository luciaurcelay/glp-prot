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
    `scripts/build_gravy_contrast.py` to have been run first (auto-built here if
    missing — see `--gravy-contrast`).

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
                                n_repeats: int, seed: int, experiment: str):
    """`clamp_fn(acts_raw, alpha) -> acts_raw` is steering-method-agnostic (SAE-clamp or
    diff-of-means) — Part 0 only needs the raw-activation-in/raw-activation-out contract.
    `experiment` ("A"/"B") is recorded as a CSV column per the spec's output schema, even
    though the two experiments already write to separate out-dirs."""
    g = torch.Generator(device=device).manual_seed(seed)
    loss_baseline = diffusion_loss(model, scaler, baseline_acts, n_repeats=n_repeats,
                                   device=device, generator=g)

    rows = [{"experiment": experiment, "alpha": "baseline", "sample_idx": i, "diffusion_loss": float(v)}
            for i, v in enumerate(loss_baseline)]
    means = {"baseline": float(loss_baseline.mean())}
    for alpha in alphas:
        steered = clamp_fn(baseline_acts.to(device), alpha)
        loss_a = diffusion_loss(model, scaler, steered, n_repeats=n_repeats,
                                device=device, generator=g)
        rows += [{"experiment": experiment, "alpha": alpha, "sample_idx": i, "diffusion_loss": float(v)}
                 for i, v in enumerate(loss_a)]
        means[alpha] = float(loss_a.mean())
        print(f"[diagnostic-{experiment}] alpha={alpha:<6} mean diffusion loss = {means[alpha]:.5f}"
              f"  (baseline {means['baseline']:.5f})", flush=True)

    csv_path = out_dir / "off_manifold_diagnostic.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment", "alpha", "sample_idx", "diffusion_loss"])
        w.writeheader()
        w.writerows(rows)

    monotonic = all(means[alphas[i]] <= means[alphas[i + 1]] + 1e-9 for i in range(len(alphas) - 1))
    if not monotonic:
        print(f"[diagnostic-{experiment}] WARNING: loss did not rise monotonically with alpha — "
              "reconsider the off-manifold framing before proceeding (spec Part 0 step 5).")
    run.log_summary(**{f"diagnostic_{experiment}_means": means,
                       f"diagnostic_{experiment}_monotonic": monotonic})

    style.apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.5))
    xs = [str(a) for a in alphas]
    sems = [float(np.std([r["diffusion_loss"] for r in rows if r["alpha"] == a]) / np.sqrt(
        max(sum(1 for r in rows if r["alpha"] == a), 1))) for a in alphas]
    ax.errorbar(xs, [means[a] for a in alphas], yerr=[1.96 * s for s in sems],
               marker="o", color=style.ACCENT, label="steered")
    ax.axhline(means["baseline"], color=style.BASELINE, linestyle="--", label="baseline (unsteered)")
    ax.set_xlabel("steering strength alpha")
    ax.set_ylabel("GLP diffusion loss (mean ± 95% CI)")
    ax.set_title(f"Off-manifold diagnostic ({experiment}): steering vs. GLP diffusion loss")
    ax.legend()
    style.save_fig(fig, out_dir / "off_manifold_diagnostic.png")
    print(f"[diagnostic-{experiment}] wrote {csv_path} + off_manifold_diagnostic.png (monotonic={monotonic})")


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
    """Best-effort ESMFold pLDDT (Exp A default-on: spec's "Needs folding: Yes"; PLAN.md
    D4 flags this Tier-2/feasibility-gated: 16-24GB GPU mem)."""
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
        print(f"[steering] WARNING: --fold pLDDT failed for one sequence ({e}); recording None")
        return None


# --------------------------------------------------------------------------- #
#  Part 3 — analysis: Pareto plots, three-regime breakdown, rescue sanity check
# --------------------------------------------------------------------------- #
def is_degenerate_sequence(row: dict, min_unique_aa: int = 8, min_homopolymer_run: int = 8,
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
    sequence being one amino acid; random 20-AA composition is ~5%)."""
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
    ax.set_title("Exp A Pareto: zinc-finger hit rate vs. naturalness")
    ax.legend()
    style.save_fig(fig, out_dir / "pareto.png")

    frac, n_comp = _rescue_fraction(naive_agg, corr_agg)
    _log_rescue(run, "A", frac, n_comp, len(corr_agg))


def pareto_exp_b(rows: List[dict], out_dir: Path, run: RunLogger):
    """Per-sample scatter + mean trajectory overlay (spec explicitly allows this for
    Exp B's continuous ΔGRAVY score, unlike Exp A's aggregate-only)."""
    naive = [r for r in rows if not r["corrected"]]
    corrected = [r for r in rows if r["corrected"]]

    style.apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    if naive:
        ax.scatter([r["naturalness_score"] for r in naive], [r["concept_score"] for r in naive],
                  c=style.ACCENT, alpha=0.35, s=18, label="naive (per-sample)")
    if corrected:
        ax.scatter([r["naturalness_score"] for r in corrected], [r["concept_score"] for r in corrected],
                  c=style.method_color("glp"), alpha=0.35, s=18, label="GLP-corrected (per-sample)")

    def mean_traj(subset):
        alphas = sorted(set(r["alpha"] for r in subset))
        return [(float(np.mean([r["naturalness_score"] for r in subset if r["alpha"] == a])),
                float(np.mean([r["concept_score"] for r in subset if r["alpha"] == a]))) for a in alphas]

    for subset, color, label in ((naive, style.ACCENT, "naive (mean)"),
                                 (corrected, style.method_color("glp"), "GLP-corrected (mean)")):
        pts = mean_traj(subset)
        if pts:
            xs, ys = zip(*sorted(pts))
            ax.plot(xs, ys, "o-", color=color, linewidth=2, label=label)
    ax.set_xlabel("naturalness (pseudo-perplexity; lower = more natural)")
    ax.set_ylabel("ΔGRAVY (final - init)")
    ax.set_title("Exp B Pareto: hydrophobicity shift vs. naturalness")
    ax.legend()
    style.save_fig(fig, out_dir / "pareto.png")

    frac, n_comp = _rescue_fraction(naive, corrected)
    _log_rescue(run, "B", frac, n_comp, len(corrected))


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
    ap.add_argument("--gravy-contrast", help="[B only] path to gravy_contrast.json (default "
                                             "<out-dir>/gravy_contrast.json — built automatically "
                                             "via build_gravy_contrast.py if missing)")

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
    ap.add_argument("--sample-mode", choices=["sample", "argmax"], default="sample")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--pppl-batch-size", type=int, default=32)

    ap.add_argument("--t-start", type=float, default=0.5, help="GLP SDEdit start noise level")
    ap.add_argument("--t-start-pilot", action="store_true",
                    help="run only the t_start in {0.3,0.5,0.7} pilot (spec Part 1) and exit")
    ap.add_argument("--sde-steps", type=int, default=20)
    ap.add_argument("--log-round-diagnostics", action="store_true",
                    help="log per-round diffusion loss before/after SDEdit (corrected=True "
                         "rows only) to <out-dir>/round_diagnostics.csv, plus a per-round "
                         "trajectory log (round_feature_activation.csv for A, round_gravy.csv "
                         "for B — the latter is how you check Exp B's round-count pilot: does "
                         "ΔGRAVY plateau by round 20-50?)")
    ap.add_argument("--diag-n-repeats", type=int, default=1,
                    help="t/eps repeats for --log-round-diagnostics' diffusion-loss calls "
                         "(kept low by default — this runs every round, not once per sequence)")

    ap.add_argument("--fold", action=argparse.BooleanOptionalAction, default=None,
                    help="ESMFold pLDDT. Default: True for A (spec: 'Needs folding: Yes'), "
                         "False for B (spec: 'Needs folding: No'). Heavy (16-24GB GPU mem) — "
                         "use --no-fold for cheap A smoke tests.")
    ap.add_argument("--skip-diagnostic", action="store_true")
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--restart", action="store_true", help="ignore any existing partial CSVs")
    ap.add_argument("--out-dir", help="default <output_dir>/steering_sae_zinc (A) or "
                                      "<output_dir>/steering_diffmean_hydro (B)")
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

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = out / ("steering_sae_zinc" if is_a else "steering_diffmean_hydro")
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
        from build_gravy_contrast import build_gravy_contrast, gravy_score
        gravy_path = Path(args.gravy_contrast) if args.gravy_contrast else out_dir / "gravy_contrast.json"
        build_gravy_contrast(cfg, gravy_path, seed=args.seed, restart=args.restart, device=device)
        gd = json.loads(gravy_path.read_text())
        diffmean_vec = torch.tensor(gd["vector"], dtype=torch.float32)
        heldout = gd["heldout"]
        print(f"[steering] gravy contrast: high={gd['gravy_high_mean']:.3f} vs "
              f"low={gd['gravy_low_in_mean']:.3f} (||v||={gd['vector_norm']:.4f}), "
              f"{len(heldout)} held-out starting sequences")
        default_alphas = "0,0.05,0.25,0.5,1,2,5,20"
        neutral_alpha = 0.0  # diffmean_clamp's alpha=0 -> x_new == x exactly, no special-casing
        if args.n_sequences is not None and args.n_sequences != len(heldout):
            print(f"[steering] --n-sequences ignored for experiment B — using all "
                 f"{len(heldout)} held-out sequences (paired design)")
        args.n_sequences = len(heldout)

    if args.n_sequences is None:
        args.n_sequences = 100
    if args.mask_fraction is None:
        args.mask_fraction = 1.0 if is_a else 0.10
    if args.n_rounds is None:
        args.n_rounds = 100 if is_a else 30
    if args.fold is None:
        args.fold = is_a

    alphas = [float(a) for a in (args.alphas or default_alphas).split(",") if a.strip()]
    if is_a and neutral_alpha not in alphas:
        # The no-steering anchor (spec's mandatory Pareto anchor, "do NOT use alpha=0
        # [as steering strength], instead run a separate arm with no clamping at all")
        # rides whichever alpha value make_steer_fn treats as an algebraic no-op for
        # the active --sae-intervention (0.0 for clamp, 1.0+b=0 for scale_shift) — but
        # the default grid doesn't include it, so it has to be added explicitly or the
        # anchor arm silently never runs.
        alphas = [neutral_alpha] + alphas

    steerer = ESMSteerer(cfg.data.model_name, cfg.data.layer, device)

    def b_for(alpha: float) -> float:
        """[A, scale_shift only] the anchor (alpha==neutral_alpha) always uses b=0 (the
        algebraic identity); every other alpha uses the fixed --shift-b."""
        return 0.0 if alpha == neutral_alpha else args.shift_b

    def build_init(i: int, generator: torch.Generator):
        if is_a:
            length = int(np.random.default_rng(args.seed * 7919 + i).integers(len_lo, len_hi + 1))
            return random_aa_init_tokens(steerer.tokenizer, length, generator), length, None, None
        ho = heldout[i]
        return seq_to_init_tokens(steerer.tokenizer, ho["sequence"]), len(ho["sequence"]), ho["id"], ho["gravy"]

    if args.t_start_pilot:
        extractor = ActivationExtractor(cfg.data.model_name, cfg.data.layer, device=device)
        mid_alpha = alphas[len(alphas) // 2]
        n_pilot = min(8, args.n_sequences)
        pilot_rows = []
        for t_start in (0.3, 0.5, 0.7):
            for i in range(n_pilot):
                seed_i = args.seed * 1000 + i
                g = torch.Generator().manual_seed(seed_i)
                g_dev = torch.Generator(device=device).manual_seed(seed_i)
                init_tokens, length, init_id, gravy_init = build_init(i, g)
                seq = masked_gibbs_generate(
                    steerer, init_tokens,
                    steer_fn=make_steer_fn(method, mid_alpha, sae=sae, feat_idxs=feat_idxs, v=diffmean_vec,
                                           intervention=args.sae_intervention, b=b_for(mid_alpha)),
                    correct_fn=lambda a: glp_correct(model, scaler, a, t_start=t_start,
                                                     num_steps=args.sde_steps, device=device, generator=g_dev),
                    mask_fraction=args.mask_fraction, n_rounds=args.n_rounds,
                    stabilize_tol=args.stabilize_tol, sample_mode=args.sample_mode,
                    temperature=args.temperature, generator=g, device=device)
                acts = torch.from_numpy(extractor.extract_single(seq)).float()
                dloss = float(diffusion_loss(model, scaler, acts, n_repeats=args.n_repeats,
                                             device=device, generator=g_dev).mean())
                concept = c2h2_concept_score(seq) if is_a else gravy_score(seq) - gravy_init
                pilot_rows.append({"t_start": t_start, "sample_idx": i, "length": length,
                                  "concept_score": concept,
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
        # neutral_alpha (null-baseline condition, see make_steer_fn) is redundant here —
        # the diagnostic already has its own unclamped "baseline" row.
        diag_alphas = [a for a in alphas if a != neutral_alpha]
        if is_a and args.sae_intervention == "scale_shift":
            diag_clamp_fn = lambda a, al: sae_scale_shift(sae, a, feat_idxs, al, args.shift_b)
        elif is_a:
            diag_clamp_fn = lambda a, al: sae_clamp(sae, a, feat_idxs, al)
        else:
            diag_clamp_fn = lambda a, al: diffmean_clamp(diffmean_vec, a, al)
        run_off_manifold_diagnostic(model, scaler, diag_clamp_fn, diag_alphas, baseline_acts,
                                    out_dir, run, device, args.n_repeats, args.seed, args.experiment)

    if not args.skip_generate:
        extractor = ActivationExtractor(cfg.data.model_name, cfg.data.layer, device=device)

        csv_name = "expA_sae_zincfinger.csv" if is_a else "expB_diffmean_gravy.csv"
        csv_path = out_dir / csv_name
        existing = []
        if csv_path.exists() and not args.restart:
            with open(csv_path) as f:
                existing = list(csv.DictReader(f))
        done_counts = {}
        for r in existing:
            key = (r["alpha"], r["corrected"])
            done_counts[key] = done_counts.get(key, 0) + 1

        if is_a:
            fieldnames = ["sample_id", "alpha", "corrected", "length", "motif_hit",
                         "concept_count", "concept_count_loose", "naturalness_score",
                         "diffusion_loss", "max_homopolymer_run", "unique_aa_count",
                         "target_feature_activation_mean", "target_feature_activation_max",
                         "target_feature_peak_residue", "plddt", "sequence"]
        else:
            fieldnames = ["sample_id", "init_sequence_id", "alpha", "corrected", "length",
                         "gravy_init", "gravy_final", "delta_gravy", "naturalness_score",
                         "diffusion_loss", "max_homopolymer_run", "unique_aa_count",
                         "target_projection_mean", "target_projection_max", "sequence"]

        mode = "a" if existing else "w"
        rows_for_analysis = []
        for r in existing:
            concept_score = (1.0 if r["motif_hit"] == "True" else 0.0) if is_a else float(r["delta_gravy"])
            rows_for_analysis.append({
                **r, "alpha": float(r["alpha"]), "corrected": r["corrected"] == "True",
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

        roundlog_name = "round_feature_activation.csv" if is_a else "round_gravy.csv"
        roundlog_path = out_dir / roundlog_name
        roundlog_fieldnames = (["sample_id", "alpha", "corrected", "round",
                                "feature_activation_mean", "feature_activation_max"] if is_a
                               else ["sample_id", "alpha", "corrected", "round", "gravy_current"])
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
            for alpha in alphas:
                for corrected in (False, True):
                    key = (str(alpha), str(corrected))
                    n_done = done_counts.get(key, 0)
                    for i in range(n_done, args.n_sequences):
                        seed_i = args.seed * 1_000_003 + sample_id
                        g = torch.Generator().manual_seed(seed_i)
                        g_dev = torch.Generator(device=device).manual_seed(seed_i)

                        init_tokens, length, init_sequence_id, gravy_init = build_init(i, g)
                        # spec's selection caveat: Exp A's no-steering anchor (alpha==
                        # neutral_alpha) keeps the LAST round (select_fn=None), not the
                        # z_k-argmax criterion, to avoid cherry-picking an unsteered run's
                        # best-looking round. Exp B never selects (always last round).
                        use_select = is_a and alpha != neutral_alpha

                        steer_fn = make_steer_fn(method, alpha, sae=sae, feat_idxs=feat_idxs, v=diffmean_vec,
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
                                    "round": round_idx, "gravy_current": gravy_score(cur_seq)})
                                roundlog_file.flush()

                        seq = masked_gibbs_generate(
                            steerer, init_tokens, steer_fn, correct_fn, args.mask_fraction,
                            args.n_rounds, args.stabilize_tol, args.sample_mode,
                            args.temperature, g, device, diag_fn=diag_fn, select_fn=select_fn,
                            log_fn=log_fn)

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
                            plddt = fold_plddt(seq, device) if args.fold else None
                            row = {"sample_id": sample_id, "alpha": alpha, "corrected": corrected,
                                  "length": length, "motif_hit": motif_hit,
                                  "concept_count": concept_count, "concept_count_loose": concept_count_loose,
                                  "naturalness_score": pppl, "diffusion_loss": dloss,
                                  "max_homopolymer_run": max_homopolymer_run(seq),
                                  "unique_aa_count": len(set(seq)),
                                  "target_feature_activation_mean": feat_act_mean,
                                  "target_feature_activation_max": feat_act_max,
                                  "target_feature_peak_residue": peak_residue,
                                  "plddt": plddt, "sequence": seq}
                            concept_score = float(motif_hit)
                            log_line = (f"[exp-a] alpha={alpha} corrected={corrected} sample={i} "
                                       f"motif_hit={motif_hit} count={concept_count}/{concept_count_loose} "
                                       f"pppl={pppl:.2f} dloss={dloss:.4f} plddt={plddt}")
                        else:
                            gravy_final = gravy_score(seq)
                            delta_gravy = gravy_final - gravy_init
                            proj_mean, proj_max = target_signal(method, acts_final, device, v=diffmean_vec)
                            row = {"sample_id": sample_id, "init_sequence_id": init_sequence_id,
                                  "alpha": alpha, "corrected": corrected, "length": length,
                                  "gravy_init": gravy_init, "gravy_final": gravy_final,
                                  "delta_gravy": delta_gravy, "naturalness_score": pppl,
                                  "diffusion_loss": dloss, "max_homopolymer_run": max_homopolymer_run(seq),
                                  "unique_aa_count": len(set(seq)),
                                  "target_projection_mean": proj_mean, "target_projection_max": proj_max,
                                  "sequence": seq}
                            concept_score = delta_gravy
                            log_line = (f"[exp-b] alpha={alpha} corrected={corrected} "
                                       f"init_id={init_sequence_id} gravy {gravy_init:.3f}->"
                                       f"{gravy_final:.3f} (d={delta_gravy:+.3f}) pppl={pppl:.2f} "
                                       f"dloss={dloss:.4f}")

                        w.writerow(row)
                        fcsv.flush()
                        rows_for_analysis.append({**row, "concept_score": concept_score,
                                                  "naturalness_score": pppl, "diffusion_loss": dloss})
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
                  "(unique_aa<=8 or max_homopolymer_run>=8) from Part 3 analysis — still present in "
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
            pareto_exp_b(rows_for_analysis, out_dir, run)
            regimes = three_regime_breakdown(rows_for_analysis, [r for r in rows_for_analysis if not r["corrected"]])
            wstats = paired_wilcoxon_by_alpha(rows_for_analysis, alphas)
            run.log_summary(paired_wilcoxon=wstats)
            for a, s in wstats.items():
                print(f"[exp-b] paired Wilcoxon alpha={a}: n={s['n_pairs']} mean_delta_gravy="
                     f"{s['mean_delta']:+.4f} p={s['wilcoxon_p']:.4f}")
        run.log_summary(three_regime=regimes)
        print(f"[steering] wrote {csv_path} + pareto.png")

    print(f"[steering] done. outputs under {out_dir}")


if __name__ == "__main__":
    main()
