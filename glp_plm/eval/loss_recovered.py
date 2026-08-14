"""SAE %Loss-Recovered (Gate G2, PLAN §6.4/§9): adapts
`InterPLM/interplm/train/fidelity.py` — reuses its
`interplm.sae.intervention.get_esm_output_with_intervention` directly, but
`calculate_cross_entropy`/`calculate_loss_recovered` are copied in verbatim rather than
imported: `fidelity.py`'s module-level `from esm import pretrained` pulls in the old
`fair-esm` package, which is not one of this project's dependencies (deliberately —
`requirements.txt` and every other InterPLM-reuse point in this codebase go through HF
`transformers`' `EsmForMaskedLM`, never `fair-esm`) and isn't needed for these two pure
functions. Also skips InterPLM's `EvaluationManager`/`ESMFidelityConfig` scaffolding
(built for its own training-loop callback use, not a one-shot eval) in favor of a plain
function this project's scripts call directly, with its own tokenization
(`AutoTokenizer` + `EsmForMaskedLM`, the same pattern `scripts/run_steering.py`'s
`ESMSteerer` already uses for nnsight 0.7 compat).

%LR = 1 - (CE_sae_recon - CE_identity) / (CE_zero_ablation - CE_identity), clipped to
[0, 100]%. 100% = the SAE's reconstruction is as good as not intervening at all; 0% =
as bad as zeroing the hook-layer activation out entirely.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm


def calculate_cross_entropy(model_output, batch_tokens, batch_attn_mask):
    """Verbatim from InterPLM/interplm/train/fidelity.py — cross entropy per sequence
    in batch, excluding start/end tokens."""
    losses = []
    for j, mask in enumerate(batch_attn_mask):
        length = mask.sum()
        seq_logits = model_output[j, 1 : length - 1]
        seq_tokens = batch_tokens[j, 1 : length - 1]
        loss = torch.nn.functional.cross_entropy(seq_logits, seq_tokens)
        losses.append(loss.item())
    return losses


def calculate_loss_recovered(ce_autoencoder, ce_identity, ce_zero_ablation):
    """Verbatim from InterPLM/interplm/train/fidelity.py."""
    numerator = ce_autoencoder - ce_identity
    denominator = ce_zero_ablation - ce_identity
    if np.isclose(denominator, 0):
        return 0.0
    loss_recovered = 1 - (numerator / denominator)
    loss_recovered = np.clip(loss_recovered, 0, 1)
    return loss_recovered * 100


def _stub_ipython_if_broken() -> None:
    """MN5's PYTHON/3.11.5 module ships a `_sqlite3` extension built against
    `sqlite3_deserialize`, which RHEL9's system libsqlite3 doesn't export, so plain
    `import sqlite3` fails there; `nnsight` only imports IPython for `get_ipython()`
    (always False outside a notebook), and the real IPython pulls in sqlite3 via its
    `history` submodule. Stub just that one function instead. Mirrors
    `scripts/run_steering.py`'s `_stub_ipython_if_broken` (same root cause, same fix;
    duplicated rather than shared to keep this module import-light)."""
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


_stub_ipython_if_broken()
from interplm.sae.intervention import get_esm_output_with_intervention  # noqa: E402


def _tokenize_batches(tokenizer, sequences: List[str], batch_size: int):
    batches = []
    for i in range(0, len(sequences), batch_size):
        chunk = sequences[i:i + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=True)
        batches.append((enc["input_ids"], enc["attention_mask"]))
    return batches


def compute_pct_loss_recovered(sae, sequences: List[str], model_name: str, layer: int,
                                device: str, batch_size: int = 8,
                                nn_model: Optional[object] = None,
                                model: Optional[object] = None,
                                tokenizer: Optional[object] = None) -> dict:
    """%Loss-Recovered for `sae` at `layer` of `model_name`, on `sequences`.

    `model`/`tokenizer`/`nn_model` may be passed in (e.g. reuse `ESMSteerer`'s) to avoid
    loading ESM twice in the same process; otherwise loaded here.
    """
    if model is None or tokenizer is None:
        from transformers import AutoTokenizer, EsmForMaskedLM
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = EsmForMaskedLM.from_pretrained(model_name).to(device).eval()
    if nn_model is None:
        from nnsight import NNsight
        nn_model = NNsight(model)

    batches = _tokenize_batches(tokenizer, sequences, batch_size)

    orig_losses, zero_losses, sae_losses = [], [], []
    for batch_tokens, batch_attn_mask in tqdm(batches, desc="[loss_recovered]"):
        batch_tokens = batch_tokens.to(device)
        batch_attn_mask = batch_attn_mask.to(device)

        orig_logits, orig_hidden = get_esm_output_with_intervention(
            model, nn_model, batch_tokens, batch_attn_mask, layer)
        orig_losses.extend(calculate_cross_entropy(orig_logits, batch_tokens, batch_attn_mask))

        zero_logits, _ = get_esm_output_with_intervention(
            model, nn_model, batch_tokens, batch_attn_mask, layer,
            torch.zeros_like(orig_hidden))
        zero_losses.extend(calculate_cross_entropy(zero_logits, batch_tokens, batch_attn_mask))

        # SAE operates on standardized/normalized activations internally (same
        # `_normalize_input_and_get_norms` / `_unnormalize_output` convention
        # `scripts/run_steering.py`'s `sae_clamp` uses) — reconstruct in raw
        # activation space so the recon-error add-back convention isn't needed here
        # (unlike steering, we want the SAE's OWN reconstruction error included, not
        # cancelled — that's exactly what %LR is meant to penalize).
        x_norm, orig_norms = sae._normalize_input_and_get_norms(orig_hidden)
        recon = sae._unnormalize_output(sae.decode(sae.encode(x_norm)), orig_norms)
        sae_logits, _ = get_esm_output_with_intervention(
            model, nn_model, batch_tokens, batch_attn_mask, layer, recon)
        sae_losses.extend(calculate_cross_entropy(sae_logits, batch_tokens, batch_attn_mask))

    ce_identity = float(np.mean(orig_losses))
    ce_zero = float(np.mean(zero_losses))
    ce_sae = float(np.mean(sae_losses))
    pct_lr = calculate_loss_recovered(ce_autoencoder=ce_sae, ce_identity=ce_identity,
                                      ce_zero_ablation=ce_zero)
    return {"pct_loss_recovered": float(pct_lr), "ce_identity": ce_identity,
            "ce_zero_ablation": ce_zero, "ce_sae_recon": ce_sae, "n_sequences": len(sequences)}
