"""The three probing baselines pinned in PLAN §6.5 (R6), beside GLP meta-neurons.

  sae        InterPLM released SAE features at layer 4      (10,240 dims)
  raw_layer  the residual stream itself (GLP's/SAE's input)    (320 dims)
  raw_mlp    ESM-2's own FFN intermediate at layer 4         (1,280 dims)

`⚠ FAIRNESS` (PLAN §6.5): every method gets the identical candidate budget and
train/val/test discipline downstream — these functions only produce the banks.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch


# --------------------------------------------------------------------------- #
#  SAE features
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sae_bank(sae, x: torch.Tensor, batch_size: int = 8192,
             normalize_features: bool = True) -> torch.Tensor:
    """InterPLM SAE features for RAW (un-standardized) activations `x` -> (N, dict_size).

    Uses `encode_feat_subset` over all features — the path InterPLM's own analysis
    uses (it applies the SAE's internal √d input normalisation, unlike bare
    `encode`). `normalize_features` divides by the released per-feature max
    (the `ae_normalized.pt` convention).
    """
    feats = list(range(sae.dict_size))
    dev = next(sae.parameters()).device      # SAE may sit on mps/cuda; inputs are CPU
    out = []
    for i in range(0, x.shape[0], batch_size):
        chunk = x[i:i + batch_size].float().to(dev)
        out.append(sae.encode_feat_subset(chunk, feats,
                                          normalize_features=normalize_features).detach().cpu())
    return torch.cat(out, dim=0)


# --------------------------------------------------------------------------- #
#  Raw layer output
# --------------------------------------------------------------------------- #
def raw_layer_bank(x: torch.Tensor) -> torch.Tensor:
    """The residual stream itself (what both GLP and the SAE consume)."""
    return x.float()


# --------------------------------------------------------------------------- #
#  Raw ESM MLP neuron (FFN intermediate)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def raw_mlp_bank(sequences: List[str], layer: int = 4,
                 model_name: str = "facebook/esm2_t6_8M_UR50D",
                 device: str = "cpu", batch_size: int = 8) -> torch.Tensor:
    """ESM-2 FFN intermediate (post-GELU, pre-output-projection) at `layer`.

    Row order matches the activation store: sequences concatenated in order, with
    <cls>/<eos>/pad dropped — the same slicing InterPLM's embedder uses
    (`esm.py:181-186`), so rows align with the per-AA labels.

    Indexing note: InterPLM "layer L" = output of block L (1-indexed) =
    `encoder.layer[L-1]` in the 0-indexed HF module list, so we hook
    `encoder.layer[L-1].intermediate`.
    """
    from interplm.embedders import get_embedder

    embedder = get_embedder("esm", model_name=model_name, device=device)
    model, tok = embedder.model, embedder.tokenizer

    encoder = getattr(model, "encoder", None) or model.esm.encoder
    target = encoder.layer[layer - 1].intermediate

    captured = {}

    def hook(_mod, _inp, out):
        captured["ffn"] = out.detach()

    handle = target.register_forward_hook(hook)
    try:
        rows = []
        for i in range(0, len(sequences), batch_size):
            batch = [embedder.preprocess_sequence(s) for s in sequences[i:i + batch_size]]
            inputs = tok(batch, return_tensors="pt", padding=True, truncation=True,
                         max_length=embedder.max_length)
            inputs = {k: v.to(embedder.device) for k, v in inputs.items()}
            model(**inputs)
            ffn = captured["ffn"].cpu()          # (B, T, d_ff)
            for j, seq in enumerate(batch):
                rows.append(ffn[j, 1:len(seq) + 1, :])
        return torch.cat(rows, dim=0).float()
    finally:
        handle.remove()


# --------------------------------------------------------------------------- #
METHODS = ["glp", "sae", "raw_layer", "raw_mlp"]
