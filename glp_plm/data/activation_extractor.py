"""Thin wrapper over InterPLM's ESM embedder — the single hook-parity guard.

Key facts (see docs/INTERPLM_MAP.md):
  * The real API is the class `interplm.embedders.esm.ESM` (factory
    `interplm.embedders.get_embedder("esm", model_name=...)`), NOT the
    plan's `interplm.esm.embed`.
  * `embed_single_sequence(sequence, layer)` and
    `extract_embeddings_multiple_layers(seqs, layers, ...)` ALREADY drop
    <cls>/<eos>/<pad> and return `(n_tokens, d)`. We therefore do NOT
    strip again — this wrapper returns the embedder output verbatim, which
    is what guarantees GLP trains on byte-identical activations to what the
    SAE saw (residual stream after block `layer`, cls+eos dropped).

We deliberately call the InterPLM helper rather than reimplementing extraction
(PLAN R5, the highest-value guard). The hook-parity test asserts this wrapper's
output equals `ESM.embed_single_sequence` exactly.
"""

from __future__ import annotations

from typing import List

import numpy as np


class ActivationExtractor:
    def __init__(self, model_name: str = "facebook/esm2_t6_8M_UR50D", layer: int = 4, device=None):
        # Imported lazily so the package/config/model/tests import without InterPLM
        # + torch + transformers installed (they're only needed to actually extract).
        from interplm.embedders import get_embedder

        self.model_name = model_name
        self.layer = layer
        self._embedder = get_embedder("esm", model_name=model_name, device=device)

    def extract_single(self, sequence: str) -> np.ndarray:
        """Per-residue activations for one sequence -> (seq_len, d), cls/eos already dropped."""
        return self._embedder.embed_single_sequence(sequence, self.layer)

    def extract_many(self, sequences: List[str], batch_size: int = 8):
        """Flat activations for a list of sequences -> (total_tokens, d), cls/eos dropped.

        Returns InterPLM's `extract_embeddings_multiple_layers` output for `self.layer`.
        """
        out = self._embedder.extract_embeddings_multiple_layers(
            sequences, [self.layer], batch_size=batch_size, shuffle=False
        )
        return out[self.layer]
