"""Streaming producer-consumer activation buffer (PLAN §5/§6.1) — W2/cluster stub.

At cluster scale (100M+ activations) pre-caching to disk blows up (~64-190 GB), so GLP
uses a fixed-size buffer: a producer thread caches ESM activations, flushed on
consumption by the trainer (PLAN §6.1). The Mac PoC uses `activation_store` (precache)
instead, so this is intentionally deferred to W2. Documented here to keep the PLAN §5
skeleton complete.
"""

from __future__ import annotations


class ProducerConsumerBuffer:  # pragma: no cover - W2 cluster feature
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Streaming producer-consumer is a W2/cluster feature (PLAN §6.1). "
            "The Mac PoC uses glp_plm.data.activation_store (precache)."
        )
