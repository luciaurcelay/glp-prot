"""Hook-parity guard (PLAN §5, R5 — the single most important guard in the repo).

Assert our ActivationExtractor output equals InterPLM's own embedder for a fixed
sequence at layer 4. docs/INTERPLM_MAP.md: InterPLM's embedder ALREADY drops <cls>/<eos>,
so parity is EXACT (not "up to the cls/eos drop") — our wrapper adds no processing.

This test needs `interplm` + torch/transformers importable AND the ESM-2-8M weights
available (HF download or cache). This session does no downloads, so the test SKIPS
cleanly when the dependency or weights are absent; it becomes live in W1 once the
model is fetched. When it runs, it must pass exactly.
"""

import numpy as np
import pytest

# A fixed, short, valid protein sequence (canonical 20 AAs only).
FIXED_SEQ = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"

# (model, layer, dim). ESM-2-8M/layer-4 is the canonical guard; the 650M rows cover the
# candidate hooked layers for the scaling experiment (docs/plan_650m_scaling.md). Each row
# SKIPS cleanly if that model's weights are absent, so 650M rows go live only where the
# ~2.5 GB weights are present (locally after the stage-0 download, and on MN5).
CASES = [
    ("facebook/esm2_t6_8M_UR50D", 4, 320),
    ("facebook/esm2_t33_650M_UR50D", 18, 1280),
    ("facebook/esm2_t33_650M_UR50D", 24, 1280),
]


def _load_interplm_embedder(model):
    """Import InterPLM's ESM embedder and instantiate it, or skip."""
    pytest.importorskip("interplm", reason="InterPLM not installed")
    pytest.importorskip("torch", reason="torch not installed")
    pytest.importorskip("transformers", reason="transformers not installed")
    from interplm.embedders import get_embedder

    try:
        return get_embedder("esm", model_name=model, device="cpu")
    except Exception as e:  # network/weights unavailable in this environment
        pytest.skip(f"{model} weights unavailable ({type(e).__name__}: {e})")


@pytest.mark.parametrize("model,layer,dim", CASES)
def test_extractor_matches_interplm_embedder(model, layer, dim):
    embedder = _load_interplm_embedder(model)

    # Ground truth: InterPLM's own single-sequence helper (cls/eos already dropped).
    ref = np.asarray(embedder.embed_single_sequence(FIXED_SEQ, layer))

    # Our wrapper (must reuse the same embedder path, no extra processing).
    from glp_plm.data.activation_extractor import ActivationExtractor

    extractor = ActivationExtractor(model_name=model, layer=layer, device="cpu")
    ours = np.asarray(extractor.extract_single(FIXED_SEQ))

    assert ours.shape == ref.shape == (len(FIXED_SEQ), dim), (
        f"expected ({len(FIXED_SEQ)}, {dim}), got ours={ours.shape} ref={ref.shape}"
    )
    assert np.allclose(ours, ref, atol=1e-6), (
        f"hook parity broken: max abs diff {np.abs(ours - ref).max():.2e}"
    )


@pytest.mark.parametrize("model,layer,dim", CASES)
def test_many_matches_single_concatenation(model, layer, dim):
    """extract_many over one sequence equals extract_single (both cls/eos-dropped, flat)."""
    _load_interplm_embedder(model)     # skips the test if deps/weights are absent
    from glp_plm.data.activation_extractor import ActivationExtractor

    extractor = ActivationExtractor(model_name=model, layer=layer, device="cpu")
    single = np.asarray(extractor.extract_single(FIXED_SEQ))
    many = np.asarray(extractor.extract_many([FIXED_SEQ], batch_size=1))
    assert many.shape == single.shape
    assert np.allclose(many, single, atol=1e-6)
