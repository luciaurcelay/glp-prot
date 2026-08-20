"""Confidence-frozen remasking (--mask-selection confidence) in run_steering's loop.

Guards Family A of the decoding-loop foldability fix: each round must remask the k
LOWEST model-endorsement positions and leave the confident ones untouched. Uses a fake
steerer with hand-crafted logits so the intended positions are unambiguous, so no ESM
download / GPU is needed (mirrors how test_hook_parity keeps the live path separate)."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "scripts", ROOT / "InterPLM"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_steering as rs  # noqa: E402


class _FakeTok:
    mask_token_id = 20
    cls_token_id = 21
    eos_token_id = 22

    @staticmethod
    def convert_ids_to_tokens(i):
        # ids 0..19 -> letters A..T (arbitrary but stable); anything else -> 'X'
        return chr(ord("A") + i) if 0 <= i < 20 else "X"


class _FakeSteerer:
    """Returns the SAME logits/hidden regardless of input tokens, so confidence is a
    fixed, hand-crafted per-position quantity and the resample target is deterministic."""

    def __init__(self, logits, hidden):
        self.tokenizer = _FakeTok()
        self.valid_aa_ids = torch.arange(20)
        self._logits = logits
        self._hidden = hidden

    def forward(self, token_ids, attn_mask, override=None):
        return self._logits, self._hidden


def test_confidence_remasks_the_least_endorsed_positions():
    L, vocab, d = 6, 23, 8
    low_conf = {2, 4}          # the two positions the model should distrust and remask
    resample_to = 2            # id the masked positions must be resampled to (letter 'C')

    # current interior tokens all id 0 ('A'); cls/eos in the flanks
    tokens = torch.zeros(1, L + 2, dtype=torch.long)
    tokens[0, 0] = _FakeTok.cls_token_id
    tokens[0, -1] = _FakeTok.eos_token_id

    # logits: confident positions put all mass on the current token (id 0); the two
    # low-confidence positions starve id 0 and favour `resample_to`, so (a) their
    # p(current) is ~0 -> selected, and (b) argmax over valid ids -> resample_to.
    logits = torch.full((1, L + 2, vocab), -10.0)
    for j in range(1, L + 1):
        if (j - 1) in low_conf:
            logits[0, j, resample_to] = 10.0
        else:
            logits[0, j, 0] = 10.0
    hidden = torch.zeros(1, L + 2, d)

    steerer = _FakeSteerer(logits, hidden)
    out = rs.masked_gibbs_generate(
        steerer, tokens.clone(),
        steer_fn=lambda a: a, correct_fn=lambda a: a,
        mask_fraction=2 / L,           # -> k = round(2) = 2
        n_rounds=1, stabilize_tol=0.0,
        sample_mode="argmax", temperature=1.0,
        generator=torch.Generator().manual_seed(0), device="cpu",
        selection="confidence",
    )

    assert len(out) == L
    # exactly the two low-confidence positions changed, to 'C'; the rest stayed 'A'
    for i, ch in enumerate(out):
        if i in low_conf:
            assert ch == "C", f"pos {i} should be resampled, got {ch!r}"
        else:
            assert ch == "A", f"pos {i} was frozen but changed to {ch!r}"


class _BeamFakeSteerer:
    """Batch-shaped fake: every position's marginal over the 20 AAs is a fixed preference
    vector `pref`, independent of context/override, so the steered PLL landscape is known in
    closed form and the beam's choices are checkable by hand."""

    def __init__(self, pref, vocab=23, d=4):
        self.tokenizer = _FakeTok()
        self.valid_aa_ids = torch.arange(20)
        self.pref = pref
        self.vocab = vocab
        self.d = d

    def forward(self, token_ids, attn_mask, override=None):
        B, S = token_ids.shape
        logits = torch.full((B, S, self.vocab), -30.0)
        logits[:, :, :20] = self.pref                    # same marginal at every position
        hidden = torch.zeros(B, S, self.d)
        return logits, hidden


def test_marginal_matrix_and_free_neighbor_pll_identity():
    import torch.nn.functional as F
    L = 5
    pref = torch.arange(20).float()                      # AA idx 19 most preferred, 0 least
    steerer = _BeamFakeSteerer(pref)
    tokens = torch.zeros(1, L + 2, dtype=torch.long)     # interior all 'A' (idx 0)
    tokens[0, 0], tokens[0, -1] = _FakeTok.cls_token_id, _FakeTok.eos_token_id
    A = rs._steered_marginal_matrix(steerer, tokens, lambda a: a, lambda a: a, "cpu")
    assert A.shape == (L, 20)
    expected_row = F.log_softmax(pref, dim=-1)
    assert torch.allclose(A[0], expected_row, atol=1e-5)
    # free-neighbor identity: PLL of the neighbour that flips pos 2 to AA idx 7, computed the
    # cheap way (base - A[k,cur] + A[k,new]) must equal a from-scratch recompute
    base_pll = float(A[torch.arange(L), torch.zeros(L, dtype=torch.long)].sum())
    cheap = base_pll - float(A[2, 0]) + float(A[2, 7])
    scratch = float(sum(A[i, 7 if i == 2 else 0] for i in range(L)))
    assert abs(cheap - scratch) < 1e-5


def test_beam_search_makes_n_edits_toward_preferred_residue():
    L, E = 6, 3
    pref = torch.arange(20).float()                      # best AA is idx 19 -> letter 'T'
    steerer = _BeamFakeSteerer(pref)
    tokens = torch.zeros(1, L + 2, dtype=torch.long)     # interior all 'A'
    tokens[0, 0], tokens[0, -1] = _FakeTok.cls_token_id, _FakeTok.eos_token_id
    out = rs.steered_beam_search_generate(
        steerer, tokens.clone(), lambda a: a, lambda a: a,
        beam_width=4, n_edits=E, temperature=0.01,       # near-deterministic (tiny Gumbel weight)
        generator=torch.Generator().manual_seed(0), device="cpu")
    assert len(out) == L
    # exactly E residues moved to the most-preferred AA ('T'), the rest stayed 'A'
    assert out.count("T") == E
    assert out.count("A") == L - E
    assert set(out) <= {"A", "T"}


def test_confidence_budget_matches_random_k():
    # with mask_fraction 0.5 and L=6 the edit budget is 3, same rule as the random loop
    L, vocab, d = 6, 23, 8
    tokens = torch.zeros(1, L + 2, dtype=torch.long)
    tokens[0, 0], tokens[0, -1] = _FakeTok.cls_token_id, _FakeTok.eos_token_id
    # strictly increasing endorsement so the three lowest are positions 0,1,2
    logits = torch.full((1, L + 2, vocab), -10.0)
    for j in range(1, L + 1):
        logits[0, j, 0] = float(j)      # higher index -> more endorsed current token
    steerer = _FakeSteerer(logits, torch.zeros(1, L + 2, d))
    out = rs.masked_gibbs_generate(
        steerer, tokens.clone(), lambda a: a, lambda a: a,
        mask_fraction=0.5, n_rounds=1, stabilize_tol=0.0,
        sample_mode="argmax", temperature=1.0,
        generator=torch.Generator().manual_seed(0), device="cpu",
        selection="confidence",
    )
    # argmax of every position is id 0 ('A'), so resampling the 3 least-endorsed is a
    # no-op on the letters; the test's point is that it runs and stays length-stable
    # with a k=3 budget (no crash, no frozen/altered flank leakage).
    assert out == "AAAAAA"
