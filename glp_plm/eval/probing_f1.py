"""InterPLM-native domain-adjusted F1 (PLAN §6.5 secondary metric, D5).

This is the SAE's *home-turf* metric: precision per amino acid, recall **per domain**,
best over a threshold sweep. Definitions match InterPLM exactly
(`interplm/analysis/concepts/calculate_f1.py:64-86` and
`compare_activations.py:106-136`):

    precision        = tp / (tp + fp)                       # per amino acid
    recall           = tp / n_positive_AAs                  # AA-level concepts
    recall_per_domain= (# distinct domain instances with >=1 activated residue)
                       / (# domain instances)               # domain-level concepts
    F1               = harmonic mean(precision, recall[_per_domain])

Two things differ from the AUC path and matter:

1. **No balanced subsampling.** F1 is prevalence-sensitive — precision depends on the
   real positive/negative ratio — so we evaluate on the FULL split at natural
   prevalence, streaming in chunks. (AUC is rank-based and prevalence-invariant, so
   subsampling there is harmless.)
2. **Rate-matched, two-sided operating points** (`⚠ FAIRNESS`, deviation from D5 —
   state it in the paper). InterPLM sweeps thresholds as a *fraction of each
   feature's max activation*, which is calibrated for sparse non-negative ReLU SAE
   features. Applied to dense signed activations that grid degenerates: measured on
   this eval set, `thr=0.0` fires on 56% of residues and `thr=0.5` on 0.07%, with
   nothing usable between, and 44% of values are negative so a one-sided `f > thr`
   cannot see concepts encoded in the negative direction. We therefore choose each
   operating point by **activation rate** — the per-feature quantile that fires on
   the top r fraction of residues — applied identically to every method, and we
   allow either **sign** of the feature. This keeps the spirit of the original
   ("sweep operating points, keep the best F1") while making representations of
   different sparsity and sign comparable.

Selection stays honest: (feature, sign, rate) is chosen on VALID — including the
threshold *values*, which are estimated on valid and reused verbatim on test — and
the winner is re-scored on TEST.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

# Operating points as ACTIVATION RATES (fraction of residues a feature fires on).
# Replaces InterPLM's fraction-of-max grid — see the module docstring.
RATES = [0.0005, 0.002, 0.01, 0.05]


def is_aa_level(concept: str) -> bool:
    """Per-residue concepts use plain recall; domain concepts use recall-per-domain."""
    from interplm.analysis.concepts.concept_constants import is_aa_level_concept

    return bool(is_aa_level_concept(concept))


def f1_score(precision: float, recall: float) -> float:
    from interplm.analysis.concepts.calculate_f1 import calculate_f1

    return float(calculate_f1(precision, recall))


@dataclass
class F1Result:
    concept: str
    method: str
    val_f1: float
    test_f1: float
    precision: float
    recall: float
    threshold: float          # the VALUE threshold, estimated on valid
    rate: float               # target activation rate that produced it
    sign: int                 # +1 or -1 (which direction of the feature was used)
    feature_index: int
    aa_level: bool
    n_pos_aa_test: int
    n_domains_test: int
    u: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


def run_concept_f1(bank, n_valid: int, n_test: int, y_val: np.ndarray, y_test: np.ndarray,
                   cand: np.ndarray, concept: str, method: str,
                   chunk: int = 10_000, u: Optional[float] = None) -> F1Result:
    """Full F1 protocol for one (concept, method).

    `bank(split, rows, cols) -> (len(rows), len(cols)) float32`, split in {"valid","test"}.
    Selection of (feature, threshold) happens on VALID; the winner is re-scored on TEST.
    """
    aa = is_aa_level(concept)

    # each candidate is evaluated in BOTH directions: cols [cand, cand], signs [+1..., -1...]
    cols = np.concatenate([cand, cand])
    signs = np.concatenate([np.ones(len(cand)), -np.ones(len(cand))])

    def vbank(r):
        return bank("valid", r, cols)

    thr = feature_thresholds(vbank, n_valid, signs)               # (K, T), from VALID only
    tp, fp, dom = accumulate_counts(vbank, y_val, thr, signs, aa, chunk)
    n_pos_v = int((y_val > 0).sum())
    n_dom_v = int(len(np.unique(y_val[y_val > 0])))
    _, _, f1_grid = score_from_counts(tp, fp, dom, n_pos_v, n_dom_v, aa)

    k, ti = np.unravel_index(int(np.argmax(f1_grid)), f1_grid.shape)
    best_feat, best_sign = int(cols[k]), int(signs[k])

    # re-score the single winning (feature, sign) on test, reusing the VALID threshold
    one = np.array([best_feat])
    one_sign = np.array([float(best_sign)])
    one_thr = thr[k:k + 1, ti:ti + 1]
    tp2, fp2, dom2 = accumulate_counts(lambda r: bank("test", r, one), y_test,
                                       one_thr, one_sign, aa, chunk)
    n_pos_t = int((y_test > 0).sum())
    n_dom_t = int(len(np.unique(y_test[y_test > 0])))
    p2, r2, f2 = score_from_counts(tp2, fp2, dom2, n_pos_t, n_dom_t, aa)

    return F1Result(
        concept=concept, method=method,
        val_f1=float(f1_grid[k, ti]), test_f1=float(f2[0, 0]),
        precision=float(p2[0, 0]), recall=float(r2[0, 0]),
        threshold=float(thr[k, ti]), rate=RATES[ti], sign=best_sign,
        feature_index=best_feat, aa_level=aa,
        n_pos_aa_test=n_pos_t, n_domains_test=n_dom_t, u=u,
    )


def feature_thresholds(bank_fn: Callable[[np.ndarray], np.ndarray], n_rows: int,
                       signs: np.ndarray, sample: int = 100_000, seed: int = 0) -> np.ndarray:
    """Per-feature VALUE thresholds hitting each target activation rate -> (K, T).

    Estimated from a random subsample of the split (a quantile needs no more), and
    always computed on VALID so test never informs the operating point.
    """
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(n_rows, min(sample, n_rows), replace=False))
    f = bank_fn(rows) * signs[None, :]
    # quantile at 1-r gives the value that the top r fraction of residues exceed
    return np.quantile(f, [1.0 - r for r in RATES], axis=0).T.astype(np.float64)


def accumulate_counts(bank_fn: Callable[[np.ndarray], np.ndarray], y_int: np.ndarray,
                      thr: np.ndarray, signs: np.ndarray, aa_level: bool,
                      chunk: int = 10_000):
    """Stream a split; return tp, fp (K,T) and domains-hit sets per (feature, rate).

    `thr` is (K, T) value thresholds; `signs` is (K,) in {+1,-1}. `y_int` holds the
    integer domain-instance index per residue (0 = negative).
    """
    n, K, T = len(y_int), thr.shape[0], thr.shape[1]
    tp = np.zeros((K, T), dtype=np.int64)
    fp = np.zeros((K, T), dtype=np.int64)
    dom_hits: List[List[set]] = [[set() for _ in range(T)] for _ in range(K)]

    for s0 in range(0, n, chunk):
        rows = np.arange(s0, min(s0 + chunk, n))
        f = bank_fn(rows) * signs[None, :]         # (chunk, K), sign-corrected
        yc = y_int[rows]
        pos = yc > 0
        for ti in range(T):
            act = f > thr[None, :, ti]             # (chunk, K)
            tp[:, ti] += (act & pos[:, None]).sum(0)
            fp[:, ti] += (act & ~pos[:, None]).sum(0)
            if not aa_level:
                for k in np.flatnonzero(act.any(0)):
                    dom_hits[k][ti].update(np.unique(yc[act[:, k] & pos]).tolist())
    return tp, fp, dom_hits


def score_from_counts(tp, fp, dom_hits, n_pos_aa: int, n_domains: int, aa_level: bool):
    """precision / recall / F1 grids (K, T) using InterPLM's definitions."""
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
        recall_aa = tp / max(n_pos_aa, 1)
    if aa_level:
        recall = recall_aa
    else:
        recall = np.zeros_like(recall_aa)
        for k in range(recall.shape[0]):
            for t in range(recall.shape[1]):
                recall[k, t] = len(dom_hits[k][t]) / max(n_domains, 1)
    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / np.maximum(denom, 1e-12), 0.0)
    return precision, recall, f1
