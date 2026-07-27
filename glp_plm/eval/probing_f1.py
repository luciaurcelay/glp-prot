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
2. **Features are scaled to [0,1] by their per-feature max on the VALID split**, so a
   percentage threshold means the same thing for every method. This mirrors
   InterPLM's own normalization step (`interplm/sae/normalize.py`), which is why the
   released weights are `ae_normalized.pt`.

Selection stays honest: pick (feature, threshold) on valid, report F1 on test.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

# InterPLM's threshold sweep (PLAN §2.2).
THRESHOLDS = [0.0, 0.15, 0.5, 0.6, 0.8]


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
    threshold: float
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

    scale = feature_max(lambda r: bank("valid", r, cand), n_valid, chunk=chunk)
    tp, fp, dom = accumulate_counts(lambda r: bank("valid", r, cand), y_val, scale, aa, chunk)
    n_pos_v = int((y_val > 0).sum())
    n_dom_v = int(len(np.unique(y_val[y_val > 0])))
    _, _, f1_grid = score_from_counts(tp, fp, dom, n_pos_v, n_dom_v, aa)

    k, ti = np.unravel_index(int(np.argmax(f1_grid)), f1_grid.shape)
    best_feat, best_thr = int(cand[k]), THRESHOLDS[ti]

    # re-score the single winning feature on the held-out split
    one = np.array([best_feat])
    s1 = np.array([scale[k]])
    tp2, fp2, dom2 = accumulate_counts(lambda r: bank("test", r, one), y_test, s1, aa, chunk)
    n_pos_t = int((y_test > 0).sum())
    n_dom_t = int(len(np.unique(y_test[y_test > 0])))
    p2, r2, f2 = score_from_counts(tp2, fp2, dom2, n_pos_t, n_dom_t, aa)

    return F1Result(
        concept=concept, method=method,
        val_f1=float(f1_grid[k, ti]), test_f1=float(f2[0, ti]),
        precision=float(p2[0, ti]), recall=float(r2[0, ti]),
        threshold=best_thr, feature_index=best_feat, aa_level=aa,
        n_pos_aa_test=n_pos_t, n_domains_test=n_dom_t, u=u,
    )


def feature_max(bank_fn: Callable[[np.ndarray], np.ndarray], n_rows: int,
                chunk: int = 50_000) -> np.ndarray:
    """Per-feature max over a split (the [0,1] normalizer), streamed."""
    mx = None
    for s in range(0, n_rows, chunk):
        f = np.abs(bank_fn(np.arange(s, min(s + chunk, n_rows))))
        m = f.max(axis=0)
        mx = m if mx is None else np.maximum(mx, m)
    return np.maximum(mx, 1e-12)


def accumulate_counts(bank_fn: Callable[[np.ndarray], np.ndarray], y_int: np.ndarray,
                      scale: np.ndarray, aa_level: bool, chunk: int = 50_000):
    """Stream a split; return tp, fp (K,T) and domains-hit sets per (feature, threshold).

    `y_int` holds the integer domain-instance index per residue (0 = negative).
    """
    n, K, T = len(y_int), len(scale), len(THRESHOLDS)
    tp = np.zeros((K, T), dtype=np.int64)
    fp = np.zeros((K, T), dtype=np.int64)
    dom_hits: List[List[set]] = [[set() for _ in range(T)] for _ in range(K)]

    for s in range(0, n, chunk):
        rows = np.arange(s, min(s + chunk, n))
        f = bank_fn(rows) / scale                      # -> [0,1]-ish
        yc = y_int[rows]
        pos = yc > 0
        for ti, thr in enumerate(THRESHOLDS):
            act = f > thr                              # (chunk, K)
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
