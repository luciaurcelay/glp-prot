"""1-D probing protocol — faithful port of GLP's released probe (PLAN §6.5).

Per concept and per method:
  1. filter to top-k=512 candidate features on TRAIN by |mean(F|y=1) − mean(F|y=0)|
     (generative_latent_prior/glp/script_probe.py:72)
  2. fit a 1-D logistic regression per candidate: StandardScaler +
     LogisticRegressionCV(Cs={1e-5…1e0}, cv=5, scoring='roc_auc', max_iter=1000, L2)
  3. `val_auc` = mean CV score at the selected C; select the candidate by val AUC
  4. report that candidate's TEST auc                       (never select on test)

Positions are per-amino-acid (PLAN §6.5 H4 — a deliberate divergence from GLP's
last-token probing, because Swiss-Prot concepts are per-residue).

`⚠ FAIRNESS`: identical k, identical subsampling, identical splits for all methods.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np

CS_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]


# --------------------------------------------------------------------------- #
#  fast per-feature AUC (rank-sum; handles the many exact ties in SAE features)
# --------------------------------------------------------------------------- #
def fast_auc_matrix(F: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """AUC of every feature against every label column. F:(N,D) Y:(N,C) -> (C,D).

    Rank-sum (Mann-Whitney) formulation with average ranks, so it is exact under
    ties — important because SAE features are mostly exact zeros.
    """
    from scipy.stats import rankdata

    ranks = rankdata(F, axis=0)                       # (N, D), average ranks
    Y = Y.astype(np.float64)
    n_pos = Y.sum(axis=0)                             # (C,)
    n_neg = F.shape[0] - n_pos
    rank_sum = Y.T @ ranks                            # (C, D)
    auc = (rank_sum - (n_pos * (n_pos + 1) / 2)[:, None]) / np.maximum(n_pos * n_neg, 1)[:, None]
    return auc


def topk_filter(F: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """GLP's candidate filter: top-k features by |mean_pos − mean_neg| on TRAIN."""
    if k >= F.shape[1]:
        return np.arange(F.shape[1])
    pos, neg = F[y == 1], F[y == 0]
    diff = np.abs(pos.mean(axis=0) - neg.mean(axis=0))
    return np.argsort(-diff)[:k]


# --------------------------------------------------------------------------- #
#  balanced subsampling (keeps the CV affordable and classes non-degenerate)
# --------------------------------------------------------------------------- #
def balanced_indices(y: np.ndarray, max_pos: int, neg_per_pos: float,
                     seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) > max_pos:
        pos = rng.choice(pos, max_pos, replace=False)
    n_neg = min(len(neg), int(round(len(pos) * neg_per_pos)))
    neg = rng.choice(neg, n_neg, replace=False)
    idx = np.concatenate([pos, neg])
    rng.shuffle(idx)
    return idx


# --------------------------------------------------------------------------- #
#  the probe
# --------------------------------------------------------------------------- #
@dataclass
class ProbeResult:
    concept: str
    method: str
    val_auc: float
    test_auc: float
    feature_index: int
    n_train: int
    n_test: int
    n_pos_train: int
    n_pos_test: int
    u: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


def _fit_one(x_tr, y_tr, x_te, y_te, seed: int):
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(Cs=CS_GRID, cv=5, scoring="roc_auc", max_iter=1000,
                             random_state=seed, penalty="l2", n_jobs=1),
    )
    pipe.fit(x_tr.reshape(-1, 1), y_tr)
    clf = pipe["logisticregressioncv"]
    best_c_idx = int(np.where(np.array(clf.Cs_) == clf.C_[0])[0][0])
    val_auc = float(clf.scores_[1].mean(axis=0)[best_c_idx])
    test_auc = float(roc_auc_score(y_te, pipe.predict_proba(x_te.reshape(-1, 1))[:, 1]))
    return val_auc, test_auc


def probe_concept(F_train: np.ndarray, y_train: np.ndarray,
                  F_test: np.ndarray, y_test: np.ndarray,
                  concept: str, method: str, topk: int = 512,
                  seed: int = 42, n_jobs: int = -1,
                  u: Optional[float] = None) -> ProbeResult:
    """Run the full protocol for one (concept, method) and return the winner."""
    from joblib import Parallel, delayed

    cand = topk_filter(F_train, y_train, topk)
    # max_nbytes=None: the per-candidate arrays are 1-D and tiny, so disable joblib's
    # spill-to-/tmp memmapping — with 100+ concepts it churns temp files and memory.
    scored = Parallel(n_jobs=n_jobs, max_nbytes=None)(
        delayed(_fit_one)(np.ascontiguousarray(F_train[:, c]), y_train,
                          np.ascontiguousarray(F_test[:, c]), y_test, seed) for c in cand
    )
    val_aucs = np.array([s[0] for s in scored])
    test_aucs = np.array([s[1] for s in scored])
    best = int(np.argmax(val_aucs))               # select on val, report test (D9)
    return ProbeResult(
        concept=concept, method=method,
        val_auc=float(val_aucs[best]), test_auc=float(test_aucs[best]),
        feature_index=int(cand[best]),
        n_train=int(len(y_train)), n_test=int(len(y_test)),
        n_pos_train=int(y_train.sum()), n_pos_test=int(y_test.sum()), u=u,
    )


def bootstrap_ci(values, n_resamples: int = 10000, confidence: float = 0.95, seed: int = 0):
    """Percentile bootstrap CI of the mean (GLP reports 95% CIs over tasks)."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return float(v.mean()) if len(v) else float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_resamples, len(v)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [(1 - confidence) / 2 * 100, (1 + confidence) / 2 * 100])
    return float(v.mean()), float(lo), float(hi)
