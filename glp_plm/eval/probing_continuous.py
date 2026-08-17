"""1-D continuous-property probing (GLProt probing spec Part 1/4, continuous axis).

Same protocol shape as `probing_1d.py`'s discrete AUC path — top-k candidate filter
on the pooled "valid" split, select by a validation metric, report on the held-out
"test" split, never select and report on the same split — but the label is a
continuous per-protein score (GRAVY, MW, pI, ...), so:
  * features are PER-PROTEIN pooled activations (`swissprot_labels.pool_rows`), not
    per-residue ones — continuous properties have no per-residue value;
  * candidates are ranked by |Pearson r| instead of |mean(pos) - mean(neg)|;
  * the fit is 1-D `LinearRegression` instead of `LogisticRegressionCV`;
  * the reported statistic is Spearman rho instead of ROC AUC. `val_spearman` comes
    from 5-fold out-of-fold predictions on "valid" (the CV-based analog of
    `probing_1d`'s `LogisticRegressionCV(cv=5)` val_auc), refit on all of "valid" for
    the final "test" prediction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class ContinuousProbeResult:
    property: str
    method: str
    val_spearman: float
    test_spearman: float
    feature_index: int
    n_train: int
    n_test: int
    u: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


def topk_filter_continuous(F: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Top-k candidates by |Pearson r| on the pooled "valid" split — the continuous
    analog of `probing_1d.topk_filter`'s mean-difference ranking (spec Part 4 step 1:
    a cheap ranking statistic, identical budget/procedure across all four
    representations)."""
    if k >= F.shape[1]:
        return np.arange(F.shape[1])
    yc = y - y.mean()
    Fc = F - F.mean(axis=0, keepdims=True)
    num = Fc.T @ yc
    den = np.sqrt((Fc ** 2).sum(axis=0) * (yc ** 2).sum()) + 1e-12
    r = num / den
    return np.argsort(-np.abs(r))[:k]


def _fit_one(x_val, y_val, x_te, y_te, seed: int):
    import warnings

    from scipy.stats import spearmanr
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_predict

    with warnings.catch_warnings(), np.errstate(divide="ignore", over="ignore",
                                                under="ignore", invalid="ignore"):
        warnings.simplefilter("ignore")
        oof = cross_val_predict(LinearRegression(), x_val.reshape(-1, 1), y_val,
                                cv=KFold(5, shuffle=True, random_state=seed))
        val_rho = spearmanr(oof, y_val).statistic

        lr = LinearRegression().fit(x_val.reshape(-1, 1), y_val)
        test_pred = lr.predict(x_te.reshape(-1, 1))
        test_rho = spearmanr(test_pred, y_te).statistic

    val_rho = float(val_rho) if np.isfinite(val_rho) else 0.0
    test_rho = float(test_rho) if np.isfinite(test_rho) else 0.0
    return val_rho, test_rho


def probe_property(F_val: np.ndarray, y_val: np.ndarray, F_test: np.ndarray, y_test: np.ndarray,
                   prop: str, method: str, topk: int = 512, seed: int = 42, n_jobs: int = -1,
                   u: Optional[float] = None) -> ContinuousProbeResult:
    """Full protocol for one (property, method): filter -> fit+select on valid ->
    report on test. `F_val`/`F_test` are per-protein pooled feature banks (rows =
    proteins, in the same order as `y_val`/`y_test`)."""
    from joblib import Parallel, delayed

    cand = topk_filter_continuous(F_val, y_val, topk)
    scored = Parallel(n_jobs=n_jobs, max_nbytes=None)(
        delayed(_fit_one)(np.ascontiguousarray(F_val[:, c]), y_val,
                          np.ascontiguousarray(F_test[:, c]), y_test, seed) for c in cand
    )
    val_rhos = np.array([s[0] for s in scored])
    test_rhos = np.array([s[1] for s in scored])
    best = int(np.argmax(np.abs(val_rhos)))    # strength of association, either sign
    return ContinuousProbeResult(
        property=prop, method=method,
        val_spearman=float(val_rhos[best]), test_spearman=float(test_rhos[best]),
        feature_index=int(cand[best]), n_train=int(len(y_val)), n_test=int(len(y_test)), u=u,
    )
