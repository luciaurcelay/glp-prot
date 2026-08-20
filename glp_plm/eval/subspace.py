"""Subspace analysis — what distinguishes the 15 not-in-dictionary concepts
(Discovery §1) from the 79 the joint SAE top-k probe recovers? Polysemanticity
was ruled out (Section~sec:discovery-organisation). This tests a geometric
hypothesis instead: does the information live in the part of GLP's meta-neuron
activations that falls OUTSIDE the linear span of the SAE dictionary?

Not the residual probe already run (`x - sae.decode(sae.encode(x))`, the part of
the ESM activation the SAE fails to reconstruct — a negative result). This
decomposes GLP's OWN meta-neuron activations `G` against the SAE feature span
`S`, via the same ridge regularization the falsification check (`glp_plm.eval.
falsification`) uses, so the two analyses are numerically comparable:

    W          = ridge fit of G ~ S on TRAIN residues (RidgeCV, alpha_per_target)
    G_parallel = S @ W            (the part of G the SAE dictionary explains)
    G_perp     = G - G_parallel   (the part it does not)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def fit_projection(S_train: np.ndarray, G_train: np.ndarray, alphas=None):
    """RidgeCV(alpha_per_target=True): one SVD of S_train, reused to pick an
    independently-optimal ridge alpha per G column via efficient leave-one-out
    generalized CV — the same regularization `falsification.regress_on_sae_dict`
    uses (`RidgeCV(alphas=np.logspace(-2, 4, 13))`, default `cv=None` -> LOO-GCV),
    just fit once for all of G's columns instead of once per column."""
    from sklearn.linear_model import RidgeCV

    if alphas is None:
        alphas = np.logspace(-2, 4, 13)
    ridge = RidgeCV(alphas=alphas, alpha_per_target=True)
    ridge.fit(S_train, G_train)
    return ridge


def decompose(ridge, S: np.ndarray, G: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(G_parallel, G_perp) for new residues, using an already-fit projection."""
    G_parallel = ridge.predict(S)
    return G_parallel, G - G_parallel


def per_unit_r2(G_true: np.ndarray, G_parallel: np.ndarray) -> np.ndarray:
    """Held-out per-unit R^2 of G_parallel against G_true — the sanity check
    against `falsification.json`'s per-concept R^2 (same quantity, computed for
    every unit here instead of just the 94 winners)."""
    from sklearn.metrics import r2_score
    return r2_score(G_true, G_parallel, multioutput="raw_values")


@dataclass
class SubspaceProbeRow:
    concept: str
    group: str                    # "not_in_dictionary" | "distributed"
    auc_glp_full: float
    auc_glp_parallel: float
    auc_glp_perp: float
    selected_unit_perp: int
    noise_level_u: float

    def to_dict(self) -> Dict:
        return asdict(self)


def mann_whitney_effect(not_in: List[float], distributed: List[float]) -> Dict:
    """Mann-Whitney U (two-sided) + rank-biserial effect size (same test family as
    the polysemanticity check, for consistency). Effect size matters more than the
    p-value at n=15 vs n=79 (underpowered for anything but a large effect)."""
    from scipy.stats import mannwhitneyu

    u_stat, p = mannwhitneyu(not_in, distributed, alternative="two-sided")
    n1, n2 = len(not_in), len(distributed)
    rank_biserial = 1 - (2 * u_stat) / (n1 * n2)   # in [-1, 1]; 0 = no effect
    return {
        "n_not_in_dictionary": n1, "n_distributed": n2,
        "median_not_in_dictionary": float(np.median(not_in)),
        "median_distributed": float(np.median(distributed)),
        "mean_not_in_dictionary": float(np.mean(not_in)),
        "mean_distributed": float(np.mean(distributed)),
        "u_statistic": float(u_stat), "p_value": float(p),
        "rank_biserial_effect_size": float(rank_biserial),
    }


def twonn_intrinsic_dimension(X: np.ndarray, seed: int = 0) -> float:
    """TwoNN intrinsic-dimension estimator (Facco et al. 2017), the same estimator
    `plm-sae` (Villegas Garcia & Ansuini, already in this paper's bibliography)
    uses to pick their hook layer. For each point, mu = (dist to 2nd NN) / (dist
    to 1st NN); under a locally-uniform density the MLE of the intrinsic
    dimension is d = N / sum(log(mu_i)). Discards mu<=1 (numerical ties)."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=3).fit(X)   # self + 2 nearest
    dist, _ = nn.kneighbors(X)
    r1, r2 = dist[:, 1], dist[:, 2]
    mu = r2 / np.maximum(r1, 1e-12)
    mu = mu[mu > 1]
    return float(len(mu) / np.sum(np.log(mu)))


def _center(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA (Kornblith et al. 2019): centered-Gram HSIC, normalized. X, Y
    are (n_samples, dim) — CKA is invariant to dim, only n_samples must match.
    Known to be dominated by a few high-variance directions and sensitive to
    preprocessing (report subsample size/preprocessing alongside any value)."""
    Kx, Ky = X @ X.T, Y @ Y.T
    Kx_c, Ky_c = _center(Kx), _center(Ky)
    hsic_xy = np.sum(Kx_c * Ky_c)
    hsic_xx = np.sum(Kx_c * Kx_c)
    hsic_yy = np.sum(Ky_c * Ky_c)
    return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy))


# --------------------------------------------------------------------------- #
#  novelty follow-up (subspace_novelty.py): PCA on the un-annotated residual,
#  and per-unit variance-fraction bookkeeping for the block/noise-level source
#  check.
# --------------------------------------------------------------------------- #
def variance_fraction_orthogonal(G_true: np.ndarray, G_perp: np.ndarray) -> np.ndarray:
    """Per-unit share of `G_true`'s variance landing in `G_perp`
    (`var(G_perp)/var(G_true)`) — the same question `per_unit_r2` answers, but as
    a variance ratio instead of held-out R^2. Unlike R^2 this cannot go negative
    (a badly-regularized unit just gives a ratio near 1), which matters when
    averaging within a block: a few negative-R^2 units would otherwise drag a
    block's mean below 0 and make the block comparison in
    `scripts/subspace_novelty.py` uninterpretable."""
    var_true = G_true.var(axis=0)
    var_perp = G_perp.var(axis=0)
    return var_perp / np.maximum(var_true, 1e-12)


def fit_pca(X: np.ndarray, n_components: int = 10, seed: int = 0):
    """PCA on a (n_residues, n_units) matrix (typically `G_perp`, the residual
    orthogonal to one or more dictionaries). Returns `(pca, scores)`: the fitted
    `sklearn.decomposition.PCA` (for `.transform` on a second, disjoint sample —
    `scripts/subspace_novelty.py` projects a protein-pooled bank through the same
    components rather than refitting) and this call's own `(n_residues,
    n_components)` scores."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, random_state=seed)
    scores = pca.fit_transform(X)
    return pca, scores
