"""Falsification test for a GLP-meta-neuron-beats-SAE probing win (GLProt probing
spec Part 6, required before claiming nonlinearity/new-structure).

A GLP meta-neuron outscoring every SAE feature on a concept is only evidence of NEW
structure if that meta-neuron isn't just a linear repackaging of the SAE dictionary
InterPLM already released. For each concept where GLP wins, regress the winning
meta-neuron's activation on the FULL SAE feature dictionary (10,240-dim; linear
regression SAE features -> meta-neuron activation) and report R^2, held out on test.

High R^2 => the SAE dictionary already spans this direction — repackaging, not new
structure, and the stronger "unconstrained generative substrate finds things
dictionaries can't" claim must be retracted for that concept. Report the R^2
distribution regardless of outcome — that distribution is the difference between a
defensible claim and an overclaim, not just the individual pass/fail calls.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np


@dataclass
class FalsificationResult:
    concept: str
    u: float
    feature_index: int
    r2: float
    n_train: int
    n_test: int

    def to_dict(self) -> Dict:
        return asdict(self)


def glp_wins(probing_results: List[dict]) -> List[dict]:
    """GLP result rows (carrying `u`/`feature_index`) for every concept where GLP's
    test_auc beats the SAE's test_auc (matched concept, both present)."""
    sae_auc = {r["concept"]: r["test_auc"] for r in probing_results if r["method"] == "sae"}
    glp_rows = {r["concept"]: r for r in probing_results if r["method"] == "glp"}
    return [r for c, r in glp_rows.items() if c in sae_auc and r["test_auc"] > sae_auc[c]]


def regress_on_sae_dict(sae_feats_train: np.ndarray, target_train: np.ndarray,
                        sae_feats_test: np.ndarray, target_test: np.ndarray) -> float:
    """R^2 of the winning meta-neuron's activation, predicted from the full SAE
    dictionary, held out on test. RidgeCV (not plain OLS) because 10,240 predictors
    against a few thousand rows would otherwise let a high train R^2 come from
    overfitting rather than genuine linear predictability."""
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score

    model = RidgeCV(alphas=np.logspace(-2, 4, 13))
    model.fit(sae_feats_train, target_train)
    pred = model.predict(sae_feats_test)
    return float(r2_score(target_test, pred))


def r2_distribution(results: List[dict]) -> Dict[str, float]:
    r2 = np.array([r["r2"] for r in results], dtype=float)
    if len(r2) == 0:
        return {"n": 0}
    return {
        "n": int(len(r2)), "mean": float(r2.mean()), "median": float(np.median(r2)),
        "p25": float(np.percentile(r2, 25)), "p75": float(np.percentile(r2, 75)),
        "min": float(r2.min()), "max": float(r2.max()),
        "frac_above_0.8": float((r2 > 0.8).mean()),
    }
