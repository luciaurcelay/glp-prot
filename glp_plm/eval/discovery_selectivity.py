"""Discovery §3 — selectivity (polysemanticity). "GLP isolates concepts" is a
sufficiency claim (the unit contains the information); this asks whether the unit
responds to ONLY that concept. Needs the full unit x concept AUC matrix (every
winning unit's AUC against every OTHER concept too, not just its own) — the
1-D probing sweep (`probing_1d.probe_concept`) never stores this, it keeps only the
per-concept winner, so `scripts/discovery_selectivity.py` computes it fresh.

Note going in (Discovery spec): the shared-winning-unit cases already visible in
`falsification.json`/`probing.json` (e.g. one GLP unit winning both Domain_Globin
and Domain_PB1 — see `tab:app-falsification`) suggest GLP meta-neurons ARE
polysemantic to some degree. Report both metrics regardless of which way they cut.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np


@dataclass
class SelectivityRow:
    representation: str
    unit: int
    winning_concept: str
    winning_auc: float
    second_best_auc: float
    second_best_concept: str
    selectivity_ratio: float
    n_concepts_above_0_8: int

    def to_dict(self) -> Dict:
        return asdict(self)


def selectivity_rows(representation: str, winning_concept: str, unit: int,
                     auc_by_concept: Dict[str, float]) -> SelectivityRow:
    """`auc_by_concept`: every concept this unit was evaluated against -> test AUC
    (must include `winning_concept` itself). Ratio = winning AUC / second-best AUC
    across all OTHER concepts; concept count = # concepts (excluding its own) where
    this unit's AUC exceeds 0.8."""
    winning_auc = auc_by_concept[winning_concept]
    others = {c: a for c, a in auc_by_concept.items() if c != winning_concept}
    second_concept, second_auc = max(others.items(), key=lambda kv: kv[1])
    n_above = sum(1 for a in others.values() if a > 0.8)
    ratio = winning_auc / second_auc if second_auc > 0 else float("inf")
    return SelectivityRow(
        representation=representation, unit=unit, winning_concept=winning_concept,
        winning_auc=winning_auc, second_best_auc=second_auc, second_best_concept=second_concept,
        selectivity_ratio=ratio, n_concepts_above_0_8=n_above,
    )


def summarize(rows: List[SelectivityRow]) -> Dict[str, Dict[str, float]]:
    """Per-representation distribution summary (median/IQR) of both metrics."""
    out = {}
    by_rep: Dict[str, List[SelectivityRow]] = {}
    for r in rows:
        by_rep.setdefault(r.representation, []).append(r)
    for rep, rs in by_rep.items():
        ratios = np.array([r.selectivity_ratio for r in rs if np.isfinite(r.selectivity_ratio)])
        counts = np.array([r.n_concepts_above_0_8 for r in rs])
        out[rep] = {
            "n_units": len(rs),
            "selectivity_ratio_median": float(np.median(ratios)) if len(ratios) else float("nan"),
            "selectivity_ratio_p25": float(np.percentile(ratios, 25)) if len(ratios) else float("nan"),
            "selectivity_ratio_p75": float(np.percentile(ratios, 75)) if len(ratios) else float("nan"),
            "n_concepts_above_0.8_median": float(np.median(counts)),
            "frac_polysemantic_gt1": float((counts > 1).mean()),
        }
    return out
