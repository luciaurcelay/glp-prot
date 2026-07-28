"""Pre-registered Swiss-Prot concept selection (PLAN §7) -> frozen_concepts.json.

`⚠ FAIRNESS`: the headline concept set is frozen BEFORE any GLP probing runs, using
only (a) InterPLM's own frequency filter, (b) split statistics, and (c) the SAE's own
tractability — **never any GLP output**. Deterministic given the data + seed.

Procedure:
  1. pool     — concepts that survived InterPLM's frequency filter (prepare_eval_set)
  2. stability— require >= min_pos_test positive AA tokens in the TEST split
  3. category — tag into five biological buckets so the table spans biology
  4. spread   — within each bucket, deliberately span the SAE-tractability range
                (easy AND hard for the SAE), so a GLP win can't be cherry-picked
  5. freeze   — n_per_category per bucket; log everything

Deviation from §7 worth stating in the paper: tractability is scored as the SAE's
best single-feature **val AUC** rather than val F1. It preserves the property that
matters (SAE-only, no GLP) and avoids running a threshold sweep at selection time.
(`probing_f1` now exists, so swapping in F1 is possible — but note it would change
which concepts are frozen, so do it deliberately, not silently.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Five biological buckets (PLAN §7 step 3). Matched against the materialized
# display-name columns (never the `ft_*` REST ids) — see docs/INTERPLM_MAP.md.
CATEGORY_PATTERNS: Dict[str, List[str]] = {
    "structural":            ["Helix", "Beta strand", "Turn", "Coiled coil"],
    "chemical_ptm":          ["Disulfide bond", "Glycosylation", "Modified residue", "Lipidation"],
    "binding_catalytic":     ["Active site", "Binding site", "Cofactor"],
    "topology_localization": ["Signal peptide", "Transit peptide", "Transmembrane", "Intramembrane"],
    "domain_motif":          ["Domain", "Motif", "Region", "Zinc finger", "Compositional bias"],
}

# Residue-identity one-hots are not annotations — excluded from the concept pool.
EXCLUDE_PREFIXES = ("amino_acid",)


def categorize(concept: str) -> Optional[str]:
    for cat, pats in CATEGORY_PATTERNS.items():
        if any(concept.startswith(p) for p in pats):
            return cat
    return None


def select_concepts(
    concept_names: List[str],
    n_pos_test: np.ndarray,
    sae_tractability: Optional[Dict[str, float]] = None,
    min_pos_test: int = 500,
    n_per_category: int = 3,
    seed: int = 0,
) -> Dict:
    """Return the frozen selection + full audit trail (see module docstring)."""
    pool, dropped = [], {}

    for name, npos in zip(concept_names, n_pos_test):
        if name.startswith(EXCLUDE_PREFIXES):
            dropped[name] = "residue-identity one-hot"
            continue
        if npos < min_pos_test:
            dropped[name] = f"only {int(npos)} positive AAs in test (<{min_pos_test})"
            continue
        cat = categorize(name)
        if cat is None:
            dropped[name] = "no category match"
            continue
        pool.append({"concept": name, "category": cat, "n_pos_test": int(npos),
                     "sae_tractability": (sae_tractability or {}).get(name)})

    # step 4+5: within each category span the SAE-tractability range
    headline: List[Dict] = []
    by_cat: Dict[str, List[Dict]] = {}
    for c in pool:
        by_cat.setdefault(c["category"], []).append(c)

    for cat, items in sorted(by_cat.items()):
        scored = [c for c in items if c["sae_tractability"] is not None]
        if len(scored) >= n_per_category:
            # span the range: hardest, median, easiest for the SAE
            scored.sort(key=lambda c: c["sae_tractability"])
            picks_idx = np.linspace(0, len(scored) - 1, n_per_category).round().astype(int)
            picks = [scored[i] for i in dict.fromkeys(picks_idx.tolist())]
        else:
            # not enough scored concepts: fall back to the most abundant, seeded tie-break
            items = sorted(items, key=lambda c: (-c["n_pos_test"], c["concept"]))
            picks = items[:n_per_category]
        for p in picks:
            p = dict(p)
            p["selection_reason"] = "SAE-tractability spread" if p["sae_tractability"] is not None else "abundance fallback"
            headline.append(p)

    return {
        "headline": headline,                       # (a) illustrative table
        "all_qualifying": [c["concept"] for c in pool],   # (b) the un-gameable aggregate
        "by_category": {k: [c["concept"] for c in v] for k, v in sorted(by_cat.items())},
        "criteria": {
            "min_pos_test": min_pos_test,
            "n_per_category": n_per_category,
            "seed": seed,
            "tractability_metric": "best single-SAE-feature val AUC",
        },
        "dropped": dropped,
    }


def save(selection: Dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection, indent=2))
    return path
