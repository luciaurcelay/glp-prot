"""Discovery §2 — does each representation's latent space reflect the Swiss-Prot
family hierarchy (EF-hand 1-4, Ig-like subtypes, RRM 1/2, ...) baked into
`configs/discovery_families.yaml`? Pure numpy logic; the CLI script
(`scripts/discovery_hierarchy.py`) supplies the per-protein pooled activations.

Design (Discovery spec Part 2):
  1. pairwise Pearson correlation between winning units' per-protein activation
     vectors, restricted to concepts that belong to a listed family;
  2. mean within-family vs. mean between-family correlation, EXCLUDING self-pairs
     (same winning unit for both concepts — spec's explicit confound: a unit that
     wins two related concepts trivially correlates with itself);
  3. a label-permutation test (shuffle which family each concept belongs to,
     preserving family sizes) for the within-minus-between difference — the
     self-pair exclusion mask is fixed across permutations (it's about unit
     identity, not family assignment, so permuting labels doesn't change it);
  4. concept-concept co-occurrence (fraction of pooled proteins where both
     concepts are present) as a check that the family isn't just "the same
     annotation duplicated" (spec Open item 3).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_families(path) -> Dict[str, List[str]]:
    import yaml
    return yaml.safe_load(open(path))["families"]


def family_of_concept(families: Dict[str, List[str]]) -> Dict[str, str]:
    """concept -> family name, for every concept listed in any family (spec: the
    hierarchy test is restricted to concepts that HAVE a family, not all 104)."""
    out = {}
    for fam, concepts in families.items():
        for c in concepts:
            out[c] = fam
    return out


@dataclass
class PairRow:
    concept_a: str
    concept_b: str
    same_family: bool
    same_unit: bool
    correlation: float
    representation: str

    def to_dict(self) -> Dict:
        return asdict(self)


def pairwise_rows(concepts: List[str], corr: np.ndarray, family_of: Dict[str, str],
                  unit_of: Dict[str, int], representation: str) -> List[PairRow]:
    """One row per unordered concept pair (i<j) — `discovery_hierarchy.csv`."""
    rows = []
    n = len(concepts)
    for i in range(n):
        for j in range(i + 1, n):
            ca, cb = concepts[i], concepts[j]
            rows.append(PairRow(
                concept_a=ca, concept_b=cb,
                same_family=family_of.get(ca) == family_of.get(cb),
                same_unit=unit_of[ca] == unit_of[cb],
                correlation=float(corr[i, j]),
                representation=representation,
            ))
    return rows


def permutation_test(concepts: List[str], corr: np.ndarray, family_of: Dict[str, str],
                     unit_of: Dict[str, int], n_perm: int = 10000, seed: int = 0
                     ) -> Dict:
    """Within-family vs. between-family mean correlation, self-pairs excluded, with a
    one-sided (within > between) label-permutation p-value. `corr`/`concepts` must
    already be restricted to concepts that have a family (every entry of `family_of`
    covers all of `concepts`)."""
    n = len(concepts)
    families = np.array([family_of[c] for c in concepts])
    units = np.array([unit_of[c] for c in concepts])
    iu, ju = np.triu_indices(n, k=1)
    self_pair = units[iu] == units[ju]
    n_excluded = int(self_pair.sum())
    pair_corr = corr[iu, ju]

    def diff_for(fam_labels: np.ndarray) -> Tuple[float, float, float]:
        same = (fam_labels[iu] == fam_labels[ju]) & ~self_pair
        diff_fam = (fam_labels[iu] != fam_labels[ju]) & ~self_pair
        w = float(pair_corr[same].mean()) if same.any() else float("nan")
        b = float(pair_corr[diff_fam].mean()) if diff_fam.any() else float("nan")
        return w, b, w - b

    within_mean, between_mean, observed = diff_for(families)

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(families)
        _, _, null[k] = diff_for(perm)
    p_value = float((null >= observed).mean())

    return {
        "within_family_mean": within_mean, "between_family_mean": between_mean,
        "difference": observed, "permutation_p_value": p_value,
        "n_pairs_total": int(len(iu)), "n_pairs_within": int(((families[iu] == families[ju]) & ~self_pair).sum()),
        "n_pairs_between": int(((families[iu] != families[ju]) & ~self_pair).sum()),
        "n_excluded_self_pairs": n_excluded, "n_permutations": n_perm,
    }


def cooccurrence(concepts: List[str], protein_presence: Dict[str, np.ndarray]) -> np.ndarray:
    """(len(concepts), len(concepts)) fraction of pooled proteins where both concepts
    are present (protein-level: >=1 positive residue) — spec Open item 3, "are these
    concepts genuinely independent, or do they co-occur in the eval proteins?"."""
    P = np.stack([protein_presence[c].astype(float) for c in concepts])   # (C, n_protein)
    n_protein = P.shape[1]
    return (P @ P.T) / n_protein
