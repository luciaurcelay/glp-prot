"""Continuous protein properties (GLProt probing spec Part 1, "new, cheap" axis).

Free labels on any sequence — no annotation lookup, no predictor — computed with
BioPython's `ProteinAnalysis` and modlAMP's `GlobalDescriptor` (aliphatic index isn't
in BioPython). Unlike the discrete Swiss-Prot concepts these are whole-protein scalars,
so probing them (glp_plm/eval/probing_continuous.py) works on PER-PROTEIN pooled
features, not per-residue ones.

GRAVY is the one that matters: it's the same target as the DiffMean steering
experiment (`scripts/build_gravy_contrast.py`, Huang et al. Eq. 2's whole-sequence
average activation convention), so "the meta-neuron that best isolates hydrophobicity"
connects probing to steering directly (spec Part 7 figure 4).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

# Order fixed here (not dict iteration order) — CSV column order and the GRAVY-panel
# figure both key off this.
PROPERTY_NAMES: List[str] = [
    "gravy", "molecular_weight", "isoelectric_point", "aromaticity",
    "instability_index", "charge_at_ph7", "aliphatic_index",
    "helix_fraction", "turn_fraction", "sheet_fraction",
]

# BioPython's Kyte-Doolittle GRAVY table (and the rest of ProteinAnalysis) has no
# entry for ambiguous/nonstandard codes UniRef50/Swiss-Prot can contain (X/B/Z/U/O) —
# same filter build_gravy_contrast.py already applies for the identical reason.
CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


def compute_properties(seq: str) -> Optional[Dict[str, float]]:
    """The ten properties in `PROPERTY_NAMES` for one sequence, or None if `seq`
    contains a non-canonical residue (undefined for these scales/descriptors)."""
    seq = seq.strip().upper()
    if not seq or not set(seq) <= CANONICAL_AA:
        return None

    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    pa = ProteinAnalysis(seq)
    helix, turn, sheet = pa.secondary_structure_fraction()
    props = {
        "gravy": float(pa.gravy()),
        "molecular_weight": float(pa.molecular_weight()),
        "isoelectric_point": float(pa.isoelectric_point()),
        "aromaticity": float(pa.aromaticity()),
        "instability_index": float(pa.instability_index()),
        "charge_at_ph7": float(pa.charge_at_pH(7.0)),
        "helix_fraction": float(helix),
        "turn_fraction": float(turn),
        "sheet_fraction": float(sheet),
    }

    from modlamp.descriptors import GlobalDescriptor

    gd = GlobalDescriptor([seq])
    gd.aliphatic_index()
    props["aliphatic_index"] = float(gd.descriptor[0, 0])
    return props


def properties_for_proteins(accessions: List[str], sequences: List[str]) -> pd.DataFrame:
    """`PROPERTY_NAMES` columns, indexed by accession, for every scoreable sequence.

    Silently drops sequences `compute_properties` can't score (logs the count) —
    callers must re-align any parallel per-protein data (e.g. activation spans) to
    the returned index, not to the original `accessions` order.
    """
    rows = []
    for acc, seq in zip(accessions, sequences):
        p = compute_properties(seq)
        if p is None:
            continue
        p["accession"] = acc
        rows.append(p)
    n_dropped = len(sequences) - len(rows)
    if n_dropped:
        print(f"[continuous_properties] dropped {n_dropped}/{len(sequences)} sequences "
              f"(non-canonical residues)")
    if not rows:
        return pd.DataFrame(columns=PROPERTY_NAMES)
    df = pd.DataFrame(rows).set_index("accession")
    return df[PROPERTY_NAMES]
