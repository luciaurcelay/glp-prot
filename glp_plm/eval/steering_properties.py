"""The whole-sequence properties Exp B (diff-of-means) can steer.

Banerjee et al. steer three biochemical properties — hydrophobicity, charge and
instability. This module is the single place that says what each one *is*, so the
contrast builder, the generation loop, and every analysis script agree.

Three design decisions live here, and they are the ones a reviewer would push on:

1. **Selection scores are length-normalized; reported scores are the same quantity.**
   GRAVY is already a per-residue mean. Net charge is a *sum*, so ranking proteins by
   it ranks them substantially by length, and a diff-of-means vector built on that
   contrast would encode "long" as much as "cationic". We therefore steer and report
   charge *per residue* at pH 7. The instability index is already length-normalized
   (10/L times a dipeptide sum). `score_all` still records net charge alongside, and
   since masked-Gibbs never changes length, the two are proportional within an
   init sequence — the pairing every effect estimate uses.

2. **Every generated sequence is scored on every property, not just the steered one.**
   That is what makes an off-target/specificity matrix possible: steering charge
   should move charge and leave the others where the natural covariance between them
   would put it. Without the off-target columns there is no way to tell a concept
   direction from a generic "make this protein weirder" direction.

   Measured on a 5,790-sequence UniRef50 pool, that covariance is GRAVY x instability
   -0.348, charge x instability +0.160, and GRAVY x charge only -0.030 — the last is
   worth knowing because the obvious intuition (charged residues are hydrophilic) is
   wrong for *net* charge: a protein rich in both Lys and Glu is hydrophilic with
   near-zero net charge. Hydrophobicity and charge are therefore near-orthogonal axes.

3. **Each property carries its own exact decomposition.** GRAVY and charge are linear
   in amino-acid composition, so a change in them decomposes exactly into per-residue
   contributions; the instability index is a sum over *dipeptides* and does not. That
   asymmetry is the point of including instability: it is the one target that a purely
   compositional artifact (such as the SDEdit correction's own Leu/Val drift, see
   docs/steering_history.md §13) cannot trivially produce, and the decomposition is
   how we check whether a measured gain came from a broad shift or from stacking a
   handful of high-weight dipeptides.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")

#: pH at which charge is evaluated. Banerjee et al. use modlAMP's charge at pH 7;
#: BioPython's `charge_at_pH` is the same Henderson-Hasselbalch quantity and needs no
#: extra dependency (MN5 is offline, and modlAMP is not in the project venv).
CHARGE_PH = 7.0


def _pa(seq: str):
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    return ProteinAnalysis(seq)


# --------------------------------------------------------------------------- #
#  Scores
# --------------------------------------------------------------------------- #
def gravy_score(seq: str) -> float:
    """Kyte-Doolittle mean hydropathy. Higher = more hydrophobic."""
    return float(_pa(seq).gravy())


def charge_score(seq: str) -> float:
    """Net charge at pH 7 divided by length. Higher = more cationic.

    Per residue, not net: see this module's docstring, point 1. Not *perfectly*
    length-invariant — `charge_at_pH` adds a fixed N- and C-terminal term that does not
    scale with length, worth ~0.005 charge/residue at 100 aa and half that at 200 aa. That
    residual is ~10% of the natural spread, an order of magnitude smaller than the length
    dependence of net charge, and the contrast builder's length-gap diagnostic covers what
    is left.
    """
    return float(_pa(seq).charge_at_pH(CHARGE_PH)) / max(len(seq), 1)


def net_charge(seq: str) -> float:
    return float(_pa(seq).charge_at_pH(CHARGE_PH))


def instability_score(seq: str) -> float:
    """Guruprasad instability index. >40 predicts an unstable protein in vivo."""
    return float(_pa(seq).instability_index())


# --------------------------------------------------------------------------- #
#  Exact decompositions — "where did the change come from?"
# --------------------------------------------------------------------------- #
def _aa_frequencies(seqs: Sequence[str]) -> Dict[str, float]:
    """Mean over sequences of each sequence's own residue frequency.

    Each sequence weighted EQUALLY, not each residue. The decomposition has to reconstruct
    the mean of a per-sequence score, and

        mean_i GRAVY(s_i) = mean_i sum_a f_i(a) w(a) = sum_a [mean_i f_i(a)] w(a),

    which is this quantity. Pooling all residues into one bag instead would weight long
    proteins more and the components would no longer sum to the effect they explain — a
    silent error, since it only shows up when the sequences differ in length (ours span
    100-200 aa).
    """
    if not seqs:
        return {a: 0.0 for a in sorted(CANONICAL_AA)}
    acc = {a: 0.0 for a in sorted(CANONICAL_AA)}
    for s in seqs:
        c = Counter(s)
        for a in acc:
            acc[a] += c.get(a, 0) / max(len(s), 1)
    return {a: v / len(seqs) for a, v in acc.items()}


def _kd_table() -> Dict[str, float]:
    from Bio.SeqUtils import ProtParamData

    return dict(ProtParamData.kd)


def _charge_table() -> Dict[str, float]:
    """Per-residue partial charge at CHARGE_PH, from BioPython's own pK tables.

    `IsoelectricPoint.charge_at_pH` is a sum of these over residue counts plus two
    terminal terms, so composition explains it exactly up to the termini.
    """
    from Bio.SeqUtils import IsoelectricPoint as IP

    q = {a: 0.0 for a in sorted(CANONICAL_AA)}
    for aa, pK in IP.positive_pKs.items():
        if aa in q:
            q[aa] += 1.0 / (10 ** (CHARGE_PH - pK) + 1.0)
    for aa, pK in IP.negative_pKs.items():
        if aa in q:
            q[aa] -= 1.0 / (10 ** (pK - CHARGE_PH) + 1.0)
    return q


def _composition_decomposition(before: Sequence[str], after: Sequence[str],
                               weights: Dict[str, float]) -> List[Tuple[str, float]]:
    """For a score of the form sum_a f(a)*w(a): each residue's contribution to the change."""
    f0, f1 = _aa_frequencies(before), _aa_frequencies(after)
    return sorted(((a, (f1[a] - f0[a]) * weights.get(a, 0.0)) for a in sorted(CANONICAL_AA)),
                  key=lambda t: -abs(t[1]))


def gravy_decomposition(before: Sequence[str], after: Sequence[str]) -> List[Tuple[str, float]]:
    return _composition_decomposition(before, after, _kd_table())


def charge_decomposition(before: Sequence[str], after: Sequence[str]) -> List[Tuple[str, float]]:
    return _composition_decomposition(before, after, _charge_table())


def instability_decomposition(before: Sequence[str], after: Sequence[str],
                              top_k: Optional[int] = None) -> List[Tuple[str, float]]:
    """Per-DIPEPTIDE contributions, largest absolute change first.

    The instability index is 10/L * sum over adjacent pairs of DIWV, so it is *not* a
    function of composition alone; the components here are dipeptides, and there are up to
    400 of them. Returns all by default so the contributions sum to the actual change —
    callers truncate for display.
    """
    from Bio.SeqUtils import ProtParamData

    diwv = ProtParamData.DIWV

    def weighted(seqs: Sequence[str]) -> Dict[str, float]:
        acc: Dict[str, float] = {}
        n = 0
        for s in seqs:
            for i in range(len(s) - 1):
                d = s[i:i + 2]
                acc[d] = acc.get(d, 0.0) + 10.0 * diwv[s[i]][s[i + 1]] / len(s)
            n += 1
        return {d: v / max(n, 1) for d, v in acc.items()}

    w0, w1 = weighted(before), weighted(after)
    diffs = [(d, w1.get(d, 0.0) - w0.get(d, 0.0)) for d in sorted(set(w0) | set(w1))]
    ranked = sorted(diffs, key=lambda t: -abs(t[1]))
    return ranked[:top_k] if top_k else ranked


# --------------------------------------------------------------------------- #
#  The registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SteeringProperty:
    key: str
    leaf: str            # run-directory suffix: steering_diffmean_<leaf>
    label: str           # axis / prose label, e.g. "hydrophobicity (GRAVY)"
    unit: str            # short unit for a delta, e.g. "GRAVY"
    score: Callable[[str], float]
    high_name: str       # what the top quartile is, e.g. "hydrophobic"
    low_name: str        # what the bottom quartile is, e.g. "hydrophilic"
    #: (before, after) -> [(component, contribution)], most-changed first
    decompose: Callable[[Sequence[str], Sequence[str]], List[Tuple[str, float]]]
    #: True when `score` is exactly sum_a frequency(a) * weight(a)
    compositional: bool
    fmt: str = "+.3f"

    @property
    def delta_label(self) -> str:
        return f"Δ{self.unit}"


PROPERTIES: Dict[str, SteeringProperty] = {
    "gravy": SteeringProperty(
        key="gravy", leaf="hydro", label="hydrophobicity (GRAVY)", unit="GRAVY",
        score=gravy_score, high_name="hydrophobic", low_name="hydrophilic",
        decompose=gravy_decomposition, compositional=True, fmt="+.3f"),
    "charge": SteeringProperty(
        key="charge", leaf="charge", label=f"net charge per residue (pH {CHARGE_PH:g})",
        unit="charge/residue", score=charge_score,
        high_name="cationic", low_name="anionic",
        decompose=charge_decomposition, compositional=True, fmt="+.4f"),
    "instability": SteeringProperty(
        key="instability", leaf="instability", label="instability index (Guruprasad)",
        unit="instability index", score=instability_score,
        high_name="unstable", low_name="stable",
        decompose=instability_decomposition, compositional=False, fmt="+.2f"),
}

#: Order used for CSV columns and the specificity matrix.
PROPERTY_KEYS: List[str] = ["gravy", "charge", "instability"]


def get_property(key: str) -> SteeringProperty:
    if key not in PROPERTIES:
        raise KeyError(f"unknown steering property {key!r}; choose from {sorted(PROPERTIES)}")
    return PROPERTIES[key]


def score_all(seq: str) -> Dict[str, float]:
    """Every steerable property plus net charge, for one sequence.

    Computed on every generation regardless of which property was steered — the
    off-target columns are what turn three separate experiments into a specificity
    matrix. Returns {} for sequences with non-canonical residues (the scales are
    undefined there); the generation loop only ever emits canonical residues, so this
    only guards externally-supplied sequences.
    """
    seq = seq.strip().upper()
    if not seq or not set(seq) <= CANONICAL_AA:
        return {}
    pa = _pa(seq)
    q = float(pa.charge_at_pH(CHARGE_PH))
    return {"gravy": float(pa.gravy()),
            "charge": q / len(seq),
            "instability": float(pa.instability_index()),
            "net_charge": q}


#: Column names `score_all` fills, in CSV order.
SCORE_ALL_KEYS: List[str] = PROPERTY_KEYS + ["net_charge"]


# --------------------------------------------------------------------------- #
#  Reading a finished run — shared by every analysis/plot script
# --------------------------------------------------------------------------- #
#: Exp B run CSVs before the generalization were always `expB_diffmean_gravy.csv` and
#: carried `delta_gravy` as the steered effect; both are still read transparently.
LEGACY_CSV = "expB_diffmean_gravy.csv"


def find_run_csv(run_dir) -> Path:
    """The one `expB_diffmean_<property>.csv` in `run_dir` (Exp A's CSV as a fallback)."""
    run_dir = Path(run_dir)
    hits = sorted(run_dir.glob("expB_diffmean_*.csv"))
    if len(hits) > 1:
        raise ValueError(f"{run_dir} holds {len(hits)} Exp B CSVs ({[h.name for h in hits]}); "
                         "a run steers one property — point --run-dir at a single run")
    if hits:
        return hits[0]
    for name in (LEGACY_CSV, "expA_sae_zincfinger.csv"):
        if (run_dir / name).exists():
            return run_dir / name
    raise FileNotFoundError(f"no steering CSV in {run_dir}")


def find_contrast_json(run_dir, prop_key: Optional[str] = None) -> Optional[Path]:
    """`<property>_contrast.json` in `run_dir`, or the legacy `gravy_contrast.json`."""
    run_dir = Path(run_dir)
    if prop_key:
        p = run_dir / f"{prop_key}_contrast.json"
        if p.exists():
            return p
    hits = sorted(run_dir.glob("*_contrast.json"))
    return hits[0] if hits else None


def run_property(rows: Sequence[dict], csv_path=None) -> SteeringProperty:
    """Which property a finished run steered.

    Prefers the `property` column, falls back to the CSV filename, and finally to GRAVY —
    which is right for every run written before the other two properties existed.
    """
    for r in rows:
        key = (r.get("property") or "").strip()
        if key:
            return get_property(key)
    if csv_path is not None:
        stem = str(csv_path).rsplit("expB_diffmean_", 1)
        if len(stem) == 2:
            key = stem[1].rsplit(".csv", 1)[0]
            if key in PROPERTIES:
                return get_property(key)
    return get_property("gravy")


def row_delta(row: dict) -> float:
    """The steered property's change for one generated sequence.

    `delta_prop` is the generic column; `delta_gravy` is what pre-generalization runs
    wrote, where it was the steered delta by construction.
    """
    v = row.get("delta_prop")
    if v is None or v == "":
        v = row["delta_gravy"]
    return float(v)
