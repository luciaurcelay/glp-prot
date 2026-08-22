#!/usr/bin/env python
"""Score our existing Exp A steering runs with Banerjee et al.'s own concept-detection
methodology (arXiv:2507.06458), post-hoc, from data already on disk -- no regeneration.

Banerjee's paper uses two detectors, both applied to the FINISHED generated sequence
(no threshold calibration against real ground truth, unlike this project's own
feature_activation/pepcoil/calibrated-strand_frac detectors):

  Secondary structure (helix/beta strand, their §3.2.1): run PsiPred, collapse its
  output to 3 states, then report (a) the number of distinct contiguous domains of the
  target state and (b) what fraction of the chain they span. We don't have PsiPred output,
  but `scripts/fold_steering_outputs.py` already wrote a DSSP `ss_string` per generated
  sequence into `fold_confidence.csv` (HELIX_CODES={H,G,I}, STRAND_CODES={E,B}, same file)
  -- DSSP is a finer, structure-based analogue of PsiPred's sequence-based 3-state
  prediction, so this reuses that column rather than re-deriving anything.

  Zinc finger (their §3.2.2): a literal regex match, Cys-X(2-4)-Cys-X(12)-His-X(3-5)-His,
  on the raw sequence -- no fold needed. That is exactly `run_steering.py`'s C2H2_RE (not
  C2H2_RE_PROSITE, which is this project's own F1-calibrated improvement on it). Any run
  launched with the default `--concept-detector motif` already recorded a hit/no-hit under
  C2H2_RE in its `concept_count_strict` column, so this script just re-aggregates that
  column -- see `run_steering.py:2278`.

Usage:
  python3 scripts/analyze_banerjee_detectors.py
(no args -- paths below point at the runs already on disk; edit RUNS to add more)
"""
from __future__ import annotations

import csv
import statistics as stats
from collections import defaultdict
from pathlib import Path

RUNS_ROOT = Path("/gpfs/projects/bsc70/bsc070332/glp-prot-runs/uniref50_500m_4096_d24_steering")

HELIX_CODES = {"H", "G", "I"}
STRAND_CODES = {"E", "B"}

# (run dir, concept label, DSSP code set) -- the four concepts asked about.
SS_RUNS = [
    ("steering_sae_zinc_helix_5355", "Helix (feature 5355)", HELIX_CODES),
    ("steering_sae_zinc_betastrand_2868", "Beta strand solo (feature 2868)", STRAND_CODES),
    ("steering_sae_zinc_betastrand_2868_5524_pair", "Beta strand pair (2868+5524)", STRAND_CODES),
    # Coiled coil has NO analogue here: DSSP has no coiled-coil class, and Banerjee's paper
    # never tests this concept at all (checked directly against the paper text) -- there is
    # no "their methodology" to port for it. Included as a documented gap, not skipped silently.
]
ZINC_RUNS = [
    ("steering_sae_zinc_union1435", "Zinc finger UNION(1-5) (feature 1435)"),
]


def domain_count(ss: str, codes: set) -> int:
    """Number of maximal contiguous runs of a residue in `codes` -- Banerjee's "N distinct
    domains" framing (§3.2.1: "5 distinct alpha helix domains")."""
    n, in_run = 0, False
    for c in ss:
        hit = c in codes
        if hit and not in_run:
            n += 1
        in_run = hit
    return n


def analyze_ss(run_dir: str, label: str, codes: set) -> None:
    path = RUNS_ROOT / run_dir / "fold_confidence.csv"
    if not path.exists():
        print(f"[skip] {label}: {path} not found")
        return
    with open(path) as f:
        rows = list(csv.DictReader(f))
    cells = defaultdict(list)
    for r in rows:
        ss = r.get("ss_string") or ""
        if not ss or r.get("dssp_error"):
            continue
        key = (float(r["alpha"]), r["corrected"] == "True")
        n_dom = domain_count(ss, codes)
        pct = sum(c in codes for c in ss) / len(ss)
        cells[key].append((n_dom, pct))

    print(f"\n=== {label} -- Banerjee-style domain count / % chain spanned ===")
    print(f"{'alpha':>6} {'arm':>10} {'n':>5} {'mean domains':>13} {'mean % spanned':>15}")
    for (alpha, corrected), vals in sorted(cells.items()):
        n_doms = [v[0] for v in vals]
        pcts = [v[1] for v in vals]
        arm = "corrected" if corrected else "naive"
        print(f"{alpha:>6.1f} {arm:>10} {len(vals):>5} {stats.mean(n_doms):>13.2f} "
              f"{100 * stats.mean(pcts):>14.1f}%")


def analyze_zinc(run_dir: str, label: str) -> None:
    path = RUNS_ROOT / run_dir / "expA_sae_zincfinger.csv"
    if not path.exists():
        print(f"[skip] {label}: {path} not found")
        return
    with open(path) as f:
        rows = list(csv.DictReader(f))
    cells = defaultdict(list)
    for r in rows:
        strict = r.get("concept_count_strict", "")
        if strict == "":
            continue
        key = (float(r["alpha"]), r["corrected"] == "True")
        cells[key].append(float(strict) > 0)

    print(f"\n=== {label} -- Banerjee-exact C2H2 regex (concept_count_strict) ===")
    print(f"{'alpha':>6} {'arm':>10} {'hits/n':>10} {'hit rate':>10}")
    total_hits, total_n = 0, 0
    for (alpha, corrected), hits in sorted(cells.items()):
        arm = "corrected" if corrected else "naive"
        n_hits = sum(hits)
        total_hits += n_hits
        total_n += len(hits)
        print(f"{alpha:>6.1f} {arm:>10} {int(n_hits):>4}/{len(hits):<5} {100 * n_hits / len(hits):>9.1f}%")
    print(f"{'ALL':>6} {'both':>10} {int(total_hits):>4}/{total_n:<5} "
          f"{100 * total_hits / total_n if total_n else 0:>9.1f}%  "
          f"(Banerjee's own reported rate: 1/100 = 1.0%)")


if __name__ == "__main__":
    for run_dir, label, codes in SS_RUNS:
        analyze_ss(run_dir, label, codes)
    for run_dir, label in ZINC_RUNS:
        analyze_zinc(run_dir, label)
    print("\nCoiled coil: no Banerjee methodology to port -- their paper tests only "
          "molecular weight/GRAVY/instability, helix, beta strand, and zinc finger; "
          "coiled coil does not appear in it at all (checked against the paper text).")
