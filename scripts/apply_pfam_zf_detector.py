#!/usr/bin/env python
"""Apply the real Pfam zf-C2H2 HMMER scan (PF00096.33, via pyhmmer) to every zinc-finger
steering run's generated sequences -- the closest already-benchmarked stand-in for
Villegas-Garcia & Ansuini's actual zinc-finger verification tool (Sathyaseelan et al.
2023's ZnF-Prot web service is not reachable from here; Pfam matching is the literal
HMM-profile half of what that tool does -- see run_steering.py's module docstring,
lines ~34-49, and the calibration table at lines ~93-116).

Two corrections vs the first pass of this script (both found by manually inspecting the
one hit the first pass reported on `steering_sae_zinc_union1435`):

1. **E-value must be computed per-query, not batch-wide.** HMMER's E-value is
   p-value * Z (Z = effective target-database size, which pyhmmer's Pipeline defaults to
   the number of sequences in the search block). Batch-searching e.g. 1800 sequences at
   once gives every hit an E-value ~1800x larger than what searching that ONE sequence
   alone (matching ZnF-Prot's own per-query protocol) would give. Fix: batch for speed
   (one pipeline call per run), then divide the reported E-value by the run's own sequence
   count to recover the single-query-equivalent E-value ZnF-Prot's thresholds (primary
   1e-4, relaxed-round 1e-1, `--cut_ga`-style) are meant to be read against.
2. **A Pfam/HMM hit alone is not a verified zinc finger -- ZnF-Prot's own protocol requires
   a second check** (their NRML/SHFT/NMTF/XMUT/XNCH categories, §5.5.3): either a matching
   curated Cys/His sequence pattern (all 74 of theirs need >=2 Cys), or, failing that,
   confirmation that the aligned HMM region itself contains the profile's conserved
   Cys/His marker residues. We don't have their 74-pattern library, so this script applies
   the cheapest necessary (not sufficient) version of that check: a sequence with fewer
   than 2 cysteines in its ENTIRE length cannot possibly be a real C2H2 motif (needs 2 Cys
   + 2 His coordinating one Zn2+) -- any such "hit" is flagged spurious regardless of
   E-value. This is exactly what caught union1435's one apparent hit: 0 cysteines, and the
   HMM alignment itself only covered the profile's C-terminal His-bearing half (columns
   8-22 of 23), missing the N-terminal Cys-bearing columns entirely.

Usage: python3 scripts/apply_pfam_zf_detector.py
"""
import csv
from pathlib import Path

import pyhmmer
from pyhmmer.easel import Alphabet, TextSequence, DigitalSequenceBlock

HMM_PATH = "/gpfs/projects/bsc70/bsc070332/glp-prot-runs/adhoc_scripts/PF00096.hmm"
RUNS_ROOT = Path("/gpfs/projects/bsc70/bsc070332/glp-prot-runs/uniref50_500m_4096_d24_steering")

# Every zinc-finger-target Exp A run (excludes helix_5355 / betastrand_* / coiledcoil_813,
# which reuse the --zinc-finger-feature CLI flag's name for an unrelated SS concept).
RUN_DIRS = [
    "steering_sae_zinc",                              # F1 shortlist: 6158, 4421
    "steering_sae_zinc_interplm_exact",
    "steering_sae_zinc_scaleshift",
    "steering_sae_zinc_feat_1547_solo",
    "steering_sae_zinc_feat_5819_solo",
    "steering_sae_zinc_feat_6311_solo",
    "steering_sae_zinc_feat_7033_solo",
    "steering_sae_zinc_top2_interplm_pair",
    "steering_sae_zinc_top2_villegas_pair",
    "steering_sae_zinc_union1435",
    "steering_sae_zinc_union1435_alpha0to50_n100_archived",
    "steering_sae_zinc_union1435_argmax",
    "steering_sae_zinc_union1435_bmask10",
    "steering_sae_zinc_union_top2",
    "steering_sae_zinc_zinc_6311_highalpha_pilot",
    "steering_sae_zinc_zincfive",          # partial/cancelled run, ~3min in -- flagged, not clean
    "steering_sae_zinc_zincsix_min",
    "steering_sae_zinc_zinc_villegas",
    "steering_sae_zinc_zinc_any_5696",
]

PRIMARY_E = 1e-4   # ZnF-Prot round-1 cutoff
RELAXED_E = 1e-1   # ZnF-Prot round-2 relaxed cutoff; anything above this is excluded


def _s(x):
    return x.decode() if isinstance(x, bytes) else x


def main():
    alphabet = Alphabet.amino()
    with pyhmmer.plan7.HMMFile(HMM_PATH) as f:
        hmm = f.read()
    print(f"[hmmer] loaded {_s(hmm.name)} {_s(hmm.accession)} LENG={hmm.M}\n")
    pipeline = pyhmmer.plan7.Pipeline(alphabet)

    grand_total_seqs = 0
    grand_relaxed_hits = 0
    grand_genuine_candidates = 0

    print(f"{'run':<55} {'n':>6} {'E<=1e-4':>8} {'E<=1e-1':>8} {'>=2 Cys':>8} {'strict-regex':>13}")
    for name in RUN_DIRS:
        path = RUNS_ROOT / name / "expA_sae_zincfinger.csv"
        if not path.exists():
            print(f"{name:<55} MISSING {path}")
            continue
        rows = [r for r in csv.DictReader(open(path)) if r["sequence"]]
        n = len(rows)
        seqs = [r["sequence"] for r in rows]
        digital = [TextSequence(name=str(i).encode(), sequence=s).digitize(alphabet)
                   for i, s in enumerate(seqs)]
        block = DigitalSequenceBlock(alphabet, digital)

        raw_hits = {}
        for hit in pipeline.search_hmm(hmm, block):
            raw_hits[int(_s(hit.name))] = hit.evalue

        primary, relaxed, genuine = 0, 0, 0
        genuine_rows = []
        for i, e_batch in raw_hits.items():
            e_single = e_batch / n   # rescale batch E-value -> single-query-equivalent
            if e_single > RELAXED_E:
                continue
            relaxed += 1
            if e_single <= PRIMARY_E:
                primary += 1
            n_cys = seqs[i].count("C")
            if n_cys >= 2:
                genuine += 1
                genuine_rows.append((i, e_single, n_cys))

        strict_hits = sum(1 for r in rows if float(r.get("concept_count_strict", 0) or 0) > 0)

        print(f"{name:<55} {n:>6} {primary:>8} {relaxed:>8} {genuine:>8} {strict_hits:>13}")
        for i, e_single, n_cys in genuine_rows:
            r = rows[i]
            print(f"    -> candidate row {i}: E(single-query-equiv)={e_single:.2g} "
                  f"n_Cys={n_cys} alpha={r['alpha']} corrected={r['corrected']} "
                  f"seq={r['sequence']}")

        grand_total_seqs += n
        grand_relaxed_hits += relaxed
        grand_genuine_candidates += genuine

    print(f"\nTOTAL across all zinc-finger runs: {grand_total_seqs} sequences, "
          f"{grand_relaxed_hits} pass ZnF-Prot's relaxed E<=1e-1 Pfam cutoff, "
          f"{grand_genuine_candidates} of those also clear the minimal >=2-cysteine "
          "sanity check (necessary, not sufficient, for a real C2H2 motif).")


if __name__ == "__main__":
    main()
