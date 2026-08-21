#!/usr/bin/env python
"""C2H2 PROSITE regex, strict (Banerjee-exact) regex, and the Villegas-Garcia-style Pfam
zf-C2H2 HMM scan, applied to the three steering runs launched this session:

  - steering_sae_zinc_helix_5355_bmask10   (Helix, feature 5355 -- NOT a zinc-finger
    concept; included as a negative control per the user's request, expected near-zero
    on all three zinc detectors)
  - steering_sae_zinc_union1435_bmask10    (UNION(1-5), feature 1435, Banerjee-style
    partial masking, --mask-fraction 0.10)
  - steering_sae_zinc_zinc_any_5696        (Zinc finger_any, feature 5696, new
    interplm_exact pick, alpha={2,4,8})

Regexes are recomputed directly from each row's `sequence` column, NOT read from the CSV's
own concept_count/concept_count_strict fields -- those only reflect a real regex evaluation
when the run used `--concept-detector motif` (the default). helix_5355_bmask10 and
zinc_any_5696 were both launched with `--concept-detector feature_activation`, so their
concept_count_strict column is just a copy of the feature-activation decision, not a regex
result -- recomputing from the raw sequence avoids silently reporting the wrong thing.

HMM E-values are per-query (batch E divided by the run's own sequence count, see
`apply_pfam_zf_detector.py`'s docstring for why batch-wide E-values are ~N times too large).

Usage: python3 scripts/score_new_zinc_runs.py
"""
import csv
import re
from pathlib import Path

import pyhmmer
from pyhmmer.easel import Alphabet, TextSequence, DigitalSequenceBlock

HMM_PATH = "/gpfs/projects/bsc70/bsc070332/glp-prot-runs/adhoc_scripts/PF00096.hmm"
RUNS_ROOT = Path("/gpfs/projects/bsc70/bsc070332/glp-prot-runs/uniref50_500m_4096_d24_steering")

RUNS = [
    ("steering_sae_zinc_helix_5355_bmask10", "Helix, feature 5355 (bmask10) -- NEGATIVE CONTROL, not a zinc concept"),
    ("steering_sae_zinc_union1435_bmask10", "UNION(1-5), feature 1435 (bmask10)"),
    ("steering_sae_zinc_zinc_any_5696", "Zinc finger_any, feature 5696"),
]

C2H2_RE_STRICT = re.compile(r"C.{2,4}C.{12}H.{3,5}H")                       # Banerjee-exact
C2H2_RE_PROSITE = re.compile(r"C.{2,4}C.{3}[LIVMFYWC].{8}H.{3,5}H")         # PROSITE PS00028
PRIMARY_E = 1e-4
RELAXED_E = 1e-1


def _s(x):
    return x.decode() if isinstance(x, bytes) else x


def main():
    alphabet = Alphabet.amino()
    with pyhmmer.plan7.HMMFile(HMM_PATH) as f:
        hmm = f.read()
    pipeline = pyhmmer.plan7.Pipeline(alphabet, bias_filter=False, F1=1.0, F2=1.0, F3=1.0,
                                      T=-1000.0, domT=-1000.0)

    for run_dir, label in RUNS:
        path = RUNS_ROOT / run_dir / "expA_sae_zincfinger.csv"
        if not path.exists():
            print(f"[not ready] {label}: {path} not found (run still generating?)")
            continue
        rows = [r for r in csv.DictReader(open(path)) if r["sequence"]]
        n = len(rows)
        seqs = [r["sequence"] for r in rows]

        prosite_hits = [i for i, s in enumerate(seqs) if C2H2_RE_PROSITE.search(s)]
        strict_hits = [i for i, s in enumerate(seqs) if C2H2_RE_STRICT.search(s)]

        digital = [TextSequence(name=str(i).encode(), sequence=s).digitize(alphabet)
                   for i, s in enumerate(seqs)]
        block = DigitalSequenceBlock(alphabet, digital)
        raw_e = {}
        for hit in pipeline.search_hmm(hmm, block):
            raw_e[int(_s(hit.name))] = hit.evalue
        hmm_primary = [i for i, e in raw_e.items() if e / n <= PRIMARY_E]
        hmm_relaxed = [i for i, e in raw_e.items() if e / n <= RELAXED_E]

        print(f"\n=== {label} ({run_dir}) ===")
        print(f"n={n}  PROSITE hits={len(prosite_hits)}  strict-regex hits={len(strict_hits)}  "
              f"HMM E<=1e-4={len(hmm_primary)}  HMM E<=1e-1={len(hmm_relaxed)}")

        for i in sorted(set(prosite_hits) | set(strict_hits) | set(hmm_relaxed)):
            r = rows[i]
            n_cys = seqs[i].count("C")
            tags = []
            if i in prosite_hits:
                tags.append("PROSITE")
            if i in strict_hits:
                tags.append("strict")
            if i in hmm_relaxed:
                e_single = raw_e[i] / n
                tags.append(f"HMM(E={e_single:.2g})")
            print(f"  row {i}: alpha={r['alpha']} corrected={r['corrected']} n_Cys={n_cys} "
                  f"[{', '.join(tags)}]  seq={seqs[i]}")


if __name__ == "__main__":
    main()
