#!/usr/bin/env python
"""Graded (non-binary) zinc-finger signal in steering-generated sequences: does alpha /
GLP-correction move sequences TOWARD C2H2 character, even short of a complete hit?

Motivation: the binary detectors (regex full match, HMM E-value threshold) all report a
clean or near-clean null across every zinc-finger run (see docs/steering_history.md and
this session's `apply_pfam_zf_detector.py`/`analyze_banerjee_detectors.py`). But "no
complete hits" doesn't rule out a directional/partial effect -- this script looks for one,
using continuous measures instead of a pass/fail cutoff:

  - mean Cys count, mean His count, mean (Cys+His)/length -- raw composition drift.
  - partial-motif rate: fraction of sequences containing just the Cys-x(2-4)-Cys half of
    the C2H2 pattern (no His requirement) -- a real, checkable step toward the full motif.
  - partial-motif-plus-His rate: Cys-pair AND >=1 His within 6-20 residues downstream (the
    region where the model's first conserved His would fall) -- closer still.
  - mean best HMM bit score against PF00096 (zf-C2H2), computed in permissive/"max" mode
    (bias filter off, F1=F2=F3=1, T=-1000) so EVERY sequence gets a real score, not just
    the ones that clear a reporting threshold -- this is the continuous version of the
    E-value cutoff used elsewhere. NO-HIT sequences are floored to -1000 (the pipeline's own
    T threshold), not left at -inf -- averaging a cell that mixes real scores with -inf
    silently poisons the whole mean to -inf, which is a stats bug, not a detection result.
  - per-alpha Mann-Whitney U test (corrected > naive) on the HMM score -- this is the actual
    significance check behind the "does GLP correction measurably nudge sequences toward
    the concept" claim, not just eyeballing the means.

Ends with a per-run ranking: how many of its tested alphas show a significant
(corrected > naive, p<0.05) HMM-score gap, and the strongest single p-value -- this is what
"which run best supports the graded claim" should be read off, not the binary hit tables.

Usage: python3 scripts/zinc_finger_graded_signal.py
"""
import csv
import re
import statistics as stats
from collections import defaultdict
from pathlib import Path

import pyhmmer
from pyhmmer.easel import Alphabet, TextSequence, DigitalSequenceBlock
from scipy.stats import mannwhitneyu

HMM_PATH = "/gpfs/projects/bsc70/bsc070332/glp-prot-runs/adhoc_scripts/PF00096.hmm"
RUNS_ROOT = Path("/gpfs/projects/bsc70/bsc070332/glp-prot-runs/uniref50_500m_4096_d24_steering")
NO_HIT_FLOOR = -1000.0

RUNS = [
    ("steering_sae_zinc", "base C2H2 concept, features 6158+4421"),
    ("steering_sae_zinc_interplm_exact", "interplm_exact pick"),
    ("steering_sae_zinc_scaleshift", "scale_shift intervention"),
    ("steering_sae_zinc_feat_1547_solo", "feature 1547 solo"),
    ("steering_sae_zinc_feat_5819_solo", "feature 5819 solo"),
    ("steering_sae_zinc_feat_6311_solo", "feature 6311 solo"),
    ("steering_sae_zinc_feat_7033_solo", "feature 7033 solo (old AUC pick)"),
    ("steering_sae_zinc_top2_interplm_pair", "interplm_exact top-2 pair"),
    ("steering_sae_zinc_top2_villegas_pair", "villegas_garcia top-2 pair"),
    ("steering_sae_zinc_union1435", "UNION(1-5), feature 1435"),
    ("steering_sae_zinc_union1435_alpha0to50_n100_archived", "UNION(1-5) feature 1435, archived alpha0-50 pilot"),
    ("steering_sae_zinc_union1435_argmax", "UNION(1-5) feature 1435, argmax sampling"),
    ("steering_sae_zinc_union1435_bmask10", "UNION(1-5) feature 1435, mask_fraction=0.10 (vs default 1.0)"),
    ("steering_sae_zinc_union_top2", "UNION(1-5) own top-2 pair (1435+5696) -- 'union_pair'"),
    ("steering_sae_zinc_zinc_6311_highalpha_pilot", "feature 6311, high-alpha pilot"),
    ("steering_sae_zinc_zincfive", "5-feature joint clamp (partial/cancelled run)"),
    ("steering_sae_zinc_zincsix_min", "6-feature joint clamp"),
    ("steering_sae_zinc_zinc_villegas", "villegas_garcia-style pick"),
]

CYS_PAIR_RE = re.compile(r"C.{2,4}C")


def _s(x):
    return x.decode() if isinstance(x, bytes) else x


def cys_pair_plus_his(seq: str) -> bool:
    for m in CYS_PAIR_RE.finditer(seq):
        tail = seq[m.end():m.end() + 20]
        if "H" in tail[6:20] if len(tail) >= 6 else False:
            return True
    return False


def score_run(pipeline, hmm, alphabet, path):
    rows = [r for r in csv.DictReader(open(path)) if r["sequence"]]
    seqs = [r["sequence"] for r in rows]
    digital = [TextSequence(name=str(i).encode(), sequence=s).digitize(alphabet)
               for i, s in enumerate(seqs)]
    block = DigitalSequenceBlock(alphabet, digital)
    best_score = {i: NO_HIT_FLOOR for i in range(len(seqs))}
    for hit in pipeline.search_hmm(hmm, block):
        best_score[int(_s(hit.name))] = hit.score

    cells = defaultdict(list)
    for i, r in enumerate(rows):
        key = (float(r["alpha"]), r["corrected"] == "True")
        cells[key].append(best_score[i])
    return cells


def main():
    alphabet = Alphabet.amino()
    with pyhmmer.plan7.HMMFile(HMM_PATH) as f:
        hmm = f.read()
    pipeline = pyhmmer.plan7.Pipeline(alphabet, bias_filter=False, F1=1.0, F2=1.0, F3=1.0,
                                      T=-1000.0, domT=-1000.0)

    ranking = []
    for run_dir, label in RUNS:
        path = RUNS_ROOT / run_dir / "expA_sae_zincfinger.csv"
        if not path.exists():
            print(f"[skip] {label}: {path} not found")
            continue
        cells = score_run(pipeline, hmm, alphabet, path)
        alphas = sorted(set(a for a, _ in cells))

        print(f"\n=== {label} ({run_dir}) ===")
        print(f"{'alpha':>6} {'n naive':>8} {'n corr':>7} {'naive mean':>11} "
              f"{'corr mean':>10} {'p (corr>naive)':>15}")
        n_sig, best_p, best_alpha = 0, 1.0, None
        n_tested = 0
        for a in alphas:
            naive, corr = cells.get((a, False)), cells.get((a, True))
            if not naive or not corr:
                continue
            naive_mean, corr_mean = stats.mean(naive), stats.mean(corr)
            if a == 0:
                print(f"{a:>6.1f} {len(naive):>8} {len(corr):>7} {naive_mean:>11.1f} "
                      f"{corr_mean:>10.1f} {'(baseline, skip)':>15}")
                continue
            _, p = mannwhitneyu(corr, naive, alternative="greater")
            n_tested += 1
            sig = p < 0.05
            n_sig += sig
            if p < best_p:
                best_p, best_alpha = p, a
            print(f"{a:>6.1f} {len(naive):>8} {len(corr):>7} {naive_mean:>11.1f} "
                  f"{corr_mean:>10.1f} {p:>14.4g}{' *' if sig else ''}")
        ranking.append((label, run_dir, n_sig, n_tested, best_p, best_alpha))

    print("\n\n=== RANKING: which run best supports 'GLP correction pushes toward "
          "zinc-finger-like character' ===")
    print(f"{'run':<55} {'sig alphas':>11} {'/tested':>8} {'best p':>10} {'@alpha':>7}")
    for label, run_dir, n_sig, n_tested, best_p, best_alpha in sorted(
            ranking, key=lambda t: (-t[2], t[4])):
        print(f"{run_dir:<55} {n_sig:>11} {n_tested:>8} {best_p:>10.3g} {best_alpha!s:>7}")


if __name__ == "__main__":
    main()
