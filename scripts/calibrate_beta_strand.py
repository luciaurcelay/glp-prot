#!/usr/bin/env python
"""Calibrate the DSSP-based Beta strand "hit" rule against real Swiss-Prot ground truth,
the way PROSITE was calibrated for zinc-finger (run_steering.py's C2H2_RE_PROSITE
docstring) and pepcoil for Coiled coil (calibrate_pepcoil.py). Neither of those methods
applies here: zinc-finger has a sequence motif, coiled coil has a sequence-only predictor
(pepcoil); Beta strand has neither, so its detector has to look at a FOLDED structure
(DSSP), and the naive whole-chain `strand_frac >= 0.7` rule was already shown to be
useless on short (~25-40 residue) fragments (docs/steering_history.md SS15.2) -- no real
hairpin embedded in a fragment that short reaches 70% pure-E content. This script
replaces the never-empirically-checked ">=2 strand runs (len>=3), gap<=6" rule invented in
that section with a real calibration: fold real Swiss-Prot fragments of the SAME length
distribution used for generation (run_steering.py --length-range, default 25-40) --
positive windows centered on a real annotated Beta strand domain instance, negative
windows with no Beta-strand overlap -- through the SAME pipeline used to score generated
sequences (ESMFold2 + mkdssp, scripts/fold_steering_outputs.py), then sweep candidate
rules and report precision/recall/F1 on held-out (valid=selection, test=report), same
convention as every other detector in this repo.

Two stages (cross-venv, like fold_steering_outputs.py itself):
  1. --build   (project venv, no GPU): sample real positive/negative windows from the
     eval store, write a CSV fold_steering_outputs.py can fold directly + a label
     sidecar JSON (sample_id -> pos/neg/split).
  2. Fold that CSV with ESMFold2 (separate 3.12 venv):
       sbatch scripts/cluster/submit_fold.sh <out-dir> \\
         --input <out-dir>/calibration_sequences.csv --target-ss strand
     (writes <out-dir>/fold_confidence.csv, same as any steering run)
  3. --score    (project venv or plain python3, no GPU): join fold_confidence.csv against
     the label sidecar, sweep candidate rules, report the val/test table, pick the winner.

Usage:
  python scripts/calibrate_beta_strand.py --build \\
    --eval-dir /gpfs/scratch/bsc70/bsc070332/glp-prot-data/uniref50_100m_4096/swissprot/processed \\
    --out-dir runs/uniref50_500m_4096_d24_steering/calibrate_beta_strand \\
    --n-per-split 150
  # ... fold via submit_fold.sh as above ...
  python scripts/calibrate_beta_strand.py --score \\
    --out-dir runs/uniref50_500m_4096_d24_steering/calibrate_beta_strand
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

CONCEPT = "Beta strand"
LAYER = 4


def build(args):
    import _bootstrap  # noqa: F401
    import numpy as np
    from glp_plm.data import swissprot_labels as sl

    processed = Path(args.eval_dir)
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    labels_meta = {}
    sample_id = 0
    for split in ("valid", "test"):
        _, labels_df, names = sl.load_split(processed, split, LAYER, binarize=False)
        assert CONCEPT in names, f"{CONCEPT!r} not in eval store concepts"
        y = labels_df[CONCEPT].to_numpy()
        spans = sl.protein_spans(processed, split)

        # ---- positive windows: centered on a real annotated Beta strand run ----
        # NB: the per-residue "domain-instance index" values are NOT globally unique
        # across the split (they collide/repeat across far-apart proteins -- confirmed
        # by inspection: value 6 in this split's Beta strand column appears at four
        # widely separated positions ~630k residues apart), so matching by raw id would
        # silently span unrelated proteins. Grouping contiguous y>0 runs WITHIN one
        # protein's own slice sidesteps that entirely and needs no global id semantics.
        protein_rows = list(spans.itertuples())
        rng.shuffle(protein_rows)
        n_pos = 0
        for prow in protein_rows:
            if n_pos >= args.n_per_split:
                break
            p_start, p_end = int(prow.start), int(prow.end)
            if p_end - p_start < 25:
                continue
            y_local = y[p_start:p_end]
            pos_idx = np.where(y_local > 0)[0]
            if len(pos_idx) == 0:
                continue
            runs, run_start, prev = [], pos_idx[0], pos_idx[0]
            for k in pos_idx[1:]:
                if k != prev + 1:
                    runs.append((int(run_start), int(prev) + 1))
                    run_start = k
                prev = k
            runs.append((int(run_start), int(prev) + 1))
            rng.shuffle(runs)
            for d_start, d_end in runs:
                if n_pos >= args.n_per_split:
                    break
                d_len = d_end - d_start
                L = rng.randint(25, 40)
                if d_len > L or (p_end - p_start) < L:
                    continue
                slack = L - d_len
                lo = d_start - rng.randint(0, slack)
                lo = max(0, min(lo, (p_end - p_start) - L))
                hi = lo + L
                window_seq = prow.sequence[lo:hi]
                if len(window_seq) != L:
                    continue
                rows.append({"sample_id": sample_id, "sequence": window_seq, "alpha": 0,
                            "corrected": False})
                labels_meta[str(sample_id)] = {"split": split, "label": "pos",
                                               "protein": prow.accession}
                sample_id += 1
                n_pos += 1

        # ---- negative windows: no Beta-strand overlap anywhere in the window ----
        neg_pos_mask = (y > 0)
        n_neg = 0
        attempts = 0
        while n_neg < args.n_per_split and attempts < args.n_per_split * 50:
            attempts += 1
            prow = spans.iloc[rng.randrange(len(spans))]
            p_start, p_end, seq = int(prow.start), int(prow.end), prow.sequence
            if p_end - p_start < 40:
                continue
            L = rng.randint(25, 40)
            lo = rng.randint(p_start, p_end - L)
            hi = lo + L
            if neg_pos_mask[lo:hi].any():
                continue
            window_seq = seq[lo - p_start: hi - p_start]
            if len(window_seq) != L:
                continue
            rows.append({"sample_id": sample_id, "sequence": window_seq, "alpha": 0,
                        "corrected": False})
            labels_meta[str(sample_id)] = {"split": split, "label": "neg",
                                           "protein": prow.accession}
            sample_id += 1
            n_neg += 1
        print(f"[calibrate-beta-strand] {split}: {n_pos} positive, {n_neg} negative "
             f"windows built", flush=True)

    csv_path = out_dir / "calibration_sequences.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "sequence", "alpha", "corrected"])
        w.writeheader()
        w.writerows(rows)
    (out_dir / "calibration_labels.json").write_text(json.dumps(labels_meta, indent=1))
    print(f"[calibrate-beta-strand] wrote {len(rows)} windows to {csv_path}\n"
         f"[calibrate-beta-strand] next: sbatch scripts/cluster/submit_fold.sh {out_dir} "
         f"--input {csv_path} --target-ss strand", flush=True)


# ---- candidate detector rules (pure stdlib, mirrors fold_steering_outputs.py's DSSP
# vocabulary so the winner can be pasted back in verbatim) ----
STRAND = {"E", "B"}


def strand_frac_of(ss: str) -> float:
    return sum(c in STRAND for c in ss) / len(ss) if ss else 0.0


def max_strand_run(ss: str) -> int:
    best = cur = 0
    for c in ss:
        cur = cur + 1 if c in STRAND else 0
        best = max(best, cur)
    return best


def strand_runs(ss: str, min_run: int) -> list:
    runs, i, n = [], 0, len(ss)
    while i < n:
        if ss[i] in STRAND:
            j = i
            while j < n and ss[j] in STRAND:
                j += 1
            if j - i >= min_run:
                runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs


def hairpin_hit(ss: str, min_run: int, max_gap: int, min_runs: int = 2) -> bool:
    runs = strand_runs(ss, min_run)
    if len(runs) < min_runs:
        return False
    for a in range(len(runs)):
        for b in range(a + 1, len(runs)):
            if runs[b][0] - runs[a][1] - 1 <= max_gap:
                return True
    return False


def any_run_hit(ss: str, min_run: int) -> bool:
    return len(strand_runs(ss, min_run)) >= 1


def score(preds: dict, labels: dict, split: str):
    tp = fp = fn = tn = 0
    for sid, lab in labels.items():
        if lab["split"] != split:
            continue
        pred = preds.get(sid, False)
        pos = lab["label"] == "pos"
        tp += pred and pos
        fp += pred and not pos
        fn += (not pred) and pos
        tn += (not pred) and not pos
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1,
           "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def run_score(args):
    out_dir = Path(args.out_dir)
    labels = json.loads((out_dir / "calibration_labels.json").read_text())
    fold_path = out_dir / "fold_confidence.csv"
    rows = list(csv.DictReader(open(fold_path)))
    ss_by_id = {r["sample_id"]: r["ss_string"] for r in rows if not r.get("fold_error")
               and not r.get("dssp_error")}
    n_missing = len(rows) - len(ss_by_id)
    if n_missing:
        print(f"[score-beta-strand] {n_missing}/{len(rows)} rows had a fold/DSSP error, "
             "excluded", flush=True)

    candidates = {}
    for t in (0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 0.70):
        candidates[f"frac>={t}"] = lambda ss, t=t: strand_frac_of(ss) >= t
    for min_run in (2, 3, 4, 5):
        candidates[f"any_run>={min_run}"] = lambda ss, m=min_run: any_run_hit(ss, m)
    for min_run in (2, 3):
        for max_gap in (2, 4, 6, 8, 10, 14):
            candidates[f"hairpin(run>={min_run},gap<={max_gap})"] = (
                lambda ss, m=min_run, g=max_gap: hairpin_hit(ss, m, g))

    results = []
    for name, fn in candidates.items():
        preds = {sid: fn(ss) for sid, ss in ss_by_id.items()}
        val = score(preds, labels, "valid")
        test = score(preds, labels, "test")
        results.append((name, val, test))
    results.sort(key=lambda r: -r[1]["f1"])  # rank by VAL f1, test is reported not selected-on

    print(f"\n{'detector':<28} {'val P':>7} {'val R':>7} {'val F1':>7}   "
         f"{'test P':>7} {'test R':>7} {'test F1':>7}")
    for name, val, test in results:
        print(f"{name:<28} {val['precision']:>7.3f} {val['recall']:>7.3f} {val['f1']:>7.3f}   "
             f"{test['precision']:>7.3f} {test['recall']:>7.3f} {test['f1']:>7.3f}")

    best_name, best_val, best_test = results[0]
    print(f"\n[score-beta-strand] winner (by val F1): {best_name}  "
         f"val F1={best_val['f1']:.3f}  test F1={best_test['f1']:.3f}")
    (out_dir / "calibration_results.json").write_text(json.dumps(
        {"ranked": [{"name": n, "val": v, "test": t} for n, v, t in results],
         "winner": best_name}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--eval-dir")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-per-split", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.build == args.score:
        raise SystemExit("pass exactly one of --build or --score")
    if args.build:
        build(args)
    else:
        run_score(args)


if __name__ == "__main__":
    main()
