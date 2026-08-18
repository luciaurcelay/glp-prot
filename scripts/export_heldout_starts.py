#!/usr/bin/env python
"""Write a contrast's held-out starting proteins as a foldable CSV.

These are the real UniRef50 sequences each steering run generates FROM, and they are the
only correct foldability baseline for that run: they are quartile extremes of the steered
property, not typical proteins, and the two directions' sets differ from each other. For
hydrophobicity the two sets fold at pTM 0.373 and 0.629 — quoting either the pooled average
or a generic reference such as ubiquitin would misstate the comparison by more than the
effect being measured (docs/steering_history.md §13, "baseline discipline").

Each property has its OWN held-out sets, drawn from its own quartiles, so a new property
needs its own baseline folded — the hydrophobicity numbers do not transfer.

Output schema matches the run CSVs so `fold_steering_outputs.py --input` accepts it
directly; `alpha` is the literal string "start" so these rows can never be confused with a
generated cell.

Usage:
  python scripts/export_heldout_starts.py --contrast <property>_contrast.json \
      --out <run-dir>/heldout_starts.csv
  sbatch scripts/cluster/submit_fold.sh <run-dir> --input <run-dir>/heldout_starts.csv \
      --out <run-dir>/heldout_starts_fold.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from glp_plm.eval.steering_properties import get_property, score_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrast", required=True, help="path to a <property>_contrast.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = json.loads(Path(args.contrast).read_text())
    prop = get_property(d.get("property", "gravy"))
    rows = []
    # "up" starts are the LOW-score set (steered +v), "down" the HIGH-score set (-v) —
    # the same convention run_steering.py's build_init uses, so the baseline lines up with
    # the arm it is the baseline for.
    for direction, key in (("up", "heldout"), ("down", "heldout_high")):
        for h in d.get(key) or []:
            seq = h["sequence"]
            rows.append({"sample_id": f"{direction}_{h['id']}", "sequence": seq,
                         "alpha": "start", "corrected": False, "direction": direction,
                         "property": prop.key,
                         **{f"{k}_init": v for k, v in score_all(seq).items()}})
    if not rows:
        raise SystemExit(f"{args.contrast} has no held-out sequences")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_up = sum(r["direction"] == "up" for r in rows)
    print(f"[starts] wrote {out} — {n_up} '{prop.low_name}' (up) + {len(rows) - n_up} "
          f"'{prop.high_name}' (down) natural starting proteins for {prop.key}")


if __name__ == "__main__":
    main()
