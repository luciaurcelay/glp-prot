#!/usr/bin/env python
"""Coiled-coil calls via EMBOSS `pepcoil` (module emboss/6.6.0 on MN5) -- the standard,
peer-reviewed McLachlan & Stewart (1975) / Lupas et al. (1991) coiled-coil algorithm,
not a homebrew reimplementation. Replaces glp_plm.eval.coiled_coil (an ad hoc
physicochemical heptad scorer written for this project, kept only for reference) --
use this module for anything that needs to go in the paper.

Sequence-only: no fold, no GPU. `pepcoil`'s own HitCount/Probability is used as-is
(the tool's own published, calibrated operating point) rather than inventing a new
threshold on top of it.

Usage as a library:
  from run_pepcoil import pepcoil_batch
  results = pepcoil_batch({"id1": "MKQ...", "id2": "GSH..."})
  # -> {"id1": {"hit": True, "max_score": 1.822, "max_probability": 1.0, "n_hits": 1},
  #     "id2": {"hit": False, "max_score": None, "max_probability": 0.0, "n_hits": 0}}

CLI:
  python scripts/run_pepcoil.py --csv path/to/expA_sae_zincfinger.csv \\
    --out path/to/pepcoil_rescore.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict

HIT_RE = re.compile(
    r"\(\d+\) Score ([\d.]+) length \d+ at residues \d+->\d+.*?Probability: ([\d.]+)",
    re.DOTALL,
)
SEQ_HEADER_RE = re.compile(r"# Sequence: (\S+)\s")


def pepcoil_batch(seqs: Dict[str, str], window: int = 28) -> Dict[str, dict]:
    """Run pepcoil once over all sequences (one process, one FASTA) and parse its
    per-sequence report. Empty/duplicate ids are the caller's problem -- pepcoil
    reports by FASTA header verbatim."""
    with tempfile.TemporaryDirectory() as td:
        fasta = Path(td) / "in.fasta"
        out = Path(td) / "out.report"
        with open(fasta, "w") as f:
            for sid, seq in seqs.items():
                f.write(f">{sid}\n{seq}\n")
        subprocess.run(
            ["pepcoil", "-sequence", str(fasta), "-outfile", str(out),
             "-window", str(window), "-auto"],
            check=True, capture_output=True, text=True,
        )
        text = out.read_text()

    # Split into per-sequence blocks on the "# Sequence: <id>" header.
    blocks = re.split(r"(?=# Sequence: )", text)
    results = {sid: {"hit": False, "max_score": None, "max_probability": 0.0, "n_hits": 0}
              for sid in seqs}
    for block in blocks:
        m = SEQ_HEADER_RE.search(block)
        if not m:
            continue
        sid = m.group(1)
        hits = HIT_RE.findall(block)
        if hits:
            scores = [float(s) for s, _ in hits]
            probs = [float(p) for _, p in hits]
            results[sid] = {"hit": True, "max_score": max(scores),
                           "max_probability": max(probs), "n_hits": len(hits)}
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with 'sample_id' and 'sequence' columns")
    ap.add_argument("--out", help="default: <csv-dir>/pepcoil_rescore.csv")
    ap.add_argument("--window", type=int, default=28)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    rows = list(csv.DictReader(open(csv_path)))
    seqs = {row["sample_id"]: row["sequence"] for row in rows}
    print(f"[pepcoil] scoring {len(seqs)} sequences from {csv_path}", flush=True)
    results = pepcoil_batch(seqs, args.window)

    for row in rows:
        r = results[row["sample_id"]]
        row["pepcoil_hit"] = r["hit"]
        row["pepcoil_max_score"] = r["max_score"] if r["max_score"] is not None else ""
        row["pepcoil_max_probability"] = r["max_probability"]
        row["pepcoil_n_hits"] = r["n_hits"]

    out_path = Path(args.out) if args.out else csv_path.parent / "pepcoil_rescore.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_hit = sum(1 for row in rows if row["pepcoil_hit"])
    print(f"[pepcoil] {n_hit}/{len(rows)} hit -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
