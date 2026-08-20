#!/usr/bin/env python
"""F1-verify EMBOSS pepcoil (McLachlan-Stewart/Lupas coiled-coil algorithm, the
standard tool -- see scripts/run_pepcoil.py) against real Swiss-Prot "Coiled coil"
labels, at the protein level pepcoil is actually meant to be used at: run it on every
real protein in valid/test, turn its predicted coil region(s) into a per-residue 0/1
call, and score against the same per-residue domain-instance labels every other
concept detector in this repo is scored against (recall-per-domain, matching
glp_plm.eval.probing_f1's convention). pepcoil's own HitCount decision is used as-is
(the tool's own calibrated operating point) -- no extra threshold invented here.

Needs no SAE, no GLP, no GPU -- sequence-only. Runs pepcoil once in a single batched
FASTA call over every valid/test protein (fast: a lightweight EMBOSS C tool).

Usage:
  module load emboss/6.6.0
  python scripts/calibrate_pepcoil.py \\
    --eval-dir /gpfs/scratch/bsc70/bsc070332/glp-prot-data/uniref50_100m_4096/swissprot/processed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from calibrate_coiled_coil import load_labels_only
from glp_plm.data.swissprot_labels import protein_spans, split_shards
from run_pepcoil import pepcoil_batch

CONCEPT = "Coiled coil"


def score_split(processed_dir: Path, split: str, y: np.ndarray, n_shards, window: int) -> dict:
    spans = protein_spans(processed_dir, split, n_shards)
    seqs = {f"{split}_{i}": row.sequence for i, row in enumerate(spans.itertuples())}
    print(f"[calibrate-pepcoil] {split}: running pepcoil on {len(seqs)} real proteins...",
         flush=True)
    results = pepcoil_batch(seqs, window)

    pred = np.zeros(len(y), dtype=bool)
    for i, row in enumerate(spans.itertuples()):
        if results[f"{split}_{i}"]["hit"]:
            pred[int(row.start):int(row.end)] = True

    pos = y > 0
    tp = int((pred & pos).sum())
    fp = int((pred & ~pos).sum())
    n_pos = int(pos.sum())
    domains_all = set(np.unique(y[pos]).tolist())
    domains_hit = set(np.unique(y[pred & pos]).tolist())
    precision = tp / max(tp + fp, 1)
    recall = len(domains_hit) / max(len(domains_all), 1)   # recall-per-domain, project convention
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall_per_domain": recall, "f1": f1,
           "tp_aa": tp, "fp_aa": fp, "n_pos_aa": n_pos,
           "n_domains": len(domains_all), "n_domains_hit": len(domains_hit),
           "n_proteins": len(seqs), "n_proteins_flagged": sum(r["hit"] for r in results.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--n-shards-per-split", type=int)
    ap.add_argument("--window", type=int, default=28)
    ap.add_argument("--out-dir")
    args = ap.parse_args()

    processed = Path(args.eval_dir)
    ns = args.n_shards_per_split
    labels_val, names = load_labels_only(processed, "valid", ns)
    labels_test, _ = load_labels_only(processed, "test", ns)
    assert CONCEPT in names
    ci = names.index(CONCEPT)
    y_val = labels_val[:, ci].astype(np.int64)
    y_test = labels_test[:, ci].astype(np.int64)

    val_result = score_split(processed, "valid", y_val, ns, args.window)
    test_result = score_split(processed, "test", y_test, ns, args.window)
    print(f"[calibrate-pepcoil] VALID  precision={val_result['precision']:.4f} "
         f"recall_per_domain={val_result['recall_per_domain']:.4f} f1={val_result['f1']:.4f} "
         f"({val_result['n_proteins_flagged']}/{val_result['n_proteins']} proteins flagged)",
         flush=True)
    print(f"[calibrate-pepcoil] TEST   precision={test_result['precision']:.4f} "
         f"recall_per_domain={test_result['recall_per_domain']:.4f} f1={test_result['f1']:.4f} "
         f"({test_result['n_proteins_flagged']}/{test_result['n_proteins']} proteins flagged)",
         flush=True)

    if args.out_dir:
        out = Path(args.out_dir) / "metrics" / "calibrate_pepcoil.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"concept": CONCEPT, "window": args.window,
                                   "valid": val_result, "test": test_result}, indent=2))
        print(f"[calibrate-pepcoil] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
