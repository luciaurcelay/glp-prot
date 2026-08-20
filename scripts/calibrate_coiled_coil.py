#!/usr/bin/env python
"""Calibrate and F1-verify the sequence-only heptad-repeat coiled-coil scorer
(glp_plm.eval.coiled_coil) against real Swiss-Prot "Coiled coil" labels, using the
SAME val-then-test F1 protocol as every SAE/GLP feature in compare_feature_methods.py
(glp_plm.eval.probing_f1.run_concept_f1) -- so the reported F1 is directly comparable
to the SAE/GLP feature F1s already in compare_feature_methods_ss.json, not a different
metric invented for this one detector.

Needs no SAE, no GLP checkpoint, no GPU -- the scorer runs on the raw amino acid
sequence. Only touches the Swiss-Prot label store + protein_data.tsv sequences (skips
the (n_aa, 320) activation arrays load_split() would otherwise load), so this runs
fine directly on a login node in well under a minute.

Usage:
  python scripts/calibrate_coiled_coil.py \\
    --eval-dir /gpfs/scratch/bsc70/bsc070332/glp-prot-data/uniref50_100m_4096/swissprot/processed \\
    [--out-dir runs/uniref50_500m_4096_d24_steering]

Also rescores the already-generated steering sequences (expA_sae_zincfinger.csv, no
folding needed) if --steering-csv is given, writing a `coiled_coil_score` /
`coiled_coil_hit` column sidecar next to the input CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
from scipy import sparse

from glp_plm.data.swissprot_labels import protein_spans, split_shards
from glp_plm.eval import probing_f1 as pf
from glp_plm.eval.coiled_coil import coiled_coil_profile, coiled_coil_score

CONCEPT = "Coiled coil"


def load_labels_only(processed_dir: Path, split: str, n_shards):
    """Same label-loading as swissprot_labels.load_split, minus the (n_aa, 320)
    activation array -- we don't need activations at all here, and skipping that
    np.load saves ~1.5GB/split, keeping this login-node-safe."""
    split_dir = processed_dir / split
    meta = json.loads((split_dir / "metadata.json").read_text())
    kept = meta["indices_of_concepts_to_keep"]
    concept_names = (split_dir / "aa_concepts_columns.txt").read_text().splitlines()
    shard_ids = split_shards(processed_dir, split, n_shards)
    labels = []
    for i in shard_ids:
        shard = processed_dir / f"shard_{i}"
        y = sparse.load_npz(shard / "aa_concepts.npz").toarray()[:, kept]
        labels.append(y)
    labels = np.concatenate(labels, axis=0)
    return labels, concept_names


def sequence_score_array(processed_dir: Path, split: str, n_shards, window: int) -> np.ndarray:
    """Per-residue coiled_coil_profile, concatenated in the SAME row order as
    load_labels_only / load_split (both walk shard_source -> protein_data.tsv rows
    in order -- see protein_spans' docstring for the alignment contract)."""
    spans = protein_spans(processed_dir, split, n_shards)
    n = int(spans["end"].iloc[-1]) if len(spans) else 0
    out = np.empty(n, dtype=np.float64)
    for r in spans.itertuples():
        out[int(r.start):int(r.end)] = coiled_coil_profile(r.sequence, window)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--n-shards-per-split", type=int)
    ap.add_argument("--window", type=int, default=28)
    ap.add_argument("--out-dir", help="if given, writes calibrate_coiled_coil.json here")
    ap.add_argument("--steering-csv", action="append", default=[],
                    help="expA_sae_zincfinger.csv to rescore with the calibrated "
                         "threshold (repeatable). Writes <csv-dir>/coiled_coil_rescore.csv")
    args = ap.parse_args()

    processed = Path(args.eval_dir)
    ns = args.n_shards_per_split

    labels_val, names = load_labels_only(processed, "valid", ns)
    labels_test, _ = load_labels_only(processed, "test", ns)
    assert CONCEPT in names, f"{CONCEPT!r} not in eval store concepts: {names}"
    ci = names.index(CONCEPT)
    y_val = labels_val[:, ci].astype(np.int64)
    y_test = labels_test[:, ci].astype(np.int64)
    print(f"[calibrate-cc] valid n_aa={len(y_val)} pos={int((y_val>0).sum())} "
         f"domains={len(np.unique(y_val[y_val>0]))}", flush=True)
    print(f"[calibrate-cc] test  n_aa={len(y_test)} pos={int((y_test>0).sum())} "
         f"domains={len(np.unique(y_test[y_test>0]))}", flush=True)

    print("[calibrate-cc] scoring real sequences (heptad scanner, sequence-only)...", flush=True)
    score_val = sequence_score_array(processed, "valid", ns, args.window)
    score_test = sequence_score_array(processed, "test", ns, args.window)

    def bank(split, rows, cols):
        s = score_val if split == "valid" else score_test
        return s[rows][:, None].astype(np.float32)

    result = pf.run_concept_f1(bank, len(y_val), len(y_test), y_val, y_test,
                               cand=np.array([0]), concept=CONCEPT, method="heptad_scanner",
                               two_sided=False)
    print(f"[calibrate-cc] heptad_scanner  val_f1={result.val_f1:.4f}  "
         f"test_f1={result.test_f1:.4f}  precision={result.precision:.4f}  "
         f"recall={result.recall:.4f}  threshold={result.threshold:.4f}  "
         f"rate={result.rate}", flush=True)

    if args.out_dir:
        out = Path(args.out_dir) / "metrics" / "calibrate_coiled_coil.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"concept": CONCEPT, "window": args.window,
                                   **result.to_dict()}, indent=2))
        print(f"[calibrate-cc] wrote {out}", flush=True)

    for csv_path in args.steering_csv:
        csv_path = Path(csv_path)
        rows = list(csv.DictReader(open(csv_path)))
        for row in rows:
            s = coiled_coil_score(row["sequence"], args.window)
            row["coiled_coil_score"] = s
            row["coiled_coil_hit"] = bool(s >= result.threshold)
        out_path = csv_path.parent / "coiled_coil_rescore.csv"
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[calibrate-cc] rescored {len(rows)} rows from {csv_path} -> {out_path} "
             f"(threshold={result.threshold:.4f})", flush=True)


if __name__ == "__main__":
    main()
