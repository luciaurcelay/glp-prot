#!/usr/bin/env python
"""Prefetch Swiss-Prot (+ optional UniRef50) data (PLAN §11).

Data counterpart of scripts/cluster/prefetch_weights.sh (model/SAE weights — those
land in the HF cache under $GLP_RUNS_ROOT/hf_cache, durable project storage). This
script's downloads are large but regenerable, so they go under $GLP_DATA_ROOT
instead (scratch — see scripts/cluster/cluster.env). Run on a LOGIN node — it needs
internet access, compute nodes are not assumed to have any (scripts/cluster/README.md)
— to warm the cache before submitting the actual pipeline jobs. Downloads:
  - the labeled Swiss-Prot eval slice (same REST query build_swissprot_eval.py uses)
    -> $GLP_DATA_ROOT/swissprot/swissprot_annotated_eval_<n_prot>.tsv
  - the unlabeled Swiss-Prot training slice (same REST query extract_activations.py
    uses), excluding every eval accession (avoids train/eval contamination — see
    CLAUDE.md / docs/results/) -> $GLP_DATA_ROOT/swissprot/train_slice_<n_seqs>.fasta
  - optionally, full UniRef50 (~12 GB compressed, --with-uniref50 — an explicit
    go-ahead per CLAUDE.md; not consumed by anything yet, producer_consumer.py is
    still a stub) -> $GLP_DATA_ROOT/uniref50/uniref50.fasta.gz

None of this is per-run: the raw REST pull is identical no matter which run/config
consumes it (same query -> same result), so it's shared, not nested under a run's
output_dir — no --config needed. Match --n-seqs/--n-prot to whatever you'll pass
extract_activations.py / build_swissprot_eval.py so the filenames line up and those
scripts' skip-if-exists check finds this file already fetched (both use
glp_plm.utils.resolve_shared_dir for the same path). Requires $GLP_DATA_ROOT to be
set (source scripts/cluster/env_mn5.sh first).
"""
import argparse
import os
import shutil
import sys
import urllib.request
from pathlib import Path

import _bootstrap  # noqa: F401  (puts repo root + InterPLM on sys.path)
from glp_plm.data import swissprot_labels as sl
from glp_plm.utils import resolve_shared_dir

UNIREF50_URL = "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seqs", type=int, default=200000,
                    help="unlabeled training slice size (match --n-seqs at extract_activations.py time)")
    ap.add_argument("--n-prot", type=int, default=20000,
                    help="labeled eval slice size (match --n-prot at build_swissprot_eval.py time)")
    ap.add_argument("--with-uniref50", action="store_true",
                    help="also fetch full UniRef50 (~12 GB compressed) — ask-first per CLAUDE.md")
    args = ap.parse_args()

    if not os.environ.get("GLP_DATA_ROOT"):
        sys.exit("GLP_DATA_ROOT is not set — source scripts/cluster/env_mn5.sh first.")

    swissprot_dir = resolve_shared_dir("swissprot", "GLP_DATA_ROOT")

    eval_tsv = swissprot_dir / f"swissprot_annotated_eval_{args.n_prot}.tsv"
    sl.fetch_uniprot_annotated_tsv(eval_tsv, n_prot=args.n_prot)
    print(f"[prefetch_data] eval tsv    -> {eval_tsv}")

    # Exclude every eval accession from the training slice — 98.4% train/eval
    # overlap was measured without this (docs/results/).
    exclude = sl.read_accessions(eval_tsv)
    train_fasta = swissprot_dir / f"train_slice_{args.n_seqs}.fasta"
    sl.fetch_uniprot_fasta_slice(train_fasta, n_seqs=args.n_seqs, exclude=exclude)
    print(f"[prefetch_data] train fasta -> {train_fasta} (excluded {len(exclude)} eval accessions)")

    if args.with_uniref50:
        uniref_gz = resolve_shared_dir("uniref50", "GLP_DATA_ROOT") / "uniref50.fasta.gz"
        if uniref_gz.exists() and uniref_gz.stat().st_size > 0:
            print(f"[prefetch_data] uniref50    -> {uniref_gz} (already present, skipping)")
        else:
            print(f"[prefetch_data] uniref50    -> {uniref_gz} (~12 GB compressed, this will take a while)")
            tmp = uniref_gz.with_suffix(".part")
            req = urllib.request.Request(UNIREF50_URL, headers={"User-Agent": "glp-plm/0.1"})
            with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f)
            tmp.rename(uniref_gz)
            print(f"[prefetch_data] uniref50 done: {uniref_gz.stat().st_size / 1e9:.1f} GB")
    else:
        print("[prefetch_data] uniref50    -> skipped (pass --with-uniref50 to fetch; "
              "~12 GB, not consumed by anything yet)")


if __name__ == "__main__":
    main()
