#!/usr/bin/env python
"""Extract the unlabeled GLP training activation store (PLAN §11, W1 Step 4).

Source is `data.train_fasta`: null -> Swiss-Prot REST slice (default); "uniref50" ->
shared local pull (`scripts/prefetch_data.py --with-uniref50`); any other string ->
that path. Embeds at the hooked layer, writes fp16 memmap shards, fits the
standardizer. Output: cfg.output_dir/train_store/{activations.dat, meta.json,
rep_statistics.pt}.
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from glp_plm.config import load_config
from glp_plm.data.activation_extractor import ActivationExtractor
from glp_plm.data.activation_store import ActivationStore
from glp_plm.data import swissprot_labels as sl
from glp_plm.utils import resolve_shared_dir, resolve_storage_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--store", default="train", help="store name (subdir under output_dir)")
    ap.add_argument("--n-seqs", type=int, default=10000, help="sequences to pull for the training store")
    ap.add_argument("--embed-batch-size", type=int, default=16)
    ap.add_argument("--exclude-from", help="UniProt TSV whose accessions to EXCLUDE from "
                                           "training (point at the eval set - see below)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.check_supported()
    out = Path(cfg.output_dir)
    # Data-sized (a fp16 memmap store) — not symlinked back under runs/, reached
    # directly at $GLP_DATA_ROOT/<run>/<store>_store (see resolve_storage_dir docstring).
    store_dir = resolve_storage_dir(out, f"{args.store}_store", "GLP_DATA_ROOT", symlink=False)

    # Training must not contain the proteins we later probe on: the two REST queries
    # otherwise overlap ~98%, giving GLP activations it has already fit while the SAE
    # (trained on UniRef50) gets no such advantage.
    exclude = sl.read_accessions(Path(args.exclude_from)) if args.exclude_from else None
    if exclude:
        print(f"[extract_activations] excluding {len(exclude):,} eval accessions")

    if cfg.data.train_fasta:
        fasta_path = (resolve_shared_dir("uniref50", "GLP_DATA_ROOT") / "uniref50.fasta.gz"
                     if cfg.data.train_fasta == "uniref50" else Path(cfg.data.train_fasta))
        seqs = sl.read_fasta_slice(fasta_path, n_seqs=args.n_seqs, exclude=exclude, seed=cfg.seed)
    else:
        # The raw Swiss-Prot FASTA is identical for any run pulling the same --n-seqs —
        # shared across runs (like UniRef50), not nested per-run.
        fasta = resolve_shared_dir("swissprot", "GLP_DATA_ROOT") / f"train_slice_{args.n_seqs}.fasta"
        sl.fetch_uniprot_fasta_slice(fasta, n_seqs=args.n_seqs, exclude=exclude)
        seqs = sl.read_fasta(fasta)
    print(f"[extract_activations] {len(seqs)} sequences; target ~{cfg.data.n_train_activations:,} activations")

    extractor = ActivationExtractor(cfg.data.model_name, layer=cfg.data.layer, device=cfg.device)
    store = ActivationStore(store_dir, d=cfg.model.activation_dim, dtype=cfg.data.activation_dtype)

    def chunks():
        bs = args.embed_batch_size
        for i in range(0, len(seqs), bs):
            acts = extractor.extract_many(seqs[i:i + bs], batch_size=bs)  # (tokens, d)
            yield acts.detach().cpu().numpy() if torch.is_tensor(acts) else acts

    n = store.write(chunks(), max_rows=cfg.data.n_train_activations)
    scaler = store.fit_scaler()
    print(f"[extract_activations] wrote {n:,} activations to {store_dir}; "
          f"scaler mean|std ~ {scaler.mean.mean():.3f}|{scaler.std.mean():.3f}")


if __name__ == "__main__":
    main()
