#!/usr/bin/env python
"""Extract the unlabeled GLP training activation store (PLAN §11, W1 Step 4).

Pull a small Swiss-Prot slice via UniProt REST (no UniRef50 for the PoC), embed at the
hooked layer with the InterPLM-wrapping extractor, write fp16 memmap shards, and fit
the standardizer. Output: cfg.output_dir/train_store/{activations.dat, meta.json,
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
    fasta = out / "swissprot" / f"train_slice_{args.n_seqs}.fasta"
    store_dir = out / f"{args.store}_store"

    # Training must not contain the proteins we later probe on: the two REST queries
    # otherwise overlap ~98%, giving GLP activations it has already fit while the SAE
    # (trained on UniRef50) gets no such advantage.
    exclude = sl.read_accessions(Path(args.exclude_from)) if args.exclude_from else None
    if exclude:
        print(f"[extract_activations] excluding {len(exclude):,} eval accessions")
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
