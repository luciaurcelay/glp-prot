#!/usr/bin/env python
"""Exp B (DiffMean x hydrophobicity) data construction: GRAVY contrast sets + held-out
starting sequences (glprot_steering_spec.md Part 2B, steps 1-5 — run first, since Exp
B's generation loop in run_steering.py depends on this script's output).

Sample raw UniRef50 sequences (glp_plm.data.swissprot_labels.read_fasta_slice), filter to
~100-200 aa, score each by BioPython GRAVY, split into top/bottom quartile, hold out a
low-GRAVY subset as generation starting points (excluded from the vector), and compute the
steering vector as the difference of whole-sequence-mean activations between the high-GRAVY
set and the low-GRAVY held-IN set (Huang et al. arXiv:2509.07983 Eq. 2's own "whole-sequence
average activation" convention — unlike run_steering.py's other diffmean vector, which is
residue-level/label-based for a local structural motif, not applicable here since GRAVY has
no per-residue label, only a whole-sequence score).

Output: <out-dir>/gravy_contrast.json - vector, quartile GRAVY stats, set sizes, and the
held-out starting sequences (id/sequence/gravy). Cached like run_steering.py's own
zinc_finger_feature.json/diffmean_vector.json (skip rebuild unless --restart).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import _bootstrap  # noqa: F401
import numpy as np

from glp_plm.config import load_config
from glp_plm.data import swissprot_labels as sl
from glp_plm.data.activation_extractor import ActivationExtractor
from glp_plm.utils import get_device, resolve_shared_dir, set_seed

CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


def gravy_score(seq: str) -> float:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    return float(ProteinAnalysis(seq).gravy())


def sample_gravy_pool(n_raw: int, min_len: int, max_len: int, seed: int
                      ) -> List[Tuple[str, float]]:
    """Reservoir-sample raw UniRef50 sequences (sl.read_fasta_slice), length-filter, score
    by GRAVY. Deduplicates identical sequences — UniRef50 near-duplicate reps can otherwise
    land in both quartiles' selection pool at small sample sizes. Drops sequences with any
    non-canonical residue code (X/B/Z/U/O, ambiguous or nonstandard AAs UniRef50 does
    contain) — BioPython's GRAVY (Kyte-Doolittle) scale has no entry for them (KeyError),
    and run_steering.py's masked-Gibbs loop only ever samples/decodes the 20 canonical AA
    tokens anyway, so a held-out sequence with e.g. an 'X' couldn't round-trip through it."""
    fasta_path = resolve_shared_dir("uniref50", "GLP_DATA_ROOT") / "uniref50.fasta.gz"
    seqs = sl.read_fasta_slice(fasta_path, n_seqs=n_raw, seed=seed)
    seen, pool = set(), []
    for s in seqs:
        s = s.strip().upper()
        if not (min_len <= len(s) <= max_len) or s in seen or not set(s) <= CANONICAL_AA:
            continue
        seen.add(s)
        pool.append((s, gravy_score(s)))
    return pool


def build_contrast(pool: List[Tuple[str, float]], n_per_set: int, n_heldout: int,
                   seed: int) -> dict:
    """Bottom/top quartile by GRAVY; n_heldout held-out low-GRAVY sequences carved out
    first (excluded from the vector), then n_per_set drawn from each quartile's remainder."""
    pool_sorted = sorted(pool, key=lambda t: t[1])
    n = len(pool_sorted)
    q = n // 4
    needed_low = n_per_set + n_heldout
    if q < needed_low or q < n_per_set:
        raise ValueError(f"quartile size {q} (of {n} length-filtered seqs) can't cover "
                         f"n_per_set={n_per_set} (+{n_heldout} heldout for the low set) "
                         f"— raise --n-raw-seqs")
    low_quartile = pool_sorted[:q]
    high_quartile = pool_sorted[n - q:]

    rng = np.random.default_rng(seed)
    low_idx = rng.choice(len(low_quartile), size=needed_low, replace=False)
    low_sel = [low_quartile[i] for i in low_idx]
    heldout, low_in = low_sel[:n_heldout], low_sel[n_heldout:]

    high_idx = rng.choice(len(high_quartile), size=n_per_set, replace=False)
    high_sel = [high_quartile[i] for i in high_idx]

    return {"high": high_sel, "low_in": low_in, "heldout": heldout}


def whole_seq_mean_acts(extractor: ActivationExtractor, seqs: List[str]) -> np.ndarray:
    """Per-sequence whole-sequence-average activation (mean over residues) -> (n, d)."""
    return np.stack([extractor.extract_single(s).mean(axis=0) for s in seqs])


def build_gravy_contrast(cfg, out_path: Path, n_raw_seqs: int = 8000, min_len: int = 100,
                         max_len: int = 200, n_per_set: int = 100, n_heldout: int = 15,
                         seed: int = 0, restart: bool = False, device=None) -> Path:
    """Core builder, callable directly (run_steering.py's Exp B path calls this to
    lazy-build `gravy_contrast.json` if missing — same auto-build-then-cache convention
    as `select_zinc_finger_features`/`compute_diffmean_vector` there) as well as from
    this script's own CLI below."""
    if out_path.exists() and not restart:
        print(f"[build_gravy_contrast] {out_path} already exists, skipping (--restart to rebuild)")
        return out_path

    device = device or get_device(cfg.device)
    n_raw = n_raw_seqs
    pool: Optional[List[Tuple[str, float]]] = None
    n = 0
    for attempt in range(4):
        pool = sample_gravy_pool(n_raw, min_len, max_len, seed)
        needed_low = n_per_set + n_heldout
        n = len(pool)
        if n // 4 >= max(needed_low, n_per_set):
            break
        print(f"[build_gravy_contrast] only {n} sequences survived length filter "
             f"[{min_len},{max_len}] from {n_raw} raw draws (need >= "
             f"{4 * max(needed_low, n_per_set)} total) — retrying with a bigger pool")
        n_raw *= 2
    else:
        raise RuntimeError("could not assemble a large-enough length-filtered UniRef50 pool "
                          "after 4 retries — raise --n-raw-seqs/n_raw_seqs manually")

    sets = build_contrast(pool, n_per_set, n_heldout, seed)
    print(f"[build_gravy_contrast] pool={n} filtered seqs; high={len(sets['high'])} "
         f"low_in={len(sets['low_in'])} heldout={len(sets['heldout'])}")

    extractor = ActivationExtractor(cfg.data.model_name, cfg.data.layer, device=device)
    high_means = whole_seq_mean_acts(extractor, [s for s, _ in sets["high"]])
    low_means = whole_seq_mean_acts(extractor, [s for s, _ in sets["low_in"]])
    vector = high_means.mean(axis=0) - low_means.mean(axis=0)

    high_gravy = [g for _, g in sets["high"]]
    low_gravy = [g for _, g in sets["low_in"]]

    payload = {
        "vector": vector.tolist(),
        "vector_norm": float(np.linalg.norm(vector)),
        "n_high": len(sets["high"]), "n_low_in": len(sets["low_in"]),
        "n_heldout": len(sets["heldout"]),
        "gravy_high_mean": float(np.mean(high_gravy)), "gravy_high_min": float(np.min(high_gravy)),
        "gravy_low_in_mean": float(np.mean(low_gravy)), "gravy_low_in_max": float(np.max(low_gravy)),
        "min_len": min_len, "max_len": max_len, "seed": seed,
        "heldout": [{"id": i, "sequence": s, "gravy": g}
                    for i, (s, g) in enumerate(sets["heldout"])],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[build_gravy_contrast] wrote {out_path} "
         f"(||v||={payload['vector_norm']:.4f}, gravy high={payload['gravy_high_mean']:.3f} "
         f"vs low={payload['gravy_low_in_mean']:.3f}, {payload['n_heldout']} held out)")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/configs_lucia/esm8m_500m_4096_d24.yaml")
    ap.add_argument("--out-dir", help="default <output_dir>/steering_diffmean_hydro")
    ap.add_argument("--n-raw-seqs", type=int, default=8000,
                    help="raw UniRef50 sequences to reservoir-sample before length "
                         "filtering (most get dropped by --min-len/--max-len; doubled "
                         "automatically, up to 4 retries, if the surviving pool can't fill "
                         "both quartiles at --n-per-set)")
    ap.add_argument("--min-len", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=200)
    ap.add_argument("--n-per-set", type=int, default=100,
                    help="sequences per contrast set (Huang et al.'s own validated "
                         "default, spec open item 5), drawn from the length-filtered "
                         "top/bottom quartile")
    ap.add_argument("--n-heldout", type=int, default=15,
                    help="low-GRAVY sequences carved out of the low quartile as Exp B "
                         "generation starting points (spec's 10-20 range); excluded from "
                         "the vector")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed if args.seed is not None else cfg.seed)
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg.output_dir) / "steering_diffmean_hydro"
    out_path = out_dir / "gravy_contrast.json"

    build_gravy_contrast(cfg, out_path, n_raw_seqs=args.n_raw_seqs, min_len=args.min_len,
                         max_len=args.max_len, n_per_set=args.n_per_set,
                         n_heldout=args.n_heldout, seed=args.seed, restart=args.restart)


if __name__ == "__main__":
    main()
