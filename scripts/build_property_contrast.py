#!/usr/bin/env python
"""Exp B (DiffMean) data construction: contrast sets + held-out starting sequences for
ONE steerable whole-sequence property (glprot_steering_spec.md Part 2B, steps 1-5 — run
first, since Exp B's generation loop in run_steering.py depends on this script's output).

Properties come from `glp_plm.eval.steering_properties`: `gravy` (hydrophobicity),
`charge` (net charge per residue at pH 7) and `instability` (Guruprasad index) — the
three Banerjee et al. steer. This file was `build_gravy_contrast.py` until the other two
were added; the GRAVY path is unchanged apart from the generic key names.

Sample raw UniRef50 sequences (glp_plm.data.swissprot_labels.read_fasta_slice), filter to
~100-200 aa, score each by the property, split into top/bottom quartile, hold out a subset
of EACH quartile as generation starting points (excluded from the vector), and compute the
steering vector as the difference of whole-sequence-mean activations between the two
held-IN sets (Huang et al. arXiv:2509.07983 Eq. 2's own "whole-sequence average activation"
convention — unlike run_steering.py's other diffmean vector, which is residue-level and
label-based for a local structural motif, not applicable here since these properties have
no per-residue label, only a whole-sequence score).

Two things this records that the GRAVY-only version did not, both needed to compare
across properties rather than just within one:

  * **the length confound.** Ranking by a property can rank by length as a side effect,
    and then the diff-of-means vector encodes length. All three scores here are
    length-normalized, which removes the mechanical part, but a residual correlation can
    survive in the data — so the length gap between the two contrast sets is measured,
    reported, and can be eliminated outright with `--match-length` (stratified draw that
    forces the two sets onto the same length distribution).
  * **the activation scale.** `mean_act_norm` (mean ||h|| over the held-out starts) makes
    alpha comparable across properties: the perturbation h -> h + alpha*v rotates h by an
    angle set by alpha*||v||/||h||, so a fixed alpha grid means different things for two
    vectors of different norm. run_steering.py's `--alpha-mode relative` divides it out.

Output: <out-dir>/<property>_contrast.json — vector, quartile stats, set sizes, the natural
between-property correlations on the pool, and the held-out starting sequences
(id/sequence/score). Cached like run_steering.py's own zinc_finger_feature.json (skip
rebuild unless --restart).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import _bootstrap  # noqa: F401
import numpy as np

from glp_plm.config import load_config
from glp_plm.data import swissprot_labels as sl
from glp_plm.data.activation_extractor import ActivationExtractor
from glp_plm.eval.steering_properties import (CANONICAL_AA, PROPERTIES, PROPERTY_KEYS,
                                              get_property, gravy_score, score_all)
from glp_plm.utils import get_device, resolve_shared_dir, set_seed

__all__ = ["build_property_contrast", "sample_property_pool", "gravy_score"]

#: A length gap this large between the two contrast sets is reported as a confound worth
#: killing with --match-length. In units of the pool's own length SD.
LENGTH_GAP_WARN_SD = 0.25


def sample_property_pool(n_raw: int, min_len: int, max_len: int, seed: int
                         ) -> List[Tuple[str, Dict[str, float]]]:
    """Reservoir-sample raw UniRef50 sequences (sl.read_fasta_slice), length-filter, and
    score each on EVERY property (so one scan serves all three, and the natural
    between-property correlations can be reported).

    Deduplicates identical sequences — UniRef50 near-duplicate reps can otherwise land in
    both quartiles' selection pool at small sample sizes. Drops sequences with any
    non-canonical residue code (X/B/Z/U/O, ambiguous or nonstandard AAs UniRef50 does
    contain) — BioPython's scales have no entry for them (KeyError), and run_steering.py's
    masked-Gibbs loop only ever samples/decodes the 20 canonical AA tokens anyway, so a
    held-out sequence with e.g. an 'X' couldn't round-trip through it."""
    fasta_path = resolve_shared_dir("uniref50", "GLP_DATA_ROOT") / "uniref50.fasta.gz"
    seqs = sl.read_fasta_slice(fasta_path, n_seqs=n_raw, seed=seed)
    seen, pool = set(), []
    for s in seqs:
        s = s.strip().upper()
        if not (min_len <= len(s) <= max_len) or s in seen or not set(s) <= CANONICAL_AA:
            continue
        seen.add(s)
        pool.append((s, score_all(s)))
    return pool


def natural_correlations(pool: List[Tuple[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """Pearson r between the three properties (and length) on the length-filtered pool.

    This is the reference an off-target effect has to be read against: charge and
    hydrophobicity are anti-correlated in real proteins, so steering one and seeing the
    other move is only evidence of an unspecific vector if it moves MORE than nature's own
    coupling would predict."""
    keys = PROPERTY_KEYS + ["length"]
    cols = {k: np.array([p[k] if k != "length" else float(len(s)) for s, p in pool])
            for k in keys}
    return {a: {b: float(np.corrcoef(cols[a], cols[b])[0, 1]) for b in keys} for a in keys}


def _length_matched_draw(low_q: List, high_q: List, n_low: int, n_high: int,
                         rng: np.random.Generator, n_bins: int = 10) -> Tuple[List, List]:
    """Draw from both quartiles under a SHARED length distribution.

    Bins lengths, then draws the *same per-bin counts* from both sides. Two details make
    that hold exactly rather than approximately:

      * every bin's capacity is capped at min(available_low, available_high), so a bin can
        never be over-drawn on one side and topped up elsewhere — which would silently put
        the two sides back on different length distributions, i.e. undo the whole point;
      * the returned indices are SHUFFLED. They are assembled bin by bin, and the caller
        splits the draw into held-out starts and held-in contrast members by position, so
        an unshuffled result would hand the held-out set the shortest proteins.
    """
    def lens(q):
        return np.array([len(s) for s, _ in q])

    lo_len, hi_len = lens(low_q), lens(high_q)
    edges = np.linspace(min(lo_len.min(), hi_len.min()),
                        max(lo_len.max(), hi_len.max()) + 1e-6, n_bins + 1)
    lo_bin = np.clip(np.digitize(lo_len, edges) - 1, 0, n_bins - 1)
    hi_bin = np.clip(np.digitize(hi_len, edges) - 1, 0, n_bins - 1)
    lo_idx = [np.flatnonzero(lo_bin == b) for b in range(n_bins)]
    hi_idx = [np.flatnonzero(hi_bin == b) for b in range(n_bins)]

    cap = np.array([min(len(lo_idx[b]), len(hi_idx[b])) for b in range(n_bins)], dtype=int)
    if cap.sum() == 0:
        raise ValueError("no length bin is populated on both sides — cannot length-match")
    n_want = max(n_low, n_high)
    if n_want > cap.sum():
        raise ValueError(
            f"length-matched draw needs {n_want} per side but only {cap.sum()} sequences "
            "sit in bins populated on BOTH sides — raise --n-raw-seqs, widen "
            "--min-len/--max-len, or drop --match-length")
    p = cap / cap.sum()

    take = np.minimum(np.floor(p * n_want).astype(int), cap)
    # top up the rounding shortfall in bins that still have shared capacity, biggest first
    for b in np.argsort(-p):
        if take.sum() >= n_want:
            break
        take[b] = min(take[b] + (n_want - take.sum()), cap[b])

    def draw(idx_by_bin, n_side):
        picked = np.concatenate([rng.choice(idx_by_bin[b], size=take[b], replace=False)
                                 for b in range(n_bins) if take[b]])
        rng.shuffle(picked)
        return list(picked[:n_side])

    return draw(lo_idx, n_low), draw(hi_idx, n_high)


def build_contrast(pool: List[Tuple[str, Dict[str, float]]], prop_key: str, n_per_set: int,
                   n_heldout: int, seed: int, n_heldout_high: int = 0,
                   match_length: bool = False) -> dict:
    """Bottom/top quartile by the property; held-out generation starting points carved out
    of each quartile FIRST (so they never enter the vector), then n_per_set drawn from each
    quartile's remainder.

    `n_heldout` low-score starts drive the "up" steering direction (+alpha*v raises the
    property); `n_heldout_high` high-score starts drive the "down" direction (-alpha*v),
    for bidirectional steering. Both are excluded from their own side of the contrast,
    keeping the two directions symmetric — otherwise the down direction's starting points
    would sit inside the very set that defined v's positive end."""
    pool_sorted = sorted(pool, key=lambda t: t[1][prop_key])
    n = len(pool_sorted)
    q = n // 4
    needed_low = n_per_set + n_heldout
    needed_high = n_per_set + n_heldout_high
    if q < max(needed_low, needed_high):
        raise ValueError(f"quartile size {q} (of {n} length-filtered seqs) can't cover "
                         f"n_per_set={n_per_set} (+{n_heldout} low / +{n_heldout_high} high "
                         f"heldout starts) — raise --n-raw-seqs")
    low_quartile = pool_sorted[:q]
    high_quartile = pool_sorted[n - q:]

    rng = np.random.default_rng(seed)
    if match_length:
        low_idx, high_idx = _length_matched_draw(low_quartile, high_quartile,
                                                 needed_low, needed_high, rng)
    else:
        low_idx = rng.choice(len(low_quartile), size=needed_low, replace=False)
        high_idx = rng.choice(len(high_quartile), size=needed_high, replace=False)

    low_sel = [low_quartile[i] for i in low_idx]
    heldout, low_in = low_sel[:n_heldout], low_sel[n_heldout:]
    high_sel = [high_quartile[i] for i in high_idx]
    # Both draws are in random order (np.random.Generator.choice without replacement, and
    # _length_matched_draw shuffles), so this split is a random partition of the drawn
    # subset — the held-out starts are NOT the most extreme members of their quartile, and
    # the two directions are symmetric by construction.
    heldout_high, high_in = high_sel[:n_heldout_high], high_sel[n_heldout_high:]

    return {"high": high_in, "low_in": low_in,
            "heldout": heldout, "heldout_high": heldout_high}


def whole_seq_mean_acts(extractor: ActivationExtractor, seqs: List[str]) -> np.ndarray:
    """Per-sequence whole-sequence-average activation (mean over residues) -> (n, d)."""
    return np.stack([extractor.extract_single(s).mean(axis=0) for s in seqs])


def assemble_pool(n_raw_seqs: int, min_len: int, max_len: int, seed: int, needed: int,
                  tag: str = "contrast") -> List[Tuple[str, Dict[str, float]]]:
    """Reservoir-sample and length-filter a pool big enough to fill both quartiles.

    Split out of `build_property_contrast` so several properties can share ONE scan: the
    pool does not depend on which property is being contrasted (every sequence is scored on
    all of them), only the quartile split does. Three separate builds would stream 12 GB
    three times for identical data.
    """
    n_raw = n_raw_seqs
    for _ in range(4):
        pool = sample_property_pool(n_raw, min_len, max_len, seed)
        if len(pool) // 4 >= needed:
            return pool
        print(f"[{tag}] only {len(pool)} sequences survived length filter "
              f"[{min_len},{max_len}] from {n_raw} raw draws (need >= {4 * needed} total) "
              "— retrying with a bigger pool")
        n_raw *= 2
    raise RuntimeError("could not assemble a large-enough length-filtered UniRef50 pool "
                       "after 4 retries — raise --n-raw-seqs/n_raw_seqs manually")


def build_property_contrast(cfg, out_path: Path, prop_key: str = "gravy",
                            n_raw_seqs: int = 20000, min_len: int = 100,
                            max_len: int = 200, n_per_set: int = 100, n_heldout: int = 40,
                            seed: int = 0, restart: bool = False, device=None,
                            n_heldout_high: int = 40, match_length: bool = False,
                            pool: Optional[List[Tuple[str, Dict[str, float]]]] = None) -> Path:
    """Core builder, callable directly (run_steering.py's Exp B path calls this to
    lazy-build `<property>_contrast.json` if missing — same auto-build-then-cache
    convention as `select_zinc_finger_features`/`compute_diffmean_vector` there) as well as
    from this script's own CLI below.

    On the defaults: `sample_property_pool` -> `read_fasta_slice` is a reservoir sample
    (Algorithm R), so it streams the ENTIRE ~12GB gzipped UniRef50 once no matter how small
    `n_raw_seqs` is. Two consequences drive the numbers here:

      * A too-small pool triggers the retry loop below, and each retry is another full scan.
        So the first pass is deliberately over-provisioned (n_raw_seqs=20000): a bigger
        reservoir costs only memory and some scoring, whereas a retry costs a second
        pass over 12GB.
      * The scan cost is fixed, so held-out-set size is essentially free at build time and
        only costs generation later (~2s/sequence). n_heldout=40 per direction rather than
        the spec's 10-20: the earlier n=15 runs left the paired tests underpowered exactly
        where they mattered (low alpha never reached significance, and the degenerate filter
        left as few as 2 usable rows in a high-alpha cell, making those estimates
        survivor-biased). 40 per direction keeps ~all cells usable after filtering while the
        quartile still comfortably covers n_per_set + n_heldout."""
    prop = get_property(prop_key)
    if out_path.exists() and not restart:
        print(f"[contrast/{prop_key}] {out_path} already exists, skipping (--restart to rebuild)")
        return out_path

    device = device or get_device(cfg.device)
    if pool is None:
        pool = assemble_pool(n_raw_seqs, min_len, max_len, seed,
                             needed=n_per_set + max(n_heldout, n_heldout_high),
                             tag=f"contrast/{prop_key}")
    n = len(pool)

    corr = natural_correlations(pool)
    sets = build_contrast(pool, prop_key, n_per_set, n_heldout, seed,
                          n_heldout_high=n_heldout_high, match_length=match_length)
    print(f"[contrast/{prop_key}] pool={n} filtered seqs; high={len(sets['high'])} "
          f"low_in={len(sets['low_in'])} heldout_low={len(sets['heldout'])} "
          f"heldout_high={len(sets['heldout_high'])}")

    # --- length confound -----------------------------------------------------------
    pool_len_sd = float(np.std([len(s) for s, _ in pool])) or 1.0
    len_high = float(np.mean([len(s) for s, _ in sets["high"]]))
    len_low = float(np.mean([len(s) for s, _ in sets["low_in"]]))
    len_gap_sd = (len_high - len_low) / pool_len_sd
    print(f"[contrast/{prop_key}] contrast-set mean length: high={len_high:.1f} "
          f"low={len_low:.1f} (gap {len_gap_sd:+.2f} pool SD"
          f"{', length-matched' if match_length else ''})")
    if not match_length and abs(len_gap_sd) > LENGTH_GAP_WARN_SD:
        print(f"[contrast/{prop_key}] WARNING: the two contrast sets differ in length by "
              f"{len_gap_sd:+.2f} pool SD, so the diff-of-means vector partly encodes "
              "LENGTH rather than the property. Rebuild with --match-length --restart.")

    # Length-matching is not free: if length and the property are confounded in the pool,
    # restricting both sets to a shared length band also narrows the PROPERTY gap, i.e.
    # buys a cleaner vector with a weaker contrast. Print both so the trade is visible in
    # one line rather than inferred by diffing two builds.
    pool_prop_sd = float(np.std([p[prop_key] for _, p in pool], ddof=1)) or 1.0
    prop_gap_sd = (float(np.mean([p[prop_key] for _, p in sets["high"]]))
                   - float(np.mean([p[prop_key] for _, p in sets["low_in"]]))) / pool_prop_sd
    print(f"[contrast/{prop_key}] contrast separation: {prop_gap_sd:+.2f} pool SD of "
          f"{prop_key} (the signal the vector is built from)")

    extractor = ActivationExtractor(cfg.data.model_name, cfg.data.layer, device=device)
    high_means = whole_seq_mean_acts(extractor, [s for s, _ in sets["high"]])
    low_means = whole_seq_mean_acts(extractor, [s for s, _ in sets["low_in"]])
    vector = high_means.mean(axis=0) - low_means.mean(axis=0)

    # ||h|| on the actual generation starting points, per-residue (that is the tensor the
    # intervention edits) — makes alpha convertible to a rotation angle, hence comparable
    # across properties. See run_steering.py --alpha-mode.
    act_norms = [float(np.linalg.norm(extractor.extract_single(s), axis=-1).mean())
                 for s, _ in sets["heldout"] + sets["heldout_high"]]
    mean_act_norm = float(np.mean(act_norms)) if act_norms else float("nan")

    high_scores = [p[prop_key] for _, p in sets["high"]]
    low_scores = [p[prop_key] for _, p in sets["low_in"]]

    def heldout_payload(entries):
        # `score` is the steered property; the rest are the off-target baselines the
        # specificity matrix differences against.
        return [{"id": i, "sequence": s, "score": p[prop_key],
                 **{f"init_{k}": v for k, v in p.items()}}
                for i, (s, p) in enumerate(entries)]

    payload = {
        "property": prop_key,
        "property_label": prop.label,
        "vector": vector.tolist(),
        "vector_norm": float(np.linalg.norm(vector)),
        "mean_act_norm": mean_act_norm,
        "n_high": len(sets["high"]), "n_low_in": len(sets["low_in"]),
        "n_heldout": len(sets["heldout"]), "n_heldout_high": len(sets["heldout_high"]),
        "prop_high_mean": float(np.mean(high_scores)), "prop_high_min": float(np.min(high_scores)),
        "prop_low_in_mean": float(np.mean(low_scores)), "prop_low_in_max": float(np.max(low_scores)),
        "length_high_mean": len_high, "length_low_mean": len_low,
        "length_gap_pool_sd": len_gap_sd, "match_length": match_length,
        "prop_gap_pool_sd": prop_gap_sd,
        "pool_size": n, "natural_correlations": corr,
        # Mean/SD of every property over the whole length-filtered pool. This — not the
        # spread of the held-out starts — is the scale that makes effects on different
        # properties comparable: the held-out sets are drawn from OPPOSITE quartiles of the
        # steered property, so their pooled SD is bimodal and inflated for that property
        # alone, which would shrink exactly the diagonal of the specificity matrix.
        "pool_property_stats": {
            k: {"mean": float(np.mean([p[k] for _, p in pool])),
                "sd": float(np.std([p[k] for _, p in pool], ddof=1))}
            for k in PROPERTY_KEYS},
        "min_len": min_len, "max_len": max_len, "seed": seed,
        "heldout": heldout_payload(sets["heldout"]),
        # high-score starting points for the "down" steering direction (-alpha*v). Ids are
        # numbered independently of the low set; run_steering.py keys pairing on
        # (direction, init_sequence_id), so the two id spaces never collide.
        "heldout_high": heldout_payload(sets["heldout_high"]),
    }
    if prop_key == "gravy":
        # Back-compat with the pre-generalization schema so already-written analysis
        # scripts and the archived runs/steering_crg/* contrasts stay interchangeable.
        payload.update(gravy_high_mean=payload["prop_high_mean"],
                       gravy_high_min=payload["prop_high_min"],
                       gravy_low_in_mean=payload["prop_low_in_mean"],
                       gravy_low_in_max=payload["prop_low_in_max"])
        for entry in payload["heldout"] + payload["heldout_high"]:
            entry["gravy"] = entry["score"]
    if sets["heldout_high"]:
        hs = [p[prop_key] for _, p in sets["heldout_high"]]
        payload["prop_heldout_high_mean"] = float(np.mean(hs))
        payload["prop_heldout_high_min"] = float(np.min(hs))
    if sets["heldout"]:
        ls = [p[prop_key] for _, p in sets["heldout"]]
        payload["prop_heldout_low_mean"] = float(np.mean(ls))
        payload["prop_heldout_low_max"] = float(np.max(ls))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[contrast/{prop_key}] wrote {out_path} (||v||={payload['vector_norm']:.4f}, "
          f"mean||h||={mean_act_norm:.4f}, {prop.label} high={payload['prop_high_mean']:.4f} "
          f"vs low={payload['prop_low_in_mean']:.4f}, {payload['n_heldout']}+"
          f"{payload['n_heldout_high']} held out)")
    return out_path


# Kept so `from build_gravy_contrast import build_gravy_contrast` style call sites and any
# stale notebook keep working; the GRAVY path is byte-identical to the generic one.
def build_gravy_contrast(cfg, out_path: Path, **kw) -> Path:
    kw.pop("prop_key", None)
    return build_property_contrast(cfg, out_path, prop_key="gravy", **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/configs_lucia/esm8m_500m_4096_d24.yaml")
    ap.add_argument("--property", default="gravy",
                    help="which property to build a contrast for: one of "
                         f"{sorted(PROPERTIES)}, a comma-separated list, or 'all'. Several "
                         "properties share ONE UniRef50 scan and therefore one pool, so "
                         "their natural-correlation references are identical rather than "
                         "merely equal in distribution.")
    ap.add_argument("--out-dir", help="default <output_dir>/steering_diffmean_<leaf>")
    ap.add_argument("--n-raw-seqs", type=int, default=20000,
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
    ap.add_argument("--n-heldout", type=int, default=40,
                    help="low-score sequences carved out of the low quartile as the 'up' "
                         "direction's generation starting points; excluded from the vector. "
                         "Above the spec's 10-20 range on purpose — see the docstring: at "
                         "n=15 the paired tests were underpowered where it mattered and "
                         "high-alpha cells were survivor-biased after the degenerate filter")
    ap.add_argument("--n-heldout-high", type=int, default=40,
                    help="high-score sequences carved out of the high quartile as starting "
                         "points for the DOWN steering direction (bidirectional steering); "
                         "excluded from the vector. 0 disables the down direction.")
    ap.add_argument("--match-length", action="store_true",
                    help="force the two contrast sets onto a shared length distribution "
                         "(stratified draw). Use when the build reports a length gap — "
                         "otherwise the diff-of-means vector partly encodes length.")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed if args.seed is not None else cfg.seed)

    keys = (PROPERTY_KEYS if args.property.strip() == "all"
            else [k.strip() for k in args.property.split(",") if k.strip()])
    unknown = [k for k in keys if k not in PROPERTIES]
    if unknown:
        ap.error(f"unknown propert{'y' if len(unknown) == 1 else 'ies'} {unknown}; "
                 f"choose from {sorted(PROPERTIES)} (or 'all')")
    if args.out_dir and len(keys) > 1:
        ap.error("--out-dir names one directory but several properties were requested; "
                 "each needs its own out-dir. Drop --out-dir to use the per-property "
                 "default, or build them one at a time.")

    targets = []
    for k in keys:
        out_dir = (Path(args.out_dir) if args.out_dir
                   else Path(cfg.output_dir) / f"steering_diffmean_{get_property(k).leaf}")
        targets.append((k, out_dir / f"{k}_contrast.json"))

    # Only scan UniRef50 if at least one target actually needs building.
    todo = [(k, p) for k, p in targets if args.restart or not p.exists()]
    pool = None
    if len(todo) > 1:
        pool = assemble_pool(args.n_raw_seqs, args.min_len, args.max_len, args.seed,
                             needed=args.n_per_set + max(args.n_heldout, args.n_heldout_high),
                             tag="contrast/shared")
        print(f"[contrast] one shared pool of {len(pool)} sequences for "
              f"{[k for k, _ in todo]}")

    for k, out_path in targets:
        build_property_contrast(cfg, out_path, prop_key=k, n_raw_seqs=args.n_raw_seqs,
                                min_len=args.min_len, max_len=args.max_len,
                                n_per_set=args.n_per_set, n_heldout=args.n_heldout,
                                seed=args.seed, restart=args.restart,
                                n_heldout_high=args.n_heldout_high,
                                match_length=args.match_length, pool=pool)

    print("\n[contrast] alpha scale per property — --alpha-mode relative divides by this, "
          "so pick ONE shared --alphas grid from these numbers:")
    for k, out_path in targets:
        if not out_path.exists():
            continue
        d = json.loads(out_path.read_text())
        scale = d["mean_act_norm"] / d["vector_norm"]
        print(f"  {k:>12}: ||v||={d['vector_norm']:.4f}  mean||h||={d['mean_act_norm']:.4f}  "
              f"raw alpha = relative alpha x {scale:.3f}  "
              f"(length gap {d.get('length_gap_pool_sd', float('nan')):+.2f} SD, "
              f"contrast {d.get('prop_gap_pool_sd', float('nan')):+.2f} SD)")


if __name__ == "__main__":
    main()
