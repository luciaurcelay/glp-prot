#!/usr/bin/env python
"""Did instability steering move a DIPEPTIDE property, or just move composition?

Instability is the only one of the three steered properties that is not a function of
amino-acid composition — it is 10/L times a sum of Guruprasad DIWV weights over ADJACENT
pairs. That is why it was included: a purely compositional artifact (the SDEdit
correction's own Leu/Val enrichment, docs/steering_history.md §13.2/§14.5) cannot produce
it directly. But "cannot produce it directly" is not "did not produce it": dipeptide
frequencies are correlated with composition, so a compositional shift moves the index
anyway. This script separates the two.

**The split is exact, not sampled.** For a uniformly random permutation of a sequence with
residue counts n_a (length L), the expected instability index is

    E[I | composition] = (10 / L^2) * sum_{a,b} DIWV(a,b) * n_a * (n_b - [a == b])

because P(x_i = a, x_{i+1} = b) = n_a (n_b - [a==b]) / (L (L-1)) at each of the L-1
adjacencies. So every sequence splits into

    I = E[I | composition]  +  order excess

and the steering-attributable change in I splits the same way. If the whole effect sits in
the first term, instability steering is doing nothing that GRAVY or charge steering does
not; if a real share sits in the second, the direction reached an order-level property.
A shuffle control re-derives the same number empirically, as a check on the algebra.

Everything is **steering-attributable**: each sequence is differenced against the same
starting protein's own alpha=0 run in the same arm and direction, so the correction's own
drift is removed (it is large here — -0.70 pool SD, §14.5) and what is left is steering.

Usage:
  python scripts/analyze_instability_mechanism.py --run-dir <instability run> [--alpha ...]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import _bootstrap  # noqa: F401
import numpy as np

from glp_plm.eval.steering_properties import (find_contrast_json, find_run_csv,
                                              instability_decomposition, instability_score,
                                              run_property)
from glp_plm.experiments import RunLogger, style


def is_degenerate(row: dict) -> bool:
    """Mirrors run_steering.is_degenerate_sequence, so this analyses exactly the row set
    the run's own summary.json reported on."""
    seq = row.get("sequence") or ""
    if seq and max(Counter(seq).values()) / len(seq) >= 0.30:
        return True
    return (int(float(row["unique_aa_count"])) <= 8
            or int(float(row["max_homopolymer_run"])) >= 12)


def expected_instability(seq: str) -> float:
    """E[instability index] over uniformly random permutations of `seq` — closed form.

    This is the part of the index that composition alone determines; `instability_score`
    minus this is the order-specific excess.
    """
    from Bio.SeqUtils import ProtParamData

    diwv = ProtParamData.DIWV
    n = Counter(seq)
    L = len(seq)
    if L < 2:
        return 0.0
    total = 0.0
    for a, na in n.items():
        for b, nb in n.items():
            total += diwv[a][b] * na * (nb - (1 if a == b else 0))
    return 10.0 * total / (L * L)


def shuffled_instability(seq: str, rng: np.random.Generator, n_draws: int = 20) -> float:
    """Empirical mean instability over `n_draws` shuffles — validates the closed form."""
    chars = list(seq)
    vals = []
    for _ in range(n_draws):
        rng.shuffle(chars)
        vals.append(instability_score("".join(chars)))
    return float(np.mean(vals))


def load(run_dir: Path, neutral_alpha: float) -> Tuple[List[dict], object]:
    csv_path = find_run_csv(run_dir)
    raw = list(csv.DictReader(open(csv_path)))
    prop = run_property(raw, csv_path)
    if prop.key != "instability":
        raise SystemExit(f"{csv_path} steered {prop.key!r}, not instability — this analysis "
                         "is about the one target that is not a function of composition")
    rows = []
    for r in raw:
        if is_degenerate(r):
            continue
        seq = r["sequence"]
        exp = expected_instability(seq)
        rows.append({"alpha": float(r["alpha"]), "corrected": r["corrected"] == "True",
                     "direction": r.get("direction") or "up", "init": r["init_sequence_id"],
                     "seq": seq, "total": instability_score(seq), "comp": exp,
                     "order": instability_score(seq) - exp})
    # steering-attributable, per component
    base = {(r["direction"], r["corrected"], r["init"]): r
            for r in rows if r["alpha"] == neutral_alpha}
    for r in rows:
        b = base.get((r["direction"], r["corrected"], r["init"]))
        for k in ("total", "comp", "order"):
            r[f"attr_{k}"] = None if b is None else r[k] - b[k]
    return rows, prop


def cell(rows, direction, corrected, alpha):
    return [r for r in rows if r["direction"] == direction and r["corrected"] == corrected
            and r["alpha"] == alpha]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--neutral-alpha", type=float, default=0.0)
    ap.add_argument("--alpha", type=float, default=None,
                    help="steering strength to decompose (default: the largest in the run, "
                         "where the effect is biggest and the split is easiest to resolve)")
    ap.add_argument("--shuffle-check", type=int, default=20,
                    help="shuffles per sequence for the empirical control (0 to skip). The "
                         "closed form is exact; this only guards the algebra.")
    ap.add_argument("--top-k", type=int, default=15, help="dipeptides listed / plotted")
    ap.add_argument("--no-titles", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    rows, prop = load(run_dir, args.neutral_alpha)
    run = RunLogger(run_dir)
    alphas = sorted({r["alpha"] for r in rows})
    alpha = args.alpha if args.alpha is not None else alphas[-1]
    if alpha not in alphas:
        raise SystemExit(f"alpha={alpha} not in this run ({alphas})")

    sd = None
    cj = find_contrast_json(run_dir, "instability")
    summary = run_dir / "metrics" / "summary.json"
    stats = None
    if cj:
        stats = json.loads(cj.read_text()).get("pool_property_stats")
    if stats is None and summary.exists():
        stats = json.loads(summary.read_text()).get("contrast_pool_property_stats")
    if stats:
        sd = stats["instability"]["sd"]

    print(f"[instability] {run_dir.name}: {len(rows)} clean rows, decomposing alpha={alpha:g} "
          f"(neutral {args.neutral_alpha:g})" + (f", pool SD {sd:.3f}" if sd else ""))

    # ---- 1. how much of the steering effect is composition, how much is order ----
    print("\n1. STEERING-ATTRIBUTABLE CHANGE, SPLIT (mean +- SEM over starting proteins)")
    print(f"{'arm':>10} {'dir':>5} {'n':>4} | {'total':>16} {'composition':>16} {'order excess':>16}"
          f" | {'order share':>12}")
    split: Dict[str, dict] = {}
    for corrected in (True, False):
        for d in ("up", "down"):
            sub = [r for r in cell(rows, d, corrected, alpha) if r["attr_total"] is not None]
            if len(sub) < 3:
                continue
            v = {k: np.array([r[f"attr_{k}"] for r in sub]) for k in ("total", "comp", "order")}
            share = v["order"].mean() / v["total"].mean() if v["total"].mean() else float("nan")
            name = "corrected" if corrected else "naive"
            split[f"{name}/{d}"] = {k: {"mean": float(a.mean()),
                                        "sem": float(a.std(ddof=1) / np.sqrt(len(a)))}
                                    for k, a in v.items()}
            split[f"{name}/{d}"]["order_share"] = float(share)
            fmt = lambda a: f"{a.mean():>+9.2f}+-{a.std(ddof=1)/np.sqrt(len(a)):<5.2f}"
            print(f"{name:>10} {d:>5} {len(sub):>4} | {fmt(v['total']):>16} "
                  f"{fmt(v['comp']):>16} {fmt(v['order']):>16} | {share:>11.0%}")

    # the headline is the divergence, as everywhere else in this experiment
    print("\n2. DIVERGENCE (up - down), SPLIT — the estimand the rest of the paper reports")
    from scipy.stats import mannwhitneyu
    div: Dict[str, dict] = {}
    for corrected in (True, False):
        name = "corrected" if corrected else "naive"
        entry = {}
        for k in ("total", "comp", "order"):
            u = [r[f"attr_{k}"] for r in cell(rows, "up", corrected, alpha)
                 if r[f"attr_{k}"] is not None]
            dn = [r[f"attr_{k}"] for r in cell(rows, "down", corrected, alpha)
                  if r[f"attr_{k}"] is not None]
            if len(u) < 3 or len(dn) < 3:
                continue
            entry[k] = {"divergence": float(np.mean(u) - np.mean(dn)),
                        "p": float(mannwhitneyu(u, dn).pvalue)}
        if entry:
            # A null is only informative with a bound on it: report a CI on the
            # order-specific component and on its share, so "no order effect" means
            # "at most X% of the effect" rather than "we could not detect one".
            for k in ("total", "comp", "order"):
                u = np.array([r[f"attr_{k}"] for r in cell(rows, "up", corrected, alpha)
                              if r[f"attr_{k}"] is not None])
                dn = np.array([r[f"attr_{k}"] for r in cell(rows, "down", corrected, alpha)
                               if r[f"attr_{k}"] is not None])
                se = float(np.sqrt(u.std(ddof=1) ** 2 / len(u) + dn.std(ddof=1) ** 2 / len(dn)))
                entry[k]["se"] = se
                entry[k]["ci95"] = [entry[k]["divergence"] - 1.96 * se,
                                    entry[k]["divergence"] + 1.96 * se]
            div[name] = entry
            t, c, o = entry["total"], entry["comp"], entry["order"]
            lo, hi = (x / t["divergence"] for x in o["ci95"])
            entry["order_share"] = o["divergence"] / t["divergence"]
            entry["order_share_ci95"] = sorted([lo, hi])
            print(f"  {name:>10}: total {t['divergence']:>+8.2f} (p={t['p']:.1e})"
                  + (f" = {t['divergence']/sd:+.2f} SD" if sd else "")
                  + f"  |  composition {c['divergence']:>+8.2f} (p={c['p']:.1e})")
            print(f"  {'':>10}  order {o['divergence']:>+8.2f} "
                  f"[95% CI {o['ci95'][0]:+.2f}, {o['ci95'][1]:+.2f}] (p={o['p']:.1e})"
                  f"  ->  order share {entry['order_share']:.0%} "
                  f"[{min(lo, hi):.0%}, {max(lo, hi):.0%}]")

    # ---- 3. shuffle control on the closed form ----
    if args.shuffle_check:
        rng = np.random.default_rng(0)
        sample = cell(rows, "up", True, alpha)[:12]
        err = [shuffled_instability(r["seq"], rng, args.shuffle_check) - r["comp"]
               for r in sample]
        print(f"\n3. SHUFFLE CONTROL on {len(sample)} sequences x {args.shuffle_check} shuffles: "
              f"empirical minus closed-form = {np.mean(err):+.3f} +- "
              f"{np.std(err, ddof=1)/np.sqrt(len(err)):.3f} (should be 0)")

    # ---- 4. which dipeptides, and how concentrated ----
    up0 = {r["init"]: r for r in cell(rows, "up", True, args.neutral_alpha)}
    upA = [r for r in cell(rows, "up", True, alpha) if r["init"] in up0]
    dec = instability_decomposition([up0[r["init"]]["seq"] for r in upA],
                                    [r["seq"] for r in upA])
    total = sum(v for _, v in dec)
    pos = sorted((v for _, v in dec if v > 0), reverse=True)
    cum = np.cumsum(pos) / sum(pos) if pos else np.array([])
    n50 = int(np.searchsorted(cum, 0.5) + 1) if len(cum) else 0
    n90 = int(np.searchsorted(cum, 0.9) + 1) if len(cum) else 0
    print(f"\n4. DIPEPTIDE CONCENTRATION (corrected/up, alpha={alpha:g} vs its own alpha=0)")
    print(f"   {len(dec)} dipeptides observed, net {total:+.2f} index; of the POSITIVE "
          f"contributions, {n50} account for 50% and {n90} for 90%")
    print("   " + ", ".join(f"{d} {v:+.2f}" for d, v in dec[:args.top_k]))

    run.log_summary(instability_alpha=alpha, instability_split=split,
                    instability_divergence_split=div,
                    instability_dipeptide_top=dict(dec[:args.top_k]),
                    instability_dipeptides_for_50pct=n50,
                    instability_dipeptides_for_90pct=n90)

    # ---- figure ----
    style.apply_style()
    import matplotlib.pyplot as plt

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 4.8),
                                   gridspec_kw={"width_ratios": [1, 1.25]})
    labels, comp_v, ord_v, comp_e, ord_e = [], [], [], [], []
    for key in ("corrected/up", "corrected/down", "naive/up", "naive/down"):
        if key not in split:
            continue
        labels.append(key.replace("/", "\n"))
        comp_v.append(split[key]["comp"]["mean"]); comp_e.append(split[key]["comp"]["sem"])
        ord_v.append(split[key]["order"]["mean"]); ord_e.append(split[key]["order"]["sem"])
    x = np.arange(len(labels)); w = 0.38
    axA.bar(x - w / 2, comp_v, w, yerr=comp_e, capsize=3, color=style.ORANGE_MID,
            edgecolor=style.SURFACE, linewidth=1, label="composition (order-free)")
    axA.bar(x + w / 2, ord_v, w, yerr=ord_e, capsize=3, color=style.PURPLE_LIGHT,
            edgecolor=style.SURFACE, linewidth=1, label="order excess (dipeptide-specific)")
    axA.axhline(0, color=style.INK, linewidth=1)
    axA.set_xticks(x); axA.set_xticklabels(labels, fontsize=9)
    axA.set_ylabel("steering-attributable change\nin instability index (mean ± SEM)")
    axA.legend(fontsize=9)
    if not args.no_titles:
        axA.set_title(f"A  Change at α={alpha:g}, split by what determines it", loc="left")

    top = dec[:args.top_k][::-1]
    axB.barh(range(len(top)), [v for _, v in top], color=style.method_color("glp"), height=0.7)
    axB.axvline(0, color=style.BASELINE, linewidth=1)
    axB.set_yticks(range(len(top)))
    axB.set_yticklabels([d for d, _ in top], fontsize=9, family="monospace")
    axB.set_xlabel("contribution to the instability index")
    if not args.no_titles:
        axB.set_title(f"B  Largest dipeptide contributions ({len(dec)} observed)", loc="left")
    p = style.save_fig(fig, run.figures_dir / "instability_mechanism.png")
    print(f"\n[instability] wrote {p}")


if __name__ == "__main__":
    main()
