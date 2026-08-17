#!/usr/bin/env python
"""Exp B alpha=0 confound: per-residue decomposition + raw-vs-attributable dose-response.

Why this exists: at alpha=0 the two Exp B arms differ ONLY by whether `glp_correct` ran
(diffmean steering at alpha=0 is an exact identity), yet the GLP-corrected arm already
gains +1.35 ΔGRAVY against the naive arm's +0.40. This script shows where that comes from
and what survives once it's subtracted:

  Panel A  GRAVY is the mean Kyte-Doolittle value over residues, so the drift decomposes
           EXACTLY as ΔGRAVY = Σ_aa Δf(aa)·KD(aa). Bars are each residue's contribution.
  Panel B  raw ΔGRAVY (what summary.json's paired_wilcoxon reports) vs steering-attributable
           ΔGRAVY (each arm differenced against its own alpha=0 run, per init sequence).

Reads `steering_attributable` from the run's metrics/summary.json when present (the repo
convention: figures read metrics, they don't recompute — glp_plm/experiments/metrics.py),
and falls back to computing it from the CSV for runs written before that field existed.

Usage:
  python scripts/analyze_steering_confound.py --run-dir <steering out-dir> [--out <png>]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import _bootstrap  # noqa: F401

from glp_plm.experiments import style

# Kyte-Doolittle hydropathy — the scale BioPython's ProteinAnalysis.gravy() averages.
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
      "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
      "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}


def is_degenerate(row: dict) -> bool:
    """Mirrors run_steering.is_degenerate_sequence (kept in sync deliberately: this script
    must analyse exactly the row set that produced summary.json's Part 3 numbers)."""
    seq = row["sequence"]
    if seq and max(Counter(seq).values()) / len(seq) >= 0.30:
        return True
    return (int(float(row["unique_aa_count"])) <= 8
            or int(float(row["max_homopolymer_run"])) >= 8)


def composition(seqs) -> dict:
    """Mean per-residue frequency across `seqs`, each sequence weighted equally."""
    out = {a: 0.0 for a in KD}
    for s in seqs:
        n, c = len(s), Counter(s)
        for a in KD:
            out[a] += c.get(a, 0) / n
    return {a: v / len(seqs) for a, v in out.items()}


def attributable_from_rows(rows, neutral_alpha=0.0) -> dict:
    """Fallback when summary.json predates the `steering_attributable` field."""
    base = {(r["_corr"], r["init_sequence_id"]): r["_delta"]
            for r in rows if r["_alpha"] == neutral_alpha}
    out = {}
    for a in sorted({r["_alpha"] for r in rows}):
        if a == neutral_alpha:
            continue
        entry = {}
        for corr, name in ((False, "naive"), (True, "corrected")):
            v = [r["_delta"] - base[(corr, r["init_sequence_id"])] for r in rows
                 if r["_alpha"] == a and r["_corr"] == corr
                 and (corr, r["init_sequence_id"]) in base]
            if v:
                entry[name] = {"mean": mean(v), "n": len(v)}
        if entry:
            out[str(a)] = entry
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="steering out-dir (holds expB_diffmean_gravy.csv + gravy_contrast.json)")
    ap.add_argument("--out", help="output PNG (default <run-dir>/figures/alpha0_confound.png)")
    ap.add_argument("--max-alpha", type=float, default=2.0,
                    help="omit alphas above this from panel B — past the breakdown point the "
                         "degenerate filter leaves too few clean corrected rows to plot "
                         "(n<=2 at alpha>=5 on the first run). Default 2.0")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    rows = [r for r in csv.DictReader(open(run_dir / "expB_diffmean_gravy.csv"))]
    for r in rows:
        r["_alpha"] = float(r["alpha"])
        r["_corr"] = r["corrected"] == "True"
        r["_delta"] = float(r["delta_gravy"])
    clean = [r for r in rows if not is_degenerate(r)]
    print(f"rows={len(rows)}  clean={len(clean)}  degenerate={len(rows) - len(clean)}")

    heldout = {str(h["id"]): h["sequence"]
               for h in json.loads((run_dir / "gravy_contrast.json").read_text())["heldout"]}

    # ---- Panel A data: alpha=0 corrected drift, decomposed per residue ----
    a0c = [r for r in clean if r["_alpha"] == 0.0 and r["_corr"]]
    c_init = composition([heldout[r["init_sequence_id"]] for r in a0c])
    c_fin = composition([r["sequence"] for r in a0c])
    contrib = {a: (c_fin[a] - c_init[a]) * KD[a] for a in KD}
    order = sorted(KD, key=lambda a: contrib[a])          # ascending -> largest at top in barh
    total = sum(contrib.values())
    print(f"\nalpha=0 corrected drift decomposes to {total:+.3f} GRAVY "
          f"(CSV mean {mean(r['_delta'] for r in a0c):+.3f})")
    print(f"{'aa':>3} {'KD':>6} {'init%':>7} {'final%':>7} {'Δpp':>7} {'contrib':>8}")
    for a in reversed(order):
        print(f"{a:>3} {KD[a]:>+6.1f} {c_init[a]*100:>7.2f} {c_fin[a]*100:>7.2f} "
              f"{(c_fin[a]-c_init[a])*100:>+7.2f} {contrib[a]:>+8.3f}")

    # ---- Panel B data: raw vs steering-attributable, per alpha ----
    summary_path = run_dir / "metrics" / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    attrib = summary.get("steering_attributable")
    if attrib:
        print(f"\nusing steering_attributable from {summary_path}")
    else:
        print("\nsummary.json has no `steering_attributable` (run predates it) "
              "— computing from the CSV")
        attrib = attributable_from_rows(clean)

    alphas = [a for a in sorted({r["_alpha"] for r in clean}) if a <= args.max_alpha]
    raw = {}
    for a in alphas:
        for corr, name in ((False, "naive"), (True, "corrected")):
            v = [r["_delta"] for r in clean if r["_alpha"] == a and r["_corr"] == corr]
            if v:
                raw[(name, a)] = mean(v)
    drift = raw.get(("corrected", 0.0))

    print(f"\n{'alpha':>6} | {'raw naive':>10} {'raw corr':>10} | {'attr naive':>11} {'attr corr':>10}")
    for a in alphas:
        an = attrib.get(str(a), {})
        f = lambda d: f"{d['mean']:+.3f}" if d else "     --"
        print(f"{a:>6} | {raw.get(('naive', a), float('nan')):>+10.3f} "
              f"{raw.get(('corrected', a), float('nan')):>+10.3f} | "
              f"{f(an.get('naive')):>11} {f(an.get('corrected')):>10}")

    # ---- render ----
    style.apply_style()
    import matplotlib.pyplot as plt

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.6),
                                   gridspec_kw={"width_ratios": [1, 1.15]})

    # Panel A — single hue: bar DIRECTION already encodes the sign of each contribution,
    # so a second hue would be redundant (and style.py documents dark orange next to dark
    # purple at 14.1-14.6 dE, under the 15 normal-vision floor).
    ys = range(len(order))
    axA.barh(list(ys), [contrib[a] for a in order], color=style.method_color("glp"),
             height=0.68)
    axA.axvline(0, color=style.BASELINE, linewidth=1.0)
    axA.set_yticks(list(ys))
    axA.set_yticklabels([f"{a}  ({(c_fin[a]-c_init[a])*100:+.1f}pp)" for a in order],
                        fontsize=9)
    axA.set_xlabel("contribution to ΔGRAVY  (Δfrequency × Kyte-Doolittle)")
    axA.set_title("A  Where the no-steering drift comes from", loc="left")
    # direct-label only the two dominant bars, not every point
    for a in ("L", "V"):
        i = order.index(a)
        axA.annotate(f"{contrib[a]:+.2f}", (contrib[a], i), xytext=(5, 0),
                     textcoords="offset points", va="center", fontsize=9,
                     color=style.INK, fontweight="bold")
    axA.annotate(f"total {total:+.2f} GRAVY\nwith zero steering applied\n"
                 "(losing a hydrophilic residue\nraises GRAVY too — e.g. K, R, D)",
                 xy=(0.97, 0.06), xycoords="axes fraction", ha="right", fontsize=9,
                 color=style.SECONDARY)

    # Panel B — color = arm identity (reusing pareto_exp_b's assignment so the same entity
    # keeps the same color across figures); linestyle = raw vs attributable.
    x = list(range(len(alphas)))
    for name, col in (("naive", style.ACCENT), ("corrected", style.method_color("glp"))):
        # full-strength hues (no alpha fade): style.py's 15.9 dE validation for this pair is
        # for the solid colors, and fading both toward the surface collapses that separation.
        # raw vs attributable is carried by linestyle instead.
        axB.plot(x, [raw.get((name, a), float("nan")) for a in alphas], "o--", color=col,
                 markersize=6, markerfacecolor=style.SURFACE, linewidth=2,
                 label=f"{'GLP-corrected' if name == 'corrected' else 'naive'} — raw")
        axB.plot(x, [attrib.get(str(a), {}).get(name, {}).get("mean", float("nan"))
                     if a != 0.0 else 0.0 for a in alphas],
                 "o-", color=col, markersize=6, linewidth=2,
                 label=f"{'GLP-corrected' if name == 'corrected' else 'naive'} — steering-attributable")
    if drift is not None:
        axB.axhline(drift, color=style.BASELINE, linestyle=":", linewidth=1.4)
        axB.annotate("correction-only drift (α=0)", xy=(len(alphas) - 1, drift),
                     xytext=(-4, 6), textcoords="offset points", ha="right", fontsize=9,
                     color=style.SECONDARY)
    axB.axhline(0, color=style.BASELINE, linewidth=1.0)
    axB.set_xticks(x)
    axB.set_xticklabels([f"{a:g}" for a in alphas])
    axB.set_xlabel("steering strength α")
    # not "final - init": that's only the raw series. The attributable series is a
    # difference-of-differences (vs. the same arm's own alpha=0 run), so the legend
    # carries the distinction and the axis stays generic.
    axB.set_ylabel("ΔGRAVY")
    axB.set_title("B  Raw effect vs. what steering actually contributes", loc="left")
    axB.legend(loc="upper left", fontsize=9)

    out = Path(args.out) if args.out else run_dir / "figures" / "alpha0_confound.png"
    print(f"\nwrote {style.save_fig(fig, out)}")


if __name__ == "__main__":
    main()
