#!/usr/bin/env python
"""Exp B alpha=0 confound: component decomposition + raw-vs-attributable dose-response.

Why this exists: at alpha=0 the two Exp B arms differ ONLY by whether `glp_correct` ran
(diffmean steering at alpha=0 is an exact identity), yet on hydrophobicity the GLP-corrected
arm already gains +1.35 ΔGRAVY against the naive arm's +0.40. This script shows where that
comes from and what survives once it's subtracted, for whichever property the run steered:

  Panel A  the drift, decomposed into the components the property is a sum over. For GRAVY
           and charge that is EXACT and per-residue (Δscore = Σ_aa Δf(aa)·w(aa)); for the
           instability index the components are DIPEPTIDES, which is the whole reason that
           target is worth running — a purely compositional artifact cannot produce it.
  Panel B  the raw effect (what summary.json's paired_wilcoxon reports) vs the
           steering-attributable one (each arm differenced against its own alpha=0 run,
           per init sequence).

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

from glp_plm.eval.steering_properties import (find_contrast_json, find_run_csv, row_delta,
                                              run_property)
from glp_plm.experiments import style


def is_degenerate(row: dict) -> bool:
    """Mirrors run_steering.is_degenerate_sequence (kept in sync deliberately: this script
    must analyse exactly the row set that produced summary.json's Part 3 numbers)."""
    seq = row["sequence"]
    if seq and max(Counter(seq).values()) / len(seq) >= 0.30:
        return True
    return (int(float(row["unique_aa_count"])) <= 8
            or int(float(row["max_homopolymer_run"])) >= 8)


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
                    help="steering out-dir (holds expB_diffmean_<property>.csv and the "
                         "matching <property>_contrast.json)")
    ap.add_argument("--out", help="output PNG (default <run-dir>/figures/alpha0_confound.png)")
    ap.add_argument("--top-components", type=int, default=20,
                    help="how many decomposition components panel A shows. All 20 residues "
                         "for GRAVY/charge; for instability the components are dipeptides "
                         "(up to 400), so only the largest contributors are drawn — the "
                         "printed total is still over all of them.")
    ap.add_argument("--max-alpha", type=float, default=2.0,
                    help="omit alphas above this from panel B — past the breakdown point the "
                         "degenerate filter leaves too few clean corrected rows to plot "
                         "(n<=2 at alpha>=5 on the first run). Default 2.0")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    csv_path = find_run_csv(run_dir)
    rows = [r for r in csv.DictReader(open(csv_path))]
    prop = run_property(rows, csv_path)
    for r in rows:
        r["_alpha"] = float(r["alpha"])
        r["_corr"] = r["corrected"] == "True"
        r["_delta"] = row_delta(r)
    clean = [r for r in rows if not is_degenerate(r)]
    print(f"property={prop.key}  rows={len(rows)}  clean={len(clean)}  "
          f"degenerate={len(rows) - len(clean)}")

    contrast = find_contrast_json(run_dir, prop.key)
    if contrast is None:
        raise SystemExit(f"no <property>_contrast.json in {run_dir}")
    heldout = {str(h["id"]): h["sequence"]
               for h in json.loads(contrast.read_text())["heldout"]}

    # ---- Panel A data: alpha=0 corrected drift, decomposed into the property's own
    # components (residues for GRAVY/charge, dipeptides for instability) ----
    a0c = [r for r in clean if r["_alpha"] == 0.0 and r["_corr"]]
    dec = prop.decompose([heldout[r["init_sequence_id"]] for r in a0c],
                         [r["sequence"] for r in a0c])
    total = sum(v for _, v in dec)
    shown = dec[:args.top_components]
    order = [k for k, _ in sorted(shown, key=lambda t: t[1])]   # ascending -> largest at top
    contrib = dict(dec)
    print(f"\nalpha=0 corrected drift decomposes to {total:+.4f} {prop.unit} "
          f"(CSV mean {mean(r['_delta'] for r in a0c):+.4f}"
          + ("" if prop.compositional else "; dipeptide components, sum over all 400") + ")")
    print(f"{'component':>10} {'contrib':>10}   ({len(dec)} total, showing {len(shown)})")
    for k, v in shown:
        print(f"{k:>10} {v:>+10.4f}")

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
    axA.set_yticklabels(order, fontsize=9)
    axA.set_xlabel(f"contribution to {prop.delta_label}"
                   + ("  (Δfrequency × scale weight)" if prop.compositional
                      else "  (Δdipeptide contribution)"))
    axA.set_title("A  Where the no-steering drift comes from", loc="left")
    # direct-label only the two dominant bars, not every point
    for k, _ in dec[:2]:
        if k in order:
            i = order.index(k)
            axA.annotate(f"{contrib[k]:+.2f}", (contrib[k], i), xytext=(5, 0),
                         textcoords="offset points", va="center", fontsize=9,
                         color=style.INK, fontweight="bold")
    axA.annotate(f"total {total:+.3g} {prop.unit}\nwith zero steering applied\n"
                 f"(over all {len(dec)} components)",
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
    axB.set_ylabel(prop.delta_label)
    axB.set_title("B  Raw effect vs. what steering actually contributes", loc="left")
    axB.legend(loc="upper left", fontsize=9)

    out = Path(args.out) if args.out else run_dir / "figures" / "alpha0_confound.png"
    print(f"\nwrote {style.save_fig(fig, out)}")


if __name__ == "__main__":
    main()
