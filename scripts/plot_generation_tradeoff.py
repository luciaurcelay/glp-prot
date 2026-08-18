#!/usr/bin/env python
"""Steering effect vs. foldability across generation settings — choosing an operating point.

This IS a multi-objective selection problem, unlike the alpha sweep (where a Pareto framing
was wrong: alpha is one knob dialled to taste, and structural confidence did not vary along
it). Here the configuration is two-dimensional — mask fraction x rounds — and we have to pick
one setting, so the achievable set in (foldability, effect) space and its non-dominated
frontier are the right objects.

Read the frontier with the error bars in view. Within the light settings alone pTM spans only
~0.04 against SEMs of 0.04-0.06, i.e. no resolvable trade-off; the cost of editing only
becomes visible once the heavy settings are included, which is why this pools pilots.

Foldability is taken from the NO-STEERING arm. Steering toward hydrophobicity raises pTM by
itself (natural hydrophobic proteins fold at 0.629 vs 0.373 for hydrophilic ones), so reading
it off the steered arm would credit the treatment with a structural gain it did not make.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple

import _bootstrap  # noqa: F401
import numpy as np

from glp_plm.experiments import RunLogger, style


def sem(v: List[float]) -> float:
    return stdev(v) / len(v) ** 0.5 if len(v) > 1 else float("nan")


def is_degenerate(seq: str) -> bool:
    """Same gate every main-run analysis applies. The pilot analyses originally omitted it,
    which inverted the temperature conclusion: low-temperature sampling MODE-COLLAPSES, and
    the collapsed sequences score high pTM (they are the artefact that also fools pLDDT and
    pseudo-perplexity). Without this filter T=0.5 at 30 rounds looked like the best setting
    in the grid while actually producing 8/15 non-proteins."""
    from collections import Counter
    if not seq:
        return True
    if max(Counter(seq).values()) / len(seq) >= 0.30:
        return True
    run = best = 1
    for i in range(1, len(seq)):
        run = run + 1 if seq[i] == seq[i - 1] else 1
        best = max(best, run)
    return len(set(seq)) <= 8 or best >= 12


def load(run_dir: Path, temperature: Optional[float]) -> List[dict]:
    gen = list(csv.DictReader(open(run_dir / "generation_pilot.csv")))
    fold = {f["sample_id"]: f for f in csv.DictReader(open(run_dir / "generation_pilot_fold.csv"))
            if not f["fold_error"]}
    out = []
    for g in gen:
        f = fold.get(g["sample_id"])
        if not f:
            continue
        t = float(g["temperature"])
        if temperature is not None and abs(t - temperature) > 1e-9:
            continue
        out.append({"rounds": int(g["n_rounds"]),
                    "mask": float(g.get("mask_fraction") or 0.10),
                    "temp": t, "alpha": float(g["alpha"]),
                    "corr": g["corrected"] == "True", "init": g["init_sequence_id"],
                    "dgravy": float(g["delta_gravy"]), "ptm": float(f["ptm"]),
                    "degenerate": is_degenerate(g["sequence"])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", required=True,
                    help="comma-separated generation-pilot run dirs to pool")
    ap.add_argument("--out-run-dir", required=True, help="where the figure is written")
    ap.add_argument("--temperature", type=float, default=0.5,
                    help="only pool cells at this temperature, so pooled pilots are comparable")
    ap.add_argument("--neutral-alpha", type=float, default=0.0)
    ap.add_argument("--steered-alpha", type=float, default=2.0)
    ap.add_argument("--natural-fold", help="heldout_starts_fold.csv for the starting proteins")
    ap.add_argument("--natural-direction", default="up",
                    help="which held-out set the pilots generate from — the baseline MUST be "
                         "that set alone, not both pooled (they differ: 0.373 vs 0.629)")
    args = ap.parse_args()

    rows: List[dict] = []
    for d in args.run_dirs.split(","):
        rows += load(Path(d.strip()), args.temperature)
    print(f"[tradeoff] {len(rows)} folded sequences at T={args.temperature:g}")

    natural = None
    if args.natural_fold and Path(args.natural_fold).exists():
        nf = [r for r in csv.DictReader(open(args.natural_fold))
              if not r["fold_error"] and r.get("direction") == args.natural_direction]
        if nf:
            v = [float(r["ptm"]) for r in nf]
            natural = (mean(v), sem(v), len(v))
            print(f"[tradeoff] natural '{args.natural_direction}' starting proteins: "
                  f"pTM {natural[0]:.3f}±{natural[1]:.3f} (n={natural[2]})")

    cells: Dict[Tuple[float, int], dict] = {}
    for mask in sorted({r["mask"] for r in rows}):
        for rr in sorted({r["rounds"] for r in rows}):
            all_uns = [r for r in rows if r["mask"] == mask and r["rounds"] == rr
                       and r["alpha"] == args.neutral_alpha and not r["corr"]]
            unsteered = [r["ptm"] for r in all_uns if not r["degenerate"]]
            # degenerate output is a FAILURE of the setting, not a row to drop quietly: pTM
            # among survivors alone is survivor-biased, and at T=0.5/30 rounds only 7 of 15
            # survive, so the surviving mean says little about the setting.
            deg_rate = (sum(r["degenerate"] for r in all_uns) / len(all_uns)) if all_uns else None
            base = {r["init"]: r["dgravy"] for r in rows if r["mask"] == mask
                    and r["rounds"] == rr and r["alpha"] == args.neutral_alpha and r["corr"]
                    and not r["degenerate"]}
            eff = [r["dgravy"] - base[r["init"]] for r in rows if r["mask"] == mask
                   and r["rounds"] == rr and r["alpha"] == args.steered_alpha and r["corr"]
                   and not r["degenerate"] and r["init"] in base]
            if len(unsteered) < 3 or len(eff) < 3:
                continue
            cells[(mask, rr)] = {"ptm": mean(unsteered), "ptm_sem": sem(unsteered),
                                 "effect": mean(eff), "effect_sem": sem(eff),
                                 "n": min(len(unsteered), len(eff)),
                                 "degenerate_rate": deg_rate}

    print(f"\n{'mask':>5} {'rounds':>7} | {'pTM (no steering)':>20} | {'steering effect':>20} "
          f"| {'degenerate':>11}")
    for (mask, rr), c in sorted(cells.items()):
        dr = c.get("degenerate_rate")
        flag = "  <-- unusable" if dr is not None and dr >= 0.25 else ""
        print(f"{mask:>5.2f} {rr:>7} | {c['ptm']:>10.3f} ±{c['ptm_sem']:.3f} | "
              f"{c['effect']:>+10.3f} ±{c['effect_sem']:.3f} | "
              f"{'--' if dr is None else f'{dr:>10.0%}'}{flag}")

    # non-dominated set under maximise-both
    # a setting whose output is mostly degenerate is not a candidate, whatever its surviving
    # sequences score
    pts = [(c["ptm"], c["effect"], k) for k, c in cells.items()
           if (c.get("degenerate_rate") or 0) < 0.25]
    best_y, front = -float("inf"), []
    for x, y, k in sorted(pts, key=lambda p: p[0], reverse=True):
        if y > best_y:
            front.append((x, y, k))
            best_y = y
    front = list(reversed(front))
    print("\nnon-dominated settings (no other setting beats them on BOTH axes):")
    for x, y, k in front:
        print(f"  mask={k[0]:.2f} rounds={k[1]:<3} pTM {x:.3f}  effect {y:+.3f}")

    RunLogger(Path(args.out_run_dir)).log_summary(generation_tradeoff={
        "temperature": args.temperature,
        "cells": {f"mask{k[0]:g}_rounds{k[1]}": v for k, v in cells.items()},
        "non_dominated": [{"mask": k[0], "rounds": k[1], "ptm": x, "effect": y}
                          for x, y, k in front],
        "natural_ptm": natural[0] if natural else None})

    style.apply_style()
    import matplotlib.pyplot as plt

    # Plotted against expected coverage -- the fraction of residues resampled at least once,
    # 1-(1-mask)^rounds -- rather than as a scatter in (pTM, effect) space.
    #
    # Two reasons. (a) mask fraction and round count turn out to be interchangeable: at
    # coverage 0.10, settings 0.02x5, 0.05x2 and 0.10x1 give pTM 0.364/0.364/0.366 and effect
    # +0.057/+0.081/+0.078. The 2-D grid is a 1-D family, so the three series are overlaid
    # here to show that collapse rather than separated by colour. (b) The objective-space
    # scatter spent its x-axis on pTM, whose per-cell SEM (0.04-0.08) is comparable to its
    # whole range (0.30-0.42) -- every point's horizontal position was noise, the frontier was
    # undetermined, and nine of fifteen settings piled into an unreadable blob.
    #
    # One colour, shape for mask fraction: three shades of the same hue failed the palette's
    # own adjacency check (5.3-7.6 dE), and separating the series was never the point anyway.
    def coverage(mask: float, rounds: int) -> float:
        return 1.0 - (1.0 - mask) ** rounds

    ok = {k: c for k, c in cells.items() if (c.get("degenerate_rate") or 0) < 0.25}
    bad = {k: c for k, c in cells.items() if (c.get("degenerate_rate") or 0) >= 0.25}
    shapes = {m: mk for m, mk in zip(sorted({k[0] for k in cells}), ("o", "s", "^", "D"))}
    accent = style.method_color("glp")

    fig, (axT, axB) = plt.subplots(2, 1, figsize=(7.4, 7.0), sharex=True,
                                   gridspec_kw={"hspace": 0.12})
    for src, colour, alpha, lbl in ((ok, accent, 1.0, None),
                                    (bad, style.MUTED, 0.55, "mostly degenerate output")):
        for (mask, rr), c in sorted(src.items(), key=lambda kv: coverage(*kv[0])):
            x = coverage(mask, rr)
            axT.errorbar([x], [c["effect"]], yerr=[c["effect_sem"]], color=colour, alpha=alpha,
                         marker=shapes[mask], ms=7, capsize=2.5, elinewidth=1, linestyle="none")
            axB.errorbar([x], [c["ptm"]], yerr=[c["ptm_sem"]], color=colour, alpha=alpha,
                         marker=shapes[mask], ms=7, capsize=2.5, elinewidth=1, linestyle="none")
    # trend through the usable settings only
    xs = sorted(coverage(*k) for k in ok)
    for ax, key in ((axT, "effect"), (axB, "ptm")):
        ys = [ok[k][key] for k in sorted(ok, key=lambda k: coverage(*k))]
        ax.plot(xs, ys, "-", color=accent, linewidth=1.4, alpha=0.55, zorder=0)

    if natural:
        axB.axhspan(natural[0] - natural[1], natural[0] + natural[1],
                    color=style.BASELINE, alpha=0.3, zorder=0)
        axB.annotate("starting proteins", xy=(0.985, natural[0]),
                     xycoords=("axes fraction", "data"), ha="right", va="bottom",
                     fontsize=8.5, color=style.SECONDARY)
    for m, mk in shapes.items():
        axT.plot([], [], mk, color=accent, ms=7, linestyle="none", label=f"mask {m:g}")
    if bad:
        axT.plot([], [], "o", color=style.MUTED, alpha=0.55, ms=7, linestyle="none",
                 label="mostly degenerate output")
    axT.legend(fontsize=9, loc="upper left")
    axT.set_ylabel(f"steering-attributable ΔGRAVY\nat α={args.steered_alpha:g}")
    axB.set_ylabel("structural confidence\n(ESMFold2 pTM, no steering)")
    axB.set_xscale("log")
    axB.set_xlabel("expected fraction of residues resampled  "
                   r"$1-(1-\mathrm{mask})^{\,\mathrm{rounds}}$")
    axB.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    axB.set_xticklabels(["2%", "5%", "10%", "20%", "50%", "100%"])
    axT.set_title("Steering effect and foldability against how much of the protein is rewritten",
                  loc="left")
    p = style.save_fig(fig, Path(args.out_run_dir) / "figures" / "generation_tradeoff.png")
    print(f"\n[tradeoff] wrote {p}")


if __name__ == "__main__":
    main()
