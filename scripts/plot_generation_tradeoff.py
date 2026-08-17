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
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    masks = sorted({k[0] for k in cells})
    shades = [style.ORANGE_DARK, style.ORANGE_MID, style.ORANGE_LIGHT]
    if natural:
        ax.axvspan(natural[0] - natural[1], natural[0] + natural[1],
                   color=style.BASELINE, alpha=0.25, zorder=0)
        ax.annotate("starting proteins\n(±SEM)", xy=(natural[0], 0.98),
                    xycoords=("data", "axes fraction"), ha="center", va="top", fontsize=8.5,
                    color=style.SECONDARY)
    for i, mask in enumerate(masks):
        ks = sorted([k for k in cells if k[0] == mask], key=lambda k: k[1])
        xs = [cells[k]["ptm"] for k in ks]
        ys = [cells[k]["effect"] for k in ks]
        ax.errorbar(xs, ys, xerr=[cells[k]["ptm_sem"] for k in ks],
                    yerr=[cells[k]["effect_sem"] for k in ks], fmt="-o", ms=6,
                    color=shades[i % len(shades)], linewidth=1.8, capsize=2.5, elinewidth=1,
                    label=f"mask {mask:g}")
        for k, x, y in zip(ks, xs, ys):
            ax.annotate(f"{k[1]}", (x, y), xytext=(6, 5), textcoords="offset points",
                        fontsize=8, color=style.INK,
                        path_effects=[pe.withStroke(linewidth=2.4, foreground=style.SURFACE)])
    if front:
        ax.step([p[0] for p in front], [p[1] for p in front], where="post",
                color=style.PURPLE_LIGHT, linewidth=2.2, alpha=0.9, zorder=1,
                label="non-dominated settings")
    ax.set_xlabel("structural confidence with no steering applied (ESMFold2 pTM)")
    ax.set_ylabel("steering-attributable ΔGRAVY at α=2")
    ax.set_title("Steering effect vs. foldability across generation settings\n"
                 "point labels are round counts; both axes mean ± SEM", loc="left")
    ax.legend(fontsize=9, loc="upper left")
    p = style.save_fig(fig, Path(args.out_run_dir) / "figures" / "generation_tradeoff.png")
    print(f"\n[tradeoff] wrote {p}")


if __name__ == "__main__":
    main()
