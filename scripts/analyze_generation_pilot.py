#!/usr/bin/env python
"""Generation pilot: what does the masked-Gibbs decode loop cost, and can it be tuned back?

Context. The starting proteins fold at pTM 0.501 / pLDDT 75.3; the alpha=0 NAIVE arm — no
steering, no GLP — comes out at 0.264 / 54.1. So the decode loop halves foldability on its
own. GLP's activation-level claim is untouched by this (one correction pass still recovers
89-95% of a clamp's excess diffusion loss); what breaks is decoding activations back to
tokens and iterating that 30 times.

Two candidate mechanisms, which this separates:
  * temperature — every resample draws from an 8M model at T=1.0, so noise is injected on
    every one of ~3 visits per position regardless of how few rounds are run. If this
    dominates, lowering T is a fix with no cost in steering.
  * cumulative edits — foldability decays with the number of resamples. If this dominates
    there is a genuine trade-off: fewer rounds keeps structure but loses steering.

Reports, per (rounds, temperature) cell: foldability of the no-steering naive arm (what the
loop costs when nothing else happens) and the steering-attributable effect at the mid alpha
(what we would give up). The useful configuration maximises the second while keeping the
first near the starting proteins.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional

import _bootstrap  # noqa: F401
import numpy as np

from glp_plm.experiments import RunLogger, style


def sem(v: List[float]) -> float:
    return stdev(v) / len(v) ** 0.5 if len(v) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--neutral-alpha", type=float, default=0.0)
    ap.add_argument("--natural-fold",
                    help="heldout_starts_fold.csv, to print the starting proteins' own scores")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    gen = {r["sample_id"]: r for r in csv.DictReader(open(run_dir / "generation_pilot.csv"))}
    fold = [f for f in csv.DictReader(open(run_dir / "generation_pilot_fold.csv"))
            if not f["fold_error"]]
    rows = []
    for f in fold:
        g = gen.get(f["sample_id"])
        if not g:
            continue
        rows.append({"rounds": int(g["n_rounds"]), "temp": float(g["temperature"]),
                     "alpha": float(g["alpha"]), "corrected": g["corrected"] == "True",
                     "init": g["init_sequence_id"], "dgravy": float(g["delta_gravy"]),
                     "ptm": float(f["ptm"]), "plddt": float(f["plddt"])})
    print(f"[gen-pilot] {len(rows)} folded pilot sequences")

    nat = None
    if args.natural_fold and Path(args.natural_fold).exists():
        nf = [r for r in csv.DictReader(open(args.natural_fold)) if not r["fold_error"]]
        nat = (mean(float(r["ptm"]) for r in nf), mean(float(r["plddt"]) for r in nf), len(nf))
        print(f"[gen-pilot] starting proteins: pTM {nat[0]:.3f}  pLDDT {nat[1]:.1f}  n={nat[2]}")

    rounds = sorted({r["rounds"] for r in rows})
    temps = sorted({r["temp"] for r in rows}, reverse=True)
    mid_alpha = max({r["alpha"] for r in rows})

    # steering-attributable effect, per (rounds, temp): corrected at mid alpha minus that same
    # init's corrected alpha=0 run, so the correction's own drift is removed as everywhere else
    def attributable(rr: int, t: float) -> Optional[List[float]]:
        base = {r["init"]: r["dgravy"] for r in rows if r["rounds"] == rr and r["temp"] == t
                and r["alpha"] == args.neutral_alpha and r["corrected"]}
        out = [r["dgravy"] - base[r["init"]] for r in rows
               if r["rounds"] == rr and r["temp"] == t and r["alpha"] == mid_alpha
               and r["corrected"] and r["init"] in base]
        return out or None

    def naive_fold(rr: int, t: float, field: str) -> Optional[List[float]]:
        v = [r[field] for r in rows if r["rounds"] == rr and r["temp"] == t
             and r["alpha"] == args.neutral_alpha and not r["corrected"]]
        return v or None

    print(f"\nFOLDABILITY OF THE NO-STEERING NAIVE ARM (what the decode loop costs by itself)")
    print("  " + "rounds".rjust(7) + "".join(f"{f'T={t:g}':>18}" for t in temps))
    table: Dict[str, dict] = {}
    for rr in rounds:
        line = f"  {rr:>7}"
        for t in temps:
            v = naive_fold(rr, t, "ptm")
            line += f"{(f'{mean(v):.3f}±{sem(v):.3f}' if v else '--'):>18}"
            table.setdefault(f"rounds={rr}", {})[f"T={t:g}"] = {
                "naive_ptm": float(mean(v)) if v else None}
        print(line + "   pTM")

    print(f"\nSTEERING-ATTRIBUTABLE ΔGRAVY AT α={mid_alpha:g} (what we would give up)")
    print("  " + "rounds".rjust(7) + "".join(f"{f'T={t:g}':>18}" for t in temps))
    for rr in rounds:
        line = f"  {rr:>7}"
        for t in temps:
            v = attributable(rr, t)
            line += f"{(f'{mean(v):+.3f}±{sem(v):.3f}' if v else '--'):>18}"
            table[f"rounds={rr}"][f"T={t:g}"]["attributable"] = float(mean(v)) if v else None
        print(line)

    print("\nWHICH MECHANISM DOMINATES?")
    for t in temps:
        v = [table[f"rounds={rr}"][f"T={t:g}"]["naive_ptm"] for rr in rounds]
        if all(x is not None for x in v):
            print(f"  T={t:g}: pTM across rounds {rounds} -> "
                  + " ".join(f"{x:.3f}" for x in v)
                  + f"   (spread {max(v) - min(v):+.3f})")
    for rr in rounds:
        v = [table[f"rounds={rr}"][f"T={t:g}"]["naive_ptm"] for t in temps]
        if all(x is not None for x in v):
            print(f"  rounds={rr}: pTM across T {temps} -> "
                  + " ".join(f"{x:.3f}" for x in v)
                  + f"   (spread {max(v) - min(v):+.3f})")

    run = RunLogger(run_dir)
    run.log_summary(generation_pilot={"mid_alpha": mid_alpha, "by_cell": table,
                                      "natural_starts_ptm": nat[0] if nat else None})

    style.apply_style()
    import matplotlib.pyplot as plt

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    shades = [style.ORANGE_DARK, style.ORANGE_MID, style.ORANGE_LIGHT]
    for i, t in enumerate(temps):
        c = shades[i % len(shades)]
        y = [table[f"rounds={rr}"][f"T={t:g}"]["naive_ptm"] for rr in rounds]
        axA.plot(rounds, y, "-o", color=c, ms=5, label=f"T = {t:g}")
        y2 = [table[f"rounds={rr}"][f"T={t:g}"]["attributable"] for rr in rounds]
        axB.plot(rounds, y2, "-o", color=c, ms=5, label=f"T = {t:g}")
    if nat:
        axA.axhline(nat[0], color=style.BASELINE, linestyle=":", linewidth=1.4)
        axA.annotate("starting proteins", xy=(rounds[-1], nat[0]), xytext=(-4, 4),
                     textcoords="offset points", ha="right", fontsize=8.5,
                     color=style.SECONDARY)
    axA.set_xlabel("masked-Gibbs rounds")
    axA.set_ylabel("structural confidence (ESMFold2 pTM)\nno steering, no correction")
    axA.set_title("Foldability retained by the decode loop", loc="left")
    axA.legend(fontsize=9)
    axB.set_xlabel("masked-Gibbs rounds")
    axB.set_ylabel(f"steering-attributable ΔGRAVY at α={mid_alpha:g}")
    axB.set_title("Steering effect retained", loc="left")
    axB.legend(fontsize=9)
    p = style.save_fig(fig, run.figures_dir / "generation_pilot.png")
    print(f"\n[gen-pilot] wrote {p}")


if __name__ == "__main__":
    main()
