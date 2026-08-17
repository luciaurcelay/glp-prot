#!/usr/bin/env python
"""Sequence-complexity scan, and what the quality gate does to the reported effects.

Motivation. Every effect this experiment reports is computed AFTER a hand-tuned composition
gate (`is_degenerate_sequence`: >=30% one residue, or <=8 distinct amino acids, or an >=8-long
homopolymer run). Three numbers picked by eye therefore sit upstream of the headline result,
and the metrics that could have policed quality instead (pseudo-perplexity, pLDDT) score the
rejected sequences ABOVE our own output, so the gate is doing the work alone. That is the
weakest assumption in the experiment and it has never been checked.

Composition distance cannot check it, for three reasons: it is global (25% Leu spread evenly
scores like a 30-residue Leu block), it collapses to one scalar, and — decisively — the gate
is itself defined on composition, so using it to validate the gate is circular.

A windowed complexity scan is not circular and is the field-standard instrument. SEG (Wootton
& Federhen; shipped as BLAST+ `segmasker`) slides a window and flags regions of low
compositional entropy. Reimplemented here rather than shelled out to, because it is ~20 lines
and MN5 is offline. Defaults are SEG's own: window 12, low-complexity trigger K1 = 2.2 bits.
Using published parameters rather than inventing new ones is the point — the whole complaint
about the current gate is that its constants came from nowhere.

What this produces:
  1. a validity check — does complexity separate the sequences the gate keeps from the ones
     it rejects? (it should; if it does not, one of the two is wrong)
  2. differential attrition — does the gate remove sequences from the two arms at DIFFERENT
     rates? If so, every filtered comparison is between differently-selected populations.
  3. a sensitivity curve — the headline divergence recomputed under no gate, the current
     gate, and complexity gates at a range of cutoffs. If the result only exists at one
     setting, it is an artefact of that setting.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import _bootstrap  # noqa: F401
import numpy as np

from glp_plm.experiments import RunLogger, style

SEG_WINDOW = 12          # Wootton & Federhen's default
SEG_K1 = 2.2             # bits; SEG's low-complexity trigger


def window_entropies(seq: str, window: int = SEG_WINDOW) -> np.ndarray:
    """Shannon entropy (bits) of the residue distribution in each sliding window."""
    if len(seq) < window:
        window = max(2, len(seq))
    out = []
    for i in range(0, len(seq) - window + 1):
        counts = Counter(seq[i:i + window])
        n = float(window)
        out.append(-sum((c / n) * math.log2(c / n) for c in counts.values()))
    return np.asarray(out) if out else np.asarray([float("nan")])


def complexity(seq: str, window: int = SEG_WINDOW, k1: float = SEG_K1) -> Dict[str, float]:
    """Per-sequence complexity summary. `min_entropy` is the worst local patch — the quantity
    SEG thresholds — while `frac_low` says how much of the sequence is in such patches."""
    h = window_entropies(seq, window)
    return {"min_entropy": float(np.nanmin(h)), "mean_entropy": float(np.nanmean(h)),
            "frac_low": float(np.mean(h < k1))}


def is_degenerate(row: dict) -> bool:
    seq = row.get("sequence") or ""
    if seq and max(Counter(seq).values()) / len(seq) >= 0.30:
        return True
    return (int(float(row["unique_aa_count"])) <= 8
            or int(float(row["max_homopolymer_run"])) >= 8)


def attributable(rows: List[dict], neutral_alpha: float) -> None:
    base = {(r["direction"], r["corrected"], r["init"]): r["dgravy"]
            for r in rows if r["alpha"] == neutral_alpha}
    for r in rows:
        b = base.get((r["direction"], r["corrected"], r["init"]))
        r["attr"] = None if b is None else r["dgravy"] - b


def divergence(rows: List[dict], alpha: float, corrected: bool) -> Optional[Tuple[float, float, int]]:
    """up-minus-down of the attributable effect at one alpha, with an unpaired test."""
    from scipy.stats import mannwhitneyu

    u = [r["attr"] for r in rows if r["direction"] == "up" and r["corrected"] == corrected
         and r["alpha"] == alpha and r["attr"] is not None]
    d = [r["attr"] for r in rows if r["direction"] == "down" and r["corrected"] == corrected
         and r["alpha"] == alpha and r["attr"] is not None]
    if len(u) < 3 or len(d) < 3:
        return None
    try:
        pv = float(mannwhitneyu(u, d).pvalue)
    except ValueError:
        pv = float("nan")
    return float(np.mean(u) - np.mean(d)), pv, min(len(u), len(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--neutral-alpha", type=float, default=0.0)
    ap.add_argument("--window", type=int, default=SEG_WINDOW)
    ap.add_argument("--headline-alpha", type=float, default=2.0,
                    help="alpha at which the sensitivity curve is evaluated")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    raw = list(csv.DictReader(open(run_dir / "expB_diffmean_gravy.csv")))
    rows = []
    for r in raw:
        c = complexity(r["sequence"], args.window)
        rows.append({"alpha": float(r["alpha"]), "corrected": r["corrected"] == "True",
                     "direction": r.get("direction") or "up", "init": r["init_sequence_id"],
                     "dgravy": float(r["delta_gravy"]), "gated_out": is_degenerate(r), **c})
    attributable(rows, args.neutral_alpha)
    run = RunLogger(run_dir)
    kept = [r for r in rows if not r["gated_out"]]
    cut = [r for r in rows if r["gated_out"]]
    print(f"[complexity] {len(rows)} sequences; gate keeps {len(kept)}, rejects {len(cut)}")
    print(f"[complexity] SEG-style window={args.window}, low-complexity trigger "
          f"{SEG_K1} bits (max possible {math.log2(args.window):.2f} at this window)")

    # ---- 0. the reference that decides whether any threshold is meaningful ----
    # SEG's 2.2-bit trigger was calibrated on natural proteins at large. Our held-out starting
    # sequences ARE natural proteins (real UniRef50 entries), and they are the right yardstick:
    # they were drawn from GRAVY extremes, so if very hydrophobic natural proteins are
    # themselves locally low-complexity — transmembrane stretches are — then a threshold that
    # flags them is measuring the property we steer toward, not degeneracy.
    import json
    natural: Dict[str, List[dict]] = {}
    gc = run_dir / "gravy_contrast.json"
    if gc.exists():
        d = json.loads(gc.read_text())
        for setname, entries in (("held-out, low-GRAVY (up starts)", d.get("heldout") or []),
                                 ("held-out, high-GRAVY (down starts)", d.get("heldout_high") or [])):
            natural[setname] = [complexity(e["sequence"], args.window) for e in entries]
    print("\n0. NATURAL PROTEINS, SAME MEASURE (the reference for any threshold)")
    for setname, cs in natural.items():
        if not cs:
            continue
        me = [c["min_entropy"] for c in cs]
        below = sum(m < SEG_K1 for m in me)
        print(f"   {setname:<38} min_entropy {np.mean(me):5.3f}  "
              f"below {SEG_K1} bits: {below}/{len(me)} ({below/len(me):.0%})")
    gen_me = [r["min_entropy"] for r in rows]
    print(f"   {'our generations':<38} min_entropy {np.mean(gen_me):5.3f}  "
          f"below {SEG_K1} bits: {sum(m < SEG_K1 for m in gen_me)}/{len(gen_me)} "
          f"({np.mean([m < SEG_K1 for m in gen_me]):.0%})")
    if natural:
        run.log_summary(complexity_natural_reference={
            k: {"mean_min_entropy": float(np.mean([c["min_entropy"] for c in v])),
                "frac_below_seg_k1": float(np.mean([c["min_entropy"] < SEG_K1 for c in v])),
                "n": len(v)}
            for k, v in natural.items() if v})

    # ---- 1. validity: does complexity see what the gate sees? ----
    from scipy.stats import mannwhitneyu
    print("\\n1. DOES COMPLEXITY AGREE WITH THE GATE?")
    for field in ("min_entropy", "frac_low"):
        a = [r[field] for r in kept]
        b = [r[field] for r in cut]
        if not b:
            continue
        pv = float(mannwhitneyu(a, b).pvalue)
        print(f"   {field:>13}: kept {np.mean(a):6.3f}   rejected {np.mean(b):6.3f}   p={pv:.1e}")
    overlap = [r for r in kept if r["min_entropy"] < SEG_K1]
    print(f"   sequences the gate KEEPS that SEG would call low-complexity: "
          f"{len(overlap)}/{len(kept)} ({len(overlap)/max(len(kept),1):.1%})")

    # ---- 2. differential attrition ----
    print("\\n2. DOES THE GATE REMOVE THE ARMS AT DIFFERENT RATES?")
    print("   (if so, every filtered comparison is between differently-selected populations)")
    att: Dict[str, dict] = {}
    for a in sorted({r["alpha"] for r in rows}):
        e = {}
        for corrected, nm in ((False, "naive"), (True, "corrected")):
            cell = [r for r in rows if r["alpha"] == a and r["corrected"] == corrected]
            if cell:
                e[nm] = sum(r["gated_out"] for r in cell) / len(cell)
        if len(e) == 2:
            att[f"{a:g}"] = e
            flag = "  <-- differs by >10pp" if abs(e["naive"] - e["corrected"]) > 0.10 else ""
            print(f"   α={a:<5} removed: naive {e['naive']:5.1%}   corrected {e['corrected']:5.1%}{flag}")

    # ---- 3. sensitivity of the headline result to the gate ----
    print(f"\\n3. HEADLINE DIVERGENCE AT α={args.headline_alpha:g} UNDER DIFFERENT GATES")
    print(f"   {'gate':<34} {'naive':>22} {'GLP-corrected':>24}")
    gates: List[Tuple[str, List[dict]]] = [
        ("none (all generated sequences)", rows),
        ("current composition gate", kept),
    ]
    for k in (1.8, 2.2, 2.6, 3.0):
        gates.append((f"SEG-style min entropy >= {k}", [r for r in rows if r["min_entropy"] >= k]))
    sens: Dict[str, dict] = {}
    for name, subset in gates:
        attributable(subset, args.neutral_alpha)
        line, entry = f"   {name:<34}", {"n": len(subset)}
        for corrected, nm in ((False, "naive"), (True, "corrected")):
            res = divergence(subset, args.headline_alpha, corrected)
            if res is None:
                line += f"{'--':>22}"
                continue
            dv, pv, n = res
            entry[nm] = {"divergence": dv, "p": pv, "n_min": n}
            line += f"{dv:>+10.3f} p={pv:<9.2e}"
        sens[name] = entry
        print(line + f"   (n={len(subset)})")
    attributable(rows, args.neutral_alpha)   # restore
    run.log_summary(complexity_gate_sensitivity={
        "window": args.window, "seg_k1": SEG_K1, "headline_alpha": args.headline_alpha,
        "attrition_by_alpha": att, "divergence_under_gates": sens})

    # ---- figure: effect vs how strict the gate is ----
    style.apply_style()
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ks = np.arange(1.4, 3.45, 0.1)
    for corrected, nm in ((False, "naive"), (True, "GLP-corrected")):
        ys, ns = [], []
        for k in ks:
            sub = [r for r in rows if r["min_entropy"] >= k]
            attributable(sub, args.neutral_alpha)
            res = divergence(sub, args.headline_alpha, corrected)
            ys.append(res[0] if res else np.nan)
            ns.append(len(sub))
        axL.plot(ks, ys, "-o", ms=3.5, color=style.steering_arm_color(corrected), label=nm)
        if corrected:
            axR.plot(ks, [100 * n / len(rows) for n in ns], "-o", ms=3.5,
                     color=style.SECONDARY)
    attributable(rows, args.neutral_alpha)
    axL.axvline(SEG_K1, color=style.BASELINE, linestyle=":", linewidth=1.4)
    axL.set_xlabel("minimum window entropy required (bits)")
    axL.set_ylabel(f"divergence at α={args.headline_alpha:g}\\n(up − down, GRAVY)")
    axL.set_title("Effect vs. how strict the complexity gate is", loc="left")
    axL.legend(fontsize=9)
    axR.axvline(SEG_K1, color=style.BASELINE, linestyle=":", linewidth=1.4)
    axR.set_xlabel("minimum window entropy required (bits)")
    axR.set_ylabel("% of generated sequences retained")
    axR.set_title("Sequences retained at that strictness", loc="left")
    p = style.save_fig(fig, run.figures_dir / "complexity_gate_sensitivity.png")
    print(f"\\n[complexity] wrote {p}")


if __name__ == "__main__":
    main()
