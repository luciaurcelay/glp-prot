#!/usr/bin/env python
"""Rebuild Exp B's rescue / Pareto analysis on a naturalness axis that isn't gamed.

Why this exists. The rescue fraction and Pareto plot that `run_steering.py` writes use
pseudo-perplexity as the naturalness axis, and on this generation setup pppl is gamed by
exactly the failure mode the correction produces. Measured on the t07_full run (300
sequences, ESMFold2 @ 20 sampling steps), the sequences the degenerate filter REJECTS —
unambiguously not proteins (>=30% one residue, or <=8 unique AAs, or an >=8-long homopolymer
run; 64% Leu+Val on average) — score:

    group                  pLDDT     pTM     pppl
    degenerate junk         81.0    0.452     5.46
    clean generations       54.7    0.277    14.53
    natural ubiquitin       80      0.74      -

i.e. poly-Leu/Val junk beats real ubiquitin on pLDDT and scores ~3x "better" pppl than our
clean generations (all p <= 1e-6). So:

  * pppl and pLDDT are BOTH gamed here, and are therefore not usable as the headline axis.
    On clean rows they correlate correctly with each other (rho=-0.48) — but that is shared
    bias, not mutual validation: the correlation is computed after the degenerate filter has
    removed the very sequences that expose it. (Check metrics on the rejected rows, always.)
  * pTM is the most robust of the three: it still separates junk (0.452) from a real protein
    (0.74), whereas pLDDT does not separate them at all (81 vs 80). Hence pTM is the axis
    used here, with the degenerate filter kept as a HARD GATE rather than a tidy-up, and
    composition distance reported alongside as a model-free secondary flag.

Direction matters: higher pTM = more natural, the opposite of pppl, so the rescue
comparator's inequality is flipped relative to run_steering.py's version.

Requires the fold sidecar (scripts/fold_steering_outputs.py). Runs in the PROJECT venv (it
uses glp_plm for styling/logging), unlike the fold stage which needs venv312.

Usage:
  .venv/bin/python scripts/analyze_steering_naturalness.py --run-dir <steering-out-dir>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import _bootstrap  # noqa: F401
import numpy as np

from glp_plm.experiments import RunLogger, style

# a well-folded natural protein, folded through the same pipeline, as a scale anchor
UBIQUITIN_REF = {"plddt": 80.0, "ptm": 0.74}


def is_degenerate(row: dict) -> bool:
    """Mirrors run_steering.is_degenerate_sequence so this analyses the same row set."""
    seq = row.get("sequence") or ""
    if seq and max(Counter(seq).values()) / len(seq) >= 0.30:
        return True
    return (int(float(row["unique_aa_count"])) <= 8
            or int(float(row["max_homopolymer_run"])) >= 8)


def rescue_fraction(naive: List[dict], corrected: List[dict], key: str,
                    higher_is_better: bool) -> Tuple[float, int]:
    """Spec Part 3's sanity check, generalised over the naturalness axis and its direction.

    At matched-or-better naturalness, a GLP-corrected point should beat the best naive
    concept score achieved at that naturalness or worse. A corrected point with no
    naturalness-matched naive comparator is excluded rather than counted as a free win."""
    rescued = comparable = 0
    for c in corrected:
        if c.get(key) is None:
            continue
        matched = [n for n in naive if n.get(key) is not None
                   and (n[key] >= c[key] if higher_is_better else n[key] <= c[key])]
        if not matched:
            continue
        comparable += 1
        if c["concept_score"] > max(n["concept_score"] for n in matched):
            rescued += 1
    return (rescued / comparable if comparable else float("nan")), comparable


def load(run_dir: Path, csv_name: Optional[str]) -> Tuple[List[dict], List[dict]]:
    gen_path = None
    for name in ([csv_name] if csv_name else
                 ["expB_diffmean_gravy.csv", "expA_sae_zincfinger.csv"]):
        if name and (run_dir / name).exists():
            gen_path = run_dir / name
            break
    if gen_path is None:
        raise SystemExit(f"no generation CSV found in {run_dir}")
    side = run_dir / "fold_confidence.csv"
    if not side.exists():
        raise SystemExit(f"{side} missing — run scripts/cluster/submit_fold.sh on this run first")

    gen = {r["sample_id"]: r for r in csv.DictReader(open(gen_path))}
    folds = list(csv.DictReader(open(side)))
    print(f"[naturalness] {gen_path.name}: {len(gen)} rows; sidecar: {len(folds)} folds")

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    merged, mismatch, failed = [], 0, 0
    for f in folds:
        g = gen.get(f["sample_id"])
        if g is None:
            continue
        # the sidecar records the sha1 of what it folded; refuse to merge against
        # regenerated sequences rather than silently mixing runs
        if hashlib.sha1(g["sequence"].encode()).hexdigest()[:12] != f["seq_sha1"]:
            mismatch += 1
            continue
        if f.get("fold_error"):
            failed += 1
            continue
        merged.append({
            **g, "alpha": float(g["alpha"]), "corrected": g["corrected"] == "True",
            "direction": g.get("direction") or "up",
            "concept_score": float(g["delta_gravy"]) if "delta_gravy" in g
            else (1.0 if g.get("motif_hit") == "True" else 0.0),
            "pppl": num(g.get("naturalness_score")),
            "comp": num(g.get("composition_distance")),
            "plddt": num(f.get("plddt")), "ptm": num(f.get("ptm")),
            "_degenerate": is_degenerate(g),
        })
    if mismatch:
        raise SystemExit(f"{mismatch} sidecar rows do not match the current sequences — the "
                         "run was regenerated after folding. Re-run the fold stage "
                         "(--restart) before analysing.")
    if failed:
        print(f"[naturalness] {failed} fold(s) errored and are excluded")
    return merged, folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--csv-name", help="override the generation CSV filename")
    ap.add_argument("--axis", default="ptm", choices=["ptm", "plddt", "pppl"],
                    help="naturalness axis for the rescue/Pareto rebuild. Default pTM — see "
                         "this script's docstring for why not pLDDT or pppl")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    rows, _ = load(run_dir, args.csv_name)
    run = RunLogger(run_dir)
    higher_is_better = args.axis != "pppl"

    # ---- the justification: how each axis rates the sequences the filter rejects ----
    deg = [r for r in rows if r["_degenerate"]]
    clean = [r for r in rows if not r["_degenerate"]]
    print(f"\n[naturalness] degenerate (hard-gated out): {len(deg)}   clean: {len(clean)}")
    gate = {}
    print(f"{'axis':>10} {'degenerate':>14} {'clean':>14} {'natural ref':>12}  verdict")
    for key in ("plddt", "ptm", "pppl"):
        d = [r[key] for r in deg if r[key] is not None]
        c = [r[key] for r in clean if r[key] is not None]
        if not d or not c:
            continue
        ref = UBIQUITIN_REF.get(key)
        # an axis is gamed if it rates known non-proteins BETTER than clean output, where
        # "better" is lower for pppl and higher for plddt/ptm
        gamed = np.mean(d) < np.mean(c) if key == "pppl" else np.mean(d) > np.mean(c)
        gate[key] = {"degenerate_mean": float(np.mean(d)), "clean_mean": float(np.mean(c)),
                     "n_degenerate": len(d), "gamed_by_degenerate": bool(gamed)}
        print(f"{key:>10} {np.mean(d):>14.3f} {np.mean(c):>14.3f} "
              f"{'' if ref is None else f'{ref:>12.2f}'}  "
              f"{'GAMED — rates junk above clean output' if gamed else 'discriminates'}")
    run.log_summary(naturalness_axis_gate_check=gate)

    # ---- rescue fraction on the chosen axis, per direction, degenerate hard-gated ----
    key = args.axis
    print(f"\n[naturalness] rescue fraction on {key} "
          f"({'higher' if higher_is_better else 'lower'} = more natural), degenerate excluded")
    out: Dict[str, dict] = {}
    for d in sorted({r["direction"] for r in clean}, key=lambda x: 0 if x == "up" else 1):
        sub = [r for r in clean if r["direction"] == d]
        nv = [r for r in sub if not r["corrected"]]
        cr = [r for r in sub if r["corrected"]]
        frac, comp = rescue_fraction(nv, cr, key, higher_is_better)
        out[d] = {"rescue_fraction": frac, "comparable": comp, "n_corrected": len(cr),
                  "axis": key}
        print(f"  direction={d}: rescue={frac:.1%} of {comp} comparable "
              f"(n_corrected={len(cr)})")
        # the same number on the old axis, so the change is auditable rather than asserted
        old, old_comp = rescue_fraction(nv, cr, "pppl", False)
        print(f"    (for reference, the old pppl-based number was {old:.1%} of {old_comp})")
        out[d]["rescue_fraction_pppl_legacy"] = old
    run.log_summary(**{f"rescue_{key}": out})

    # ---- Pareto on the chosen axis ----
    style.apply_style()
    import matplotlib.pyplot as plt

    dirs = sorted({r["direction"] for r in clean}, key=lambda x: 0 if x == "up" else 1)
    fig, axes = plt.subplots(1, len(dirs), figsize=(6.5 * len(dirs), 5.2), squeeze=False)
    for ax, d in zip(axes[0], dirs):
        sub = [r for r in clean if r["direction"] == d]
        for corrected, colour, lab in ((False, style.ACCENT, "naive"),
                                       (True, style.method_color("glp"), "GLP-corrected")):
            cell = [r for r in sub if r["corrected"] == corrected and r[key] is not None]
            if not cell:
                continue
            ax.scatter([r[key] for r in cell], [r["concept_score"] for r in cell],
                       c=colour, alpha=0.35, s=18, label=f"{lab} (per-sample)")
            alphas = sorted({r["alpha"] for r in cell})
            pts = [(float(np.mean([r[key] for r in cell if r["alpha"] == a])),
                    float(np.mean([r["concept_score"] for r in cell if r["alpha"] == a])))
                   for a in alphas]
            xs, ys = zip(*sorted(pts))
            ax.plot(xs, ys, "o-", color=colour, linewidth=2, label=f"{lab} (mean per α)")
        if key in UBIQUITIN_REF:
            ax.axvline(UBIQUITIN_REF[key], color=style.BASELINE, linestyle=":", linewidth=1.4)
            ax.annotate("natural protein\n(ubiquitin)", xy=(UBIQUITIN_REF[key], ax.get_ylim()[1]),
                        xytext=(-6, -28), textcoords="offset points", ha="right", fontsize=9,
                        color=style.SECONDARY)
        ax.set_xlabel(f"{key} ({'higher' if higher_is_better else 'lower'} = more natural)")
        ax.set_ylabel("ΔGRAVY (final − init)")
        ax.set_title(f"Exp B Pareto on {key}" + (f" — {d} direction" if len(dirs) > 1 else ""),
                     loc="left")
        ax.legend(fontsize=9)
    path = style.save_fig(fig, run.figures_dir / f"pareto_{key}.png")
    print(f"\n[naturalness] wrote {path}")

    # ---- the gate-check figure: why this axis and not the others ----
    fig2, axs = plt.subplots(1, 3, figsize=(12.5, 4.4))
    for ax, k in zip(axs, ("pppl", "plddt", "ptm")):
        d = [r[k] for r in deg if r[k] is not None]
        c = [r[k] for r in clean if r[k] is not None]
        if not d or not c:
            continue
        ax.boxplot([c, d], widths=0.55, medianprops={"color": style.INK})
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["clean", "degenerate\n(non-protein)"])
        if k in UBIQUITIN_REF:
            ax.axhline(UBIQUITIN_REF[k], color=style.method_color("glp"), linestyle="--",
                       linewidth=1.6, label="natural protein (ubiquitin)")
            ax.legend(fontsize=8, loc="lower right")
        ax.set_title({"pppl": "pseudo-perplexity\n(lower = 'more natural')",
                      "plddt": "ESMFold2 pLDDT", "ptm": "ESMFold2 pTM"}[k], loc="left")
    fig2.suptitle("Do the naturalness axes reject known non-proteins?  "
                  "pppl and pLDDT do not; pTM does.", x=0.01, ha="left", fontsize=12,
                  fontweight="bold")
    path2 = style.save_fig(fig2, run.figures_dir / "naturalness_axis_gate_check.png")
    print(f"[naturalness] wrote {path2}")


if __name__ == "__main__":
    main()
