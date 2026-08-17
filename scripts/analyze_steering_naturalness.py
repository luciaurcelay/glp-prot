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


def pareto_front(pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Non-dominated subset of (x, y) under maximise-x, maximise-y, sorted by x ascending.

    This is the actual Pareto content of a "Pareto plot": a point survives only if no other
    point beats it on BOTH objectives. Drawn as a staircase it is the attainment front, and
    comparing two arms' fronts is the dominance statement — which a scatter plus a mean line
    does not make. Scanning from the largest x leftwards and keeping a running max of y is
    O(n log n) and needs no pairwise comparison."""
    best_y = -float("inf")
    front: List[Tuple[float, float]] = []
    for x, y in sorted(pts, key=lambda p: p[0], reverse=True):
        if y > best_y:
            front.append((x, y))
            best_y = y
    return list(reversed(front))


def hdr_levels(kde, xy: np.ndarray, fractions=(0.9, 0.5)) -> List[float]:
    """Density levels enclosing the given fractions of the sample (highest-density regions).

    Contouring at arbitrary density values says nothing interpretable; contouring at the
    level that encloses 50%/90% of the points does — it is the 2-D analogue of a quartile
    box, so two arms' contours are directly comparable even at different sample sizes."""
    d = np.sort(kde(xy))[::-1]
    total = d.sum()
    out = []
    for f in fractions:
        c = np.cumsum(d)
        idx = int(np.searchsorted(c, f * total))
        out.append(float(d[min(idx, len(d) - 1)]))
    return sorted(set(out))


def kde_1d(vals: List[float], grid: np.ndarray):
    """gaussian_kde over `vals`, or None when the sample can't support one."""
    from scipy.stats import gaussian_kde

    v = np.asarray([x for x in vals if x is not None], dtype=float)
    if len(v) < 5 or np.allclose(v.std(), 0):
        return None
    try:
        return gaussian_kde(v)(grid)
    except np.linalg.LinAlgError:
        return None


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
    ap.add_argument("--y-metric", default="attributable", choices=["attributable", "raw"],
                    help="concept axis. 'attributable' subtracts, per sample, the same init's "
                         "own alpha=0 run in the same arm and direction, so the plot shows "
                         "what STEERING did rather than what the correction does on its own "
                         "(that drift is 65-97%% of raw ΔGRAVY at alpha<=2). 'raw' is the "
                         "unadjusted final-minus-init")
    ap.add_argument("--neutral-alpha", type=float, default=0.0,
                    help="the no-steering alpha that 'attributable' differences against")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    rows, _ = load(run_dir, args.csv_name)
    run = RunLogger(run_dir)
    higher_is_better = args.axis != "pppl"

    # per-sample concept value for the plots. Differencing against the SAME init's alpha=0 run
    # within the same arm/direction removes both the correction's own no-steering drift and
    # per-init variation; rows whose baseline is missing (e.g. it was degenerate-filtered) get
    # None and drop out of the plots rather than silently comparing against nothing.
    base = {(r["direction"], r["corrected"], r.get("init_sequence_id")): r["concept_score"]
            for r in rows if r["alpha"] == args.neutral_alpha}
    n_nobase = 0
    for r in rows:
        if args.y_metric == "raw":
            r["_y"] = r["concept_score"]
            continue
        b = base.get((r["direction"], r["corrected"], r.get("init_sequence_id")))
        r["_y"] = None if b is None else r["concept_score"] - b
        n_nobase += b is None
    if args.y_metric == "attributable":
        print(f"[naturalness] y = steering-attributable ΔGRAVY "
              f"(alpha={args.neutral_alpha} baseline; {n_nobase} row(s) had no baseline)")

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

    # ---- Pareto: attainment front + densities, and the alpha path, as separate views ----
    style.apply_style()
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    ylab = ("steering-attributable ΔGRAVY" if args.y_metric == "attributable"
            else "ΔGRAVY (final − init)")
    xlab = f"{key} ({'higher' if higher_is_better else 'lower'} = more natural)"
    dirs = sorted({r["direction"] for r in clean}, key=lambda x: 0 if x == "up" else 1)
    arms = ((False, style.ACCENT, "naive"), (True, style.method_color("glp"), "GLP-corrected"))
    # where the sequences the filter REJECTED sit on this axis — the figure should carry its
    # own warning that a high score here is not automatically good
    junk = [r[key] for r in deg if r[key] is not None]

    for d in dirs:
        sub = [r for r in clean if r["direction"] == d
               and r[key] is not None and r["_y"] is not None]
        tag = "" if len(dirs) == 1 else f"_{d}"

        # --- view 1: joint plot, attainment fronts + 2-D HDR contours + marginal KDEs ---
        fig = plt.figure(figsize=(8.2, 7.4))
        gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                      hspace=0.04, wspace=0.04, figure=fig)
        ax = fig.add_subplot(gs[1, 0])
        axt = fig.add_subplot(gs[0, 0], sharex=ax)
        axr = fig.add_subplot(gs[1, 1], sharey=ax)
        for a in (axt, axr):
            a.tick_params(labelbottom=False, labelleft=False, length=0)
            for sp in a.spines.values():
                sp.set_visible(False)
            a.grid(False)

        if len(junk) >= 4:
            # IQR, not min-max: the full range of 23 points spans nearly the whole panel and
            # reads as alarm rather than information
            q1, q3 = np.percentile(junk, [25, 75])
            ax.axvspan(q1, q3, color=style.BASELINE, alpha=0.22, zorder=0)
            ax.annotate("middle 50% of the sequences\nthe filter rejected",
                        xy=((q1 + q3) / 2, 0.01), xycoords=("data", "axes fraction"),
                        ha="center", va="bottom", fontsize=8, color=style.SECONDARY)
        if key in UBIQUITIN_REF:
            ax.axvline(UBIQUITIN_REF[key], color=style.BASELINE, linestyle=":", linewidth=1.4)
            # rotated, pinned low: the previous version anchored this to the top of the axes
            # where it collided with the legend
            ax.annotate("natural protein (ubiquitin)", xy=(UBIQUITIN_REF[key], 0.02),
                        xycoords=("data", "axes fraction"), rotation=90, ha="right",
                        va="bottom", fontsize=8, color=style.SECONDARY)

        for corrected, colour, lab in arms:
            cell = [r for r in sub if r["corrected"] == corrected]
            if not cell:
                continue
            xs = np.array([r[key] for r in cell]); ys = np.array([r["_y"] for r in cell])
            ax.scatter(xs, ys, s=11, c=colour, alpha=0.28, linewidths=0, zorder=2)
            # 2-D HDR contours instead of relying on the scatter, which is a hairball at n~140
            if len(cell) >= 10:
                from scipy.stats import gaussian_kde
                try:
                    k = gaussian_kde(np.vstack([xs, ys]))
                    gx = np.linspace(xs.min(), xs.max(), 90)
                    gy = np.linspace(ys.min(), ys.max(), 90)
                    GX, GY = np.meshgrid(gx, gy)
                    Z = k(np.vstack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
                    lv = hdr_levels(k, np.vstack([xs, ys]))
                    if len(lv) >= 1:
                        ax.contour(GX, GY, Z, levels=lv, colors=[colour], linewidths=1.1,
                                   alpha=0.85, zorder=3)
                except np.linalg.LinAlgError:
                    pass
            # Front over per-alpha MEANS. A front over individual samples is a maximum
            # statistic — it is fixed by whichever few points happen to be extreme (4 of 146
            # here), i.e. by the noisiest part of the data. Per-alpha means are stable
            # estimates, so their non-dominated subset is the defensible dominance statement.
            means = []
            for a in sorted({r["alpha"] for r in cell}):
                c = [r for r in cell if r["alpha"] == a]
                if len(c) >= 2:
                    means.append((float(np.mean([r[key] for r in c])),
                                  float(np.mean([r["_y"] for r in c]))))
            front = pareto_front(means)
            if front:
                fx, fy = zip(*front)
                ax.step(fx, fy, where="post", color=colour, linewidth=2.4, zorder=4,
                        label=f"{lab} — front over per-α means ({len(front)}/{len(means)} α)")
                ax.plot(fx, fy, "o", ms=5.5, color=colour, zorder=5)
            # marginals
            gxm = np.linspace(min(xs), max(xs), 200)
            dv = kde_1d(list(xs), gxm)
            if dv is not None:
                axt.fill_between(gxm, dv, color=colour, alpha=0.28, linewidth=0)
                axt.plot(gxm, dv, color=colour, linewidth=1.4)
            gym = np.linspace(min(ys), max(ys), 200)
            dh = kde_1d(list(ys), gym)
            if dh is not None:
                axr.fill_betweenx(gym, dh, color=colour, alpha=0.28, linewidth=0)
                axr.plot(dh, gym, color=colour, linewidth=1.4)

        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.legend(loc="upper left", fontsize=9)
        axt.set_title(f"Exp B objective space on {key}"
                      + ("" if len(dirs) == 1 else f" — {d} direction")
                      + "\nstaircase = attainment front over per-α means; "
                        "contours = 50% / 90% of samples",
                      loc="left", fontsize=11)
        p1 = style.save_fig(fig, run.figures_dir / f"pareto_front_{key}{tag}.png")
        print(f"[naturalness] wrote {p1}")

        # --- view 2: the alpha path, ordered BY ALPHA (the previous line sorted by x, which
        # made the connecting segments meaningless) with SEM bars on both noisy objectives ---
        fig2, ax2 = plt.subplots(figsize=(7.4, 6.0))
        if len(junk) >= 4:
            q1, q3 = np.percentile(junk, [25, 75])
            ax2.axvspan(q1, q3, color=style.BASELINE, alpha=0.22, zorder=0)
            ax2.annotate("rejected\nsequences\n(middle 50%)", xy=((q1 + q3) / 2, 0.97),
                         xycoords=("data", "axes fraction"), ha="center", va="top",
                         fontsize=8, color=style.SECONDARY)
        for corrected, colour, lab in arms:
            cell = [r for r in sub if r["corrected"] == corrected]
            alphas_here = sorted({r["alpha"] for r in cell})
            mx, my, ex, ey, keep = [], [], [], [], []
            for a in alphas_here:
                c = [r for r in cell if r["alpha"] == a]
                if len(c) < 2:
                    continue
                xv = np.array([r[key] for r in c]); yv = np.array([r["_y"] for r in c])
                mx.append(xv.mean()); my.append(yv.mean())
                ex.append(xv.std(ddof=1) / np.sqrt(len(xv)))
                ey.append(yv.std(ddof=1) / np.sqrt(len(yv)))
                keep.append(a)
            if not mx:
                continue
            ax2.errorbar(mx, my, xerr=ex, yerr=ey, color=colour, linewidth=2, marker="o",
                         ms=6, capsize=2.5, elinewidth=1, label=f"{lab} (mean ± SEM per α)")
            # label selectively: the ends plus a few landmarks. Labelling all ten alphas
            # made the low-alpha cluster illegible (they pile up within ~0.05 pTM).
            show = {keep[0], keep[-1]} | {a for a in keep if a in (1.0, 2.0, 5.0)}
            xr = (max(mx) - min(mx)) or 1.0
            yr = (max(my) - min(my)) or 1.0
            placed: List[Tuple[float, float]] = []
            for a, x, y in zip(keep, mx, my):
                if a not in show:
                    continue
                # alpha=5 and alpha=20 can land within a hair of each other; drop the label
                # rather than overprint it
                if any(abs(x - px) / xr < 0.06 and abs(y - py) / yr < 0.06
                       for px, py in placed):
                    continue
                placed.append((x, y))
                ax2.annotate(f"α={a:g}", (x, y), xytext=(7, 6), textcoords="offset points",
                             fontsize=8.5, color=style.INK,
                             path_effects=[pe.withStroke(linewidth=2.6,
                                                         foreground=style.SURFACE)])
        ax2.set_xlabel(xlab)
        ax2.set_ylabel(ylab)
        ax2.set_title(f"Exp B dose-response path in objective space"
                      + ("" if len(dirs) == 1 else f" — {d} direction")
                      + "\nordered by α, not by naturalness", loc="left")
        ax2.legend(loc="upper left", fontsize=9)
        if key in UBIQUITIN_REF:
            # kept as text rather than a line: at pTM 0.74 it is far off the data range and a
            # line there leaves most of the panel empty
            ax2.annotate(f"natural protein (ubiquitin) sits at {key} = {UBIQUITIN_REF[key]:g},"
                         " off-scale right", xy=(0.99, 0.015), xycoords="axes fraction",
                         ha="right", va="bottom", fontsize=8, color=style.SECONDARY)
        p2 = style.save_fig(fig2, run.figures_dir / f"pareto_path_{key}{tag}.png")
        print(f"[naturalness] wrote {p2}")

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
    # Be precise: ALL THREE axes rate the known non-proteins better than our clean output.
    # pTM's only (but decisive) advantage is that it keeps a real protein ranked ABOVE the
    # junk, whereas pLDDT inverts even that ordering (junk 81 vs ubiquitin 80).
    fig2.suptitle("All three axes rate known non-proteins ABOVE our clean output.  "
                  "Only pTM still ranks a real protein above the junk.",
                  x=0.01, ha="left", fontsize=12, fontweight="bold")
    path2 = style.save_fig(fig2, run.figures_dir / "naturalness_axis_gate_check.png")
    print(f"[naturalness] wrote {path2}")


if __name__ == "__main__":
    main()
