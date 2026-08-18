#!/usr/bin/env python
"""Publication figures for a diff-of-means steering run (any steered property).

Replaces the Pareto-framed figures. A Pareto front compares solutions you cannot index; here
the solutions ARE indexed, by the steering strength alpha, so the set of achievable
(naturalness, effect) pairs simply is the dose-response curve and the "front" is that curve
with the dominated alphas deleted — optimisation vocabulary over a filtered line plot. Worse,
the trade-off it was drawn to show does not exist in these data: structural confidence barely
moves across alpha or arm, so a 2-D objective-space plot spends its second axis on noise and
becomes unreadable (overlapping error bars, colliding labels, one arm a tangle).

What the figures need to say instead:
  1. steering works with the correction and not without it   -> divergence figure
  2. the correction carries its own bias, which must be subtracted -> baseline figure
  3. the effect grows while naturalness does not             -> stacked dose-response
Point 3 is the dominance claim, made by showing the naturalness panel flat rather than by
computing a front.

Figure titles avoid internal names ("Exp B") and bare metric tokens ("ptm").
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import _bootstrap  # noqa: F401
import numpy as np

from glp_plm.eval.steering_properties import (SteeringProperty, find_run_csv,
                                              row_delta, run_property)
from glp_plm.experiments import RunLogger, style

ARMS = ((False, "naive"), (True, "GLP-corrected"))


def is_degenerate(row: dict) -> bool:
    seq = row.get("sequence") or ""
    if seq and max(Counter(seq).values()) / len(seq) >= 0.30:
        return True
    return (int(float(row["unique_aa_count"])) <= 8
            or int(float(row["max_homopolymer_run"])) >= 12)   # 12, not 8: 8 rejected a real
            # protein with a run of 9 (see run_steering.is_degenerate_sequence)


def load_rows(run_dir: Path, neutral_alpha: float, apply_gate: bool = True
              ) -> Tuple[List[dict], SteeringProperty]:
    path = find_run_csv(run_dir)
    raw = [r for r in csv.DictReader(open(path))]
    prop = run_property(raw, path)
    rows = []
    for r in raw:
        if apply_gate and is_degenerate(r):
            continue          # degenerate output is excluded from effect estimates
        rows.append({"alpha": float(r["alpha"]), "corrected": r["corrected"] == "True",
                     "direction": r.get("direction") or "up",
                     "init": r["init_sequence_id"], "dprop": row_delta(r)})
    # per-sample steering-attributable effect: subtract this init's own no-steering run in the
    # same arm and direction, removing the correction's drift and per-init variation
    base = {(r["direction"], r["corrected"], r["init"]): r["dprop"]
            for r in rows if r["alpha"] == neutral_alpha}
    for r in rows:
        b = base.get((r["direction"], r["corrected"], r["init"]))
        r["attr"] = None if b is None else r["dprop"] - b
    if apply_gate:
        print(f"[report] {len(rows)}/{len(raw)} non-degenerate rows "
              f"(steered property: {prop.label})")
    return rows, prop


def cell_stats(rows, direction, corrected, alpha, field) -> Optional[Tuple[float, float, int]]:
    v = [r[field] for r in rows if r["direction"] == direction and r["corrected"] == corrected
         and r["alpha"] == alpha and r[field] is not None]
    if len(v) < 2:
        return None
    a = np.asarray(v, dtype=float)
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a))), len(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--neutral-alpha", type=float, default=0.0)
    ap.add_argument("--property-label", default=None,
                    help="override the axis label for the steered property; by default it "
                         "is read from the run's own CSV (the `property` column)")
    ap.add_argument("--no-titles", action="store_true",
                    help="omit in-figure titles entirely, for journals that want the figure "
                         "bare and all description in the caption")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    rows, prop = load_rows(run_dir, args.neutral_alpha)
    prop_label = args.property_label or prop.label
    run = RunLogger(run_dir)
    alphas = sorted({r["alpha"] for r in rows})
    dirs = sorted({r["direction"] for r in rows}, key=lambda d: 0 if d == "up" else 1)

    style.apply_style()
    import matplotlib.pyplot as plt

    def title(ax, text, **kw):
        """Titles state WHAT IS PLOTTED, never what it means.

        Interpretation belongs in the paper's caption, which can be edited without
        re-rendering, and a claim baked into a PNG can end up contradicting its own panels
        when the data changes — which happened here once already. Axis labels, reference
        markers and unit notes stay, since those are needed to read the figure at all."""
        if not args.no_titles:
            ax.set_title(text, loc="left", **kw)

    # alpha is plotted on evenly-spaced categorical ticks. The grid (0, 0.25, .., 5) is
    # geometric-ish with a zero in it, so a linear axis crushes the low end and a log axis
    # cannot show 0 at all; symlog can, but "symlog" is jargon on a published axis.
    xpos = {a: i for i, a in enumerate(alphas)}
    xlabels = [f"{a:g}" for a in alphas]
    colour = {c: style.steering_arm_color(c) for c in (False, True)}

    # ---------------- Figure 1: bidirectional divergence ----------------
    if len(dirs) > 1:
        fig, ax = plt.subplots(figsize=(7.8, 5.6))
        ax.axhline(0, color=style.BASELINE, linewidth=1)
        for corrected, lab in ARMS:
            for d, ls, mk in (("up", "-", "o"), ("down", "--", "s")):
                xs, ys, es = [], [], []
                for a in alphas:
                    st = cell_stats(rows, d, corrected, a, "attr")
                    if st:
                        xs.append(xpos[a]); ys.append(st[0]); es.append(st[1])
                if xs:
                    ax.errorbar(xs, ys, yerr=es, color=colour[corrected], linestyle=ls,
                                marker=mk, ms=5.5, linewidth=2, capsize=2.5, elinewidth=1,
                                label=f"{lab} — steered {d}")
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels(xlabels)
        ax.set_xlabel("steering strength α")
        ax.set_ylabel(f"change in {prop_label}\n(steering-attributable, mean ± SEM)")
        title(ax, f"Change in {prop_label} vs. steering strength, both directions")
        ax.legend(fontsize=9, loc="upper left")
        ax.annotate(f"solid = steered toward {prop.high_name}\ndashed = steered away",
                    xy=(0.015, 0.03), xycoords="axes fraction", ha="left", fontsize=8.5,
                    color=style.SECONDARY)
        p = style.save_fig(fig, run.figures_dir / "fig_divergence.png")
        print(f"[report] wrote {p}")

    # ---------------- Figure 2: the no-steering baseline, by direction ----------------
    fig2, ax2 = plt.subplots(figsize=(7.2, 4.8))
    width = 0.34
    xs = np.arange(len(dirs))
    for i, (corrected, lab) in enumerate(ARMS):
        vals, errs = [], []
        for d in dirs:
            st = cell_stats(rows, d, corrected, args.neutral_alpha, "dprop")
            vals.append(st[0] if st else np.nan)
            errs.append(st[1] if st else np.nan)
        ax2.bar(xs + (i - 0.5) * width, vals, width, yerr=errs, capsize=3,
                color=colour[corrected], label=lab, edgecolor=style.SURFACE, linewidth=1)
    ax2.axhline(0, color=style.INK, linewidth=1)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"started {prop.low_name}\n(low held-out set)" if d == "up"
                         else f"started {prop.high_name}\n(high held-out set)" for d in dirs])
    ax2.set_ylabel(f"change in {prop_label}\nwith NO steering applied (mean ± SEM)")
    # keep each title line short enough to sit inside the axes at this width — a long second
    # line runs past the panel and reads as a stray break
    title(ax2, f"Change in {prop_label} with no steering applied")
    ax2.legend(fontsize=9)
    p2 = style.save_fig(fig2, run.figures_dir / "fig_baseline_drift.png")
    print(f"[report] wrote {p2}")

    # ---------------- Figure 3: effect and naturalness, stacked on a shared alpha axis ----
    side = run_dir / "fold_confidence.csv"
    fold: Dict[str, dict] = {}
    if side.exists():
        for f in csv.DictReader(open(side)):
            if not f["fold_error"]:
                fold[f["sample_id"]] = f
    have_fold = bool(fold)
    if have_fold:
        # attach pTM by sample_id
        by_id = {r["sample_id"]: r for r in csv.DictReader(open(find_run_csv(run_dir)))}
        for r in rows:
            r["ptm"] = None
        keyed = {(rr["direction"], rr["corrected"], rr["init"], rr["alpha"]): rr for rr in rows}
        for sid, f in fold.items():
            g = by_id.get(sid)
            if not g:
                continue
            k = (g.get("direction") or "up", g["corrected"] == "True",
                 g["init_sequence_id"], float(g["alpha"]))
            if k in keyed:
                try:
                    keyed[k]["ptm"] = float(f["ptm"])
                except (TypeError, ValueError):
                    pass

    nrow = 2 if have_fold else 1
    fig3, axs = plt.subplots(nrow, 1, figsize=(7.4, 3.4 * nrow + 1.4), sharex=True,
                             squeeze=False)
    d0 = dirs[0]
    axA = axs[0][0]
    axA.axhline(0, color=style.BASELINE, linewidth=1)
    for corrected, lab in ARMS:
        xs2, ys2, es2 = [], [], []
        for a in alphas:
            st = cell_stats(rows, d0, corrected, a, "attr")
            if st:
                xs2.append(xpos[a]); ys2.append(st[0]); es2.append(st[1])
        axA.errorbar(xs2, ys2, yerr=es2, color=colour[corrected], marker="o", ms=5.5,
                     linewidth=2, capsize=2.5, elinewidth=1, label=lab)
    axA.set_ylabel(f"change in\n{prop_label}")
    # the second line identifies WHICH subset is plotted, which is needed to read the panel;
    # it is description, not interpretation
    title(axA, f"Change in {prop_label} and structural confidence vs. steering strength\n"
               f"sequences started {prop.low_name}, steered toward {prop.high_name}")
    axA.legend(fontsize=9, loc="upper left")

    if have_fold:
        axB = axs[1][0]
        for corrected, lab in ARMS:
            xs3, ys3, es3 = [], [], []
            for a in alphas:
                st = cell_stats(rows, d0, corrected, a, "ptm")
                if st:
                    xs3.append(xpos[a]); ys3.append(st[0]); es3.append(st[1])
            if xs3:
                axB.errorbar(xs3, ys3, yerr=es3, color=colour[corrected], marker="o", ms=5.5,
                             linewidth=2, capsize=2.5, elinewidth=1, label=lab)
        # No reference line for a natural protein here: a benchmark drawn on the panel is an
        # interpretive anchor, and interpretation belongs in the caption. The value is in the
        # run's summary.json if the caption wants to quote it.
        axB.set_ylabel("structural confidence\n(ESMFold2 pTM)")
        axB.set_ylim(0, 0.85)
    axs[-1][0].set_xticks(range(len(alphas)))
    axs[-1][0].set_xticklabels(xlabels)
    axs[-1][0].set_xlabel("steering strength α")
    p3 = style.save_fig(fig3, run.figures_dir / "fig_dose_response.png")
    print(f"[report] wrote {p3}")
    if not have_fold:
        print("[report] (no fold_confidence.csv yet — naturalness panel omitted)")

    # ---------------- divergence, with a test ----------------
    # The headline number was being reported as a difference of means with no test. The two
    # directions use DIFFERENT held-out sequences, so this is unpaired between directions —
    # Mann-Whitney on the two arms' per-sample attributable values, not a paired test.
    if len(dirs) > 1:
        from scipy.stats import mannwhitneyu

        diverg: Dict[str, dict] = {}
        print("\n[report] divergence between directions (Mann-Whitney, unpaired: the two "
              "directions use different held-out sequences)")
        for a in alphas:
            if a == args.neutral_alpha:
                continue
            entry = {}
            for corrected, lab in ARMS:
                u = [r["attr"] for r in rows if r["direction"] == "up"
                     and r["corrected"] == corrected and r["alpha"] == a and r["attr"] is not None]
                d = [r["attr"] for r in rows if r["direction"] == "down"
                     and r["corrected"] == corrected and r["alpha"] == a and r["attr"] is not None]
                if len(u) < 3 or len(d) < 3:
                    continue
                try:
                    pv = float(mannwhitneyu(u, d).pvalue)
                except ValueError:
                    pv = float("nan")
                entry[lab] = {"up_mean": float(np.mean(u)), "down_mean": float(np.mean(d)),
                              "divergence": float(np.mean(u) - np.mean(d)),
                              "n_up": len(u), "n_down": len(d), "mannwhitney_p": pv}
            if entry:
                diverg[f"{a:g}"] = entry
                bits = "  ".join(f"{k}: {v['divergence']:+.3f} (p={v['mannwhitney_p']:.2e})"
                                 for k, v in entry.items())
                print(f"  α={a:<5} {bits}")
        run.log_summary(divergence_tested=diverg)

        # The quality gate is applied AFTER generation and complexity is affected by the
        # treatment, so gating is post-treatment selection: it preferentially removes
        # successfully-steered sequences. The gate here is mild (~3%) and the two estimates
        # agree, which is a stronger statement than choosing one — so report both.
        ungated, _ = load_rows(run_dir, args.neutral_alpha, apply_gate=False)
        ung: Dict[str, dict] = {}
        print("\n[report] same divergence WITHOUT the quality gate (the gate is "
              "post-treatment; agreement between the two is the point)")
        for a in alphas:
            if a == args.neutral_alpha:
                continue
            entry = {}
            for corrected, lab in ARMS:
                u = [r["attr"] for r in ungated if r["direction"] == "up"
                     and r["corrected"] == corrected and r["alpha"] == a and r["attr"] is not None]
                d = [r["attr"] for r in ungated if r["direction"] == "down"
                     and r["corrected"] == corrected and r["alpha"] == a and r["attr"] is not None]
                if len(u) >= 3 and len(d) >= 3:
                    entry[lab] = {"divergence": float(np.mean(u) - np.mean(d)),
                                  "n_up": len(u), "n_down": len(d)}
            if entry:
                ung[f"{a:g}"] = entry
        for a, e in ung.items():
            g = diverg.get(a, {})
            bits = "  ".join(
                f"{k}: ungated {v['divergence']:+.3f} vs gated "
                f"{g.get(k, {}).get('divergence', float('nan')):+.3f}" for k, v in e.items())
            print(f"  α={a:<5} {bits}")
        run.log_summary(divergence_ungated=ung)

    write_run_summary(run, run_dir, args, prop_label)


def write_run_summary(run, run_dir: Path, args, prop_label: str) -> None:
    """Emit run_summary.md from summary.json — the repo's per-run write-up convention.

    Reads the logged metrics rather than recomputing, so the summary cannot drift from what
    the pipeline actually recorded."""
    s = run.load_summary()

    def g(*keys, default=None):
        cur = s
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    L = [f"# {run_dir.name}", ""]
    L += [f"- git commit: `{s.get('git_commit', '?')}`"
          f"{' (dirty)' if s.get('git_dirty') else ''}",
          f"- steering property: {prop_label}", ""]

    L += ["## Steering effect (attributable, i.e. each arm differenced against its own α=0)", ""]
    att = g("steering_attributable_by_direction") or {"up": g("steering_attributable", default={})}
    for d, per_alpha in (att or {}).items():
        L += [f"### direction: {d}", "",
              "| α | naive | GLP-corrected | gap | gap p |", "|---|---|---|---|---|"]
        for a, e in (per_alpha or {}).items():
            def f(k):
                v = e.get(k)
                return "—" if not v else f"{v['mean']:+.3f} (n={v['n']})"
            gp = (e.get("gap") or {}).get("wilcoxon_p")
            L += [f"| {a} | {f('naive')} | {f('corrected')} | "
                  f"{(e.get('gap') or {}).get('mean', float('nan')):+.3f} | "
                  f"{'—' if gp is None else f'{gp:.2e}'} |"]
        L += [""]

    dv = g("divergence_tested")
    if dv:
        L += ["## Divergence between steering directions (unpaired Mann-Whitney)", "",
              "| α | naive | p | GLP-corrected | p |", "|---|---|---|---|---|"]
        for a, e in dv.items():
            n_, c_ = e.get("naive"), e.get("GLP-corrected")
            L += [f"| {a} | {n_['divergence']:+.3f} | {n_['mannwhitney_p']:.2e} | "
                  f"{c_['divergence']:+.3f} | {c_['mannwhitney_p']:.2e} |"
                  if n_ and c_ else f"| {a} | — | — | — | — |"]
        L += [""]

    gate = g("naturalness_axis_gate_check")
    if gate:
        L += ["## Do the naturalness axes reject known non-proteins?", "",
              "| axis | rejected-sequence mean | kept mean | p | verdict |",
              "|---|---|---|---|---|"]
        for k, v in gate.items():
            L += [f"| {k} | {v['degenerate_mean']:.3f} | {v['clean_mean']:.3f} | "
                  f"{v['mannwhitney_p']:.1e} | **{v['verdict']}** |"]
        L += ["", "The composition filter, not any of these metrics, is what keeps degenerate "
              "output out of the results.", ""]

    pr = g(f"pass_rate_ptm", "by_alpha")
    if pr:
        L += ["## Confidently-folded yield (non-degenerate AND above threshold)", "",
              "| α | naive | GLP-corrected | Fisher p |", "|---|---|---|---|"]
        thr = (g("pass_rate_ptm", "thresholds") or [0.4])
        t = thr[len(thr) // 2]
        for a, e in pr.items():
            nv, cr = e.get(f"naive_pass_{t:g}"), e.get(f"corrected_pass_{t:g}")
            pv = e.get(f"fisher_p_{t:g}")
            L += [f"| {a} | {'—' if nv is None else f'{nv:.1%}'} | "
                  f"{'—' if cr is None else f'{cr:.1%}'} | "
                  f"{'—' if pv is None else f'{pv:.3f}'} |"]
        L += ["", f"(threshold pTM ≥ {t:g}; every generation counted, so degenerate output "
              "counts as a failure)", ""]

    out = run_dir / "run_summary.md"
    out.write_text("\n".join(L) + "\n")
    print(f"[report] wrote {out}")


if __name__ == "__main__":
    main()
