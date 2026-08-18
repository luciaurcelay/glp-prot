#!/usr/bin/env python
"""The cross-property tables and figures the paper reports, from several finished Exp B runs.

One script rather than several because every number here is a comparison ACROSS properties,
and they have to be computed on one consistent row set (the same degeneracy gate, the same
steering-attributable differencing) or the table rows are not comparable to each other.

Emits, into `--out-run-dir`:

  metrics/summary.json     every number below, machine-readable
  tables/*.tex             LaTeX booktabs bodies, ready to \\input
  figures/decoding_fixed_point.png
  figures/alpha0_drift.png

and prints all of it. Three analyses:

**1. Efficacy.** Steering-attributable divergence per property per alpha, both arms. The
headline claim, and the reason the runs must share an `--alpha-mode relative` grid: raw
alpha is in units of 1/||v|| and the vectors have different norms.

**2. The decoding loop's fixed point.** Each run's natural starting proteins folded
(`heldout_starts_fold.csv`) against its own alpha=0 NAIVE arm — no steering, no GLP. With
six start sets spanning a wide pTM range this is a regression, not a ratio: the loop does
not scale foldability down, it pulls everything toward a fixed point, and quoting "it
roughly halves foldability" only worked because the first two start sets happened to sit
near twice it.

**3. The correction's own alpha=0 drift, per property.** A FIXED bias adds the same amount
from both starting sets; mean reversion flips sign with the set and cancels in the
corrected-minus-naive difference. Reporting both directions is what separates them, and
the three properties together test the single-compositional-bias account — including a
property where it predicts no effect.

Usage:
  python scripts/analyze_cross_property.py --run-dirs <gravy>,<charge>,<instability> \\
      --out-run-dir <dir> [--contrasts gravy=<path>,...]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import _bootstrap  # noqa: F401
import numpy as np

from glp_plm.eval.steering_properties import (_aa_frequencies, find_run_csv,
                                              is_degenerate_row, resolve_contrast, row_delta,
                                              run_property)
from glp_plm.experiments import RunLogger, style

PASS_PTM = 0.30   # "confidently folded"; reported as an OUTCOME, never used as a filter


def load(run_dir: Path, neutral_alpha: float, override: Optional[str]) -> dict:
    csv_path = find_run_csv(run_dir)
    raw = list(csv.DictReader(open(csv_path)))
    prop = run_property(raw, csv_path)

    fold: Dict[str, float] = {}
    fc = run_dir / "fold_confidence.csv"
    if fc.exists():
        by_id = {r["sample_id"]: r for r in raw}
        for f in csv.DictReader(open(fc)):
            if f.get("fold_error"):
                continue
            g = by_id.get(f["sample_id"])
            # the sidecar records the sha1 of what it folded; refuse to merge against
            # regenerated sequences rather than silently mixing two runs
            if not g or hashlib.sha1(g["sequence"].encode()).hexdigest()[:12] != f["seq_sha1"]:
                continue
            fold[f["sample_id"]] = float(f["ptm"])

    rows = []
    for r in raw:
        rows.append({"alpha": float(r["alpha"]), "corrected": r["corrected"] == "True",
                     "direction": r.get("direction") or "up", "init": r["init_sequence_id"],
                     "seq": r["sequence"], "delta": row_delta(r),
                     "ptm": fold.get(r["sample_id"]),
                     "degenerate": is_degenerate_row(r)})
    clean = [r for r in rows if not r["degenerate"]]
    base = {(r["direction"], r["corrected"], r["init"]): r["delta"]
            for r in clean if r["alpha"] == neutral_alpha}
    for r in clean:
        b = base.get((r["direction"], r["corrected"], r["init"]))
        r["attr"] = None if b is None else r["delta"] - b

    natural: Dict[str, List[float]] = {}
    hs = run_dir / "heldout_starts_fold.csv"
    if hs.exists():
        for f in csv.DictReader(open(hs)):
            if not f.get("fold_error"):
                natural.setdefault(f["direction"], []).append(float(f["ptm"]))

    contrast = resolve_contrast(run_dir, prop.key, override)
    sd = (contrast.get("pool_property_stats") or {}).get(prop.key, {}).get("sd")
    starts = {("up", str(h["id"])): h["sequence"] for h in contrast.get("heldout") or []}
    starts.update({("down", str(h["id"])): h["sequence"]
                   for h in contrast.get("heldout_high") or []})
    return {"prop": prop, "rows": rows, "clean": clean, "natural": natural,
            "pool_sd": sd, "starts": starts, "name": run_dir.name}


def cell(rows, d, corrected, alpha, field="attr"):
    return [r[field] for r in rows if r["direction"] == d and r["corrected"] == corrected
            and r["alpha"] == alpha and r.get(field) is not None]


def divergence(rows, alpha, corrected) -> Optional[Tuple[float, float, int]]:
    from scipy.stats import mannwhitneyu
    u, d = cell(rows, "up", corrected, alpha), cell(rows, "down", corrected, alpha)
    if len(u) < 3 or len(d) < 3:
        return None
    return float(np.mean(u) - np.mean(d)), float(mannwhitneyu(u, d).pvalue), min(len(u), len(d))


def tex_table(path: Path, header: str, body: List[str], caption: str, label: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([
        r"% generated by scripts/analyze_cross_property.py — do not edit by hand",
        r"\begin{table}[htbp]", r"\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        header, r"\toprule", *body, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]))
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", required=True, help="comma-separated, one per property")
    ap.add_argument("--out-run-dir", required=True)
    ap.add_argument("--contrasts", default="",
                    help="comma-separated <property>=<path>, for runs whose contrast lives "
                         "outside the run dir and predates `contrast_path` in summary.json")
    ap.add_argument("--neutral-alpha", type=float, default=0.0)
    ap.add_argument("--no-titles", action="store_true")
    args = ap.parse_args()

    overrides = dict(kv.split("=", 1) for kv in args.contrasts.split(",") if "=" in kv)
    runs = []
    for d in args.run_dirs.split(","):
        d = Path(d.strip())
        key = run_property(list(csv.DictReader(open(find_run_csv(d)))), find_run_csv(d)).key
        runs.append(load(d, args.neutral_alpha, overrides.get(key)))
    run = RunLogger(Path(args.out_run_dir))
    tables = Path(args.out_run_dir) / "tables"
    summary: Dict[str, dict] = {}

    # ---------------- 1. efficacy ----------------
    print("\n=== 1. STEERING-ATTRIBUTABLE DIVERGENCE (up - down), by property and alpha ===")
    alphas = sorted({r["alpha"] for R in runs for r in R["clean"]} - {args.neutral_alpha})
    eff: Dict[str, dict] = {}
    body = []
    for R in runs:
        p = R["prop"]
        eff[p.key] = {}
        print(f"\n  {p.label}   (pool SD {R['pool_sd']:.4f})" if R["pool_sd"] else f"\n  {p.label}")
        print(f"  {'alpha':>8} | {'naive':>12} {'p':>9} | {'corrected':>12} {'p':>9} | {'corr/SD':>8}")
        for a in alphas:
            e = {}
            for corrected, nm in ((False, "naive"), (True, "corrected")):
                dv = divergence(R["clean"], a, corrected)
                if dv:
                    e[nm] = {"divergence": dv[0], "p": dv[1], "n": dv[2],
                             "sd_units": dv[0] / R["pool_sd"] if R["pool_sd"] else None}
            if not e:
                continue
            eff[p.key][f"{a:g}"] = e
            n_, c_ = e.get("naive"), e.get("corrected")
            print(f"  {a:>8.4f} | {n_['divergence']:>+12.4f} {n_['p']:>9.1e} | "
                  f"{c_['divergence']:>+12.4f} {c_['p']:>9.1e} | "
                  + (f"{c_['sd_units']:>8.2f}" if c_["sd_units"] else f"{'--':>8}"))
            body.append(f"{p.key} & {a:g} & ${n_['divergence']:+.3f}$ & ${n_['p']:.0e}$ & "
                        f"${c_['divergence']:+.3f}$ & ${c_['p']:.0e}$ & "
                        + (f"${c_['sd_units']:+.2f}$" if c_["sd_units"] else "---") + r" \\")
    summary["efficacy"] = eff
    tex_table(tables / "efficacy.tex",
              r"\begin{tabular}{llrrrrr}" + "\n"
              r"property & $\alpha$ & naive & $p$ & GLP-corrected & $p$ & SD \\",
              body,
              "Steering-attributable divergence (up $-$ down) by property and steering "
              "strength, both arms. $\\alpha$ is in relative units (fraction of "
              "mean$\\|h\\|$), shared across properties. SD is the corrected divergence in "
              "units of the property's UniRef50 spread. Unpaired Mann--Whitney.",
              "tab:steer-efficacy")

    # ---------------- 2. the decoding loop's fixed point ----------------
    print("\n=== 2. THE DECODING LOOP'S FIXED POINT (alpha=0 NAIVE arm: no steering, no GLP) ===")
    pts, loop = [], []
    print(f"  {'start set':>28} {'natural':>9} {'after loop':>11} {'lost':>8} {'lost %':>8}")
    for R in runs:
        for d in ("up", "down"):
            nat = R["natural"].get(d)
            post = cell(R["clean"], d, False, args.neutral_alpha, "ptm")
            if not nat or not post:
                continue
            n, q = float(np.mean(nat)), float(np.mean(post))
            lab = f"{R['prop'].key}, {R['prop'].low_name if d == 'up' else R['prop'].high_name}"
            pts.append((n, q, lab))
            loop.append({"set": lab, "natural": n, "post_loop": q, "n_natural": len(nat),
                         "n_post": len(post)})
            print(f"  {lab:>28} {n:>9.3f} {q:>11.3f} {q - n:>+8.3f} {(q - n) / n:>7.0%}")
    if len(pts) >= 3:
        nat = np.array([p[0] for p in pts]); post = np.array([p[1] for p in pts])
        b, a = np.polyfit(nat, post, 1)
        r = float(np.corrcoef(nat, post)[0, 1])
        fixed = a / (1 - b)
        r_loss = float(np.corrcoef(nat, post - nat)[0, 1])
        print(f"\n  post = {a:.3f} + {b:.3f} x start   (r={r:+.2f}, n={len(pts)} start sets)")
        print(f"  -> only {b:.0%} of a start set's foldability advantage survives; "
              f"fixed point pTM {fixed:.3f}")
        print(f"  -> correlation between how well a set folds and how much it loses: "
              f"r={r_loss:+.2f}")
        summary["decoding_fixed_point"] = {"sets": loop, "slope": float(b),
                                           "intercept": float(a), "r": r,
                                           "fixed_point": float(fixed), "r_loss": r_loss}
        tex_table(tables / "decoding_loop.tex",
                  r"\begin{tabular}{lrrr}" + "\n"
                  r"starting set & natural pTM & after decoding & change \\",
                  [f"{e['set']} & ${e['natural']:.3f}$ & ${e['post_loop']:.3f}$ & "
                   f"${(e['post_loop'] - e['natural']) / e['natural'] * 100:+.0f}\\%$ \\\\"
                   for e in sorted(loop, key=lambda x: -x["natural"])],
                  f"What masked-Gibbs decoding costs, before any steering or correction "
                  f"($\\alpha{{=}}0$, naive arm). Fitting the six sets gives "
                  f"$\\mathrm{{post}} = {a:.3f} + {b:.3f}\\,\\mathrm{{start}}$ ($r={r:+.2f}$): "
                  f"a fixed point at pTM ${fixed:.3f}$ rather than a constant penalty.",
                  "tab:steer-decoding")

    # ---------------- 3. the correction's own alpha=0 drift ----------------
    print("\n=== 3. THE CORRECTION'S OWN alpha=0 DRIFT (fixed bias vs mean reversion) ===")
    print(f"  {'property':>14} {'poolSD':>8} | {'naive up':>10} {'naive dn':>10} | "
          f"{'corr up':>10} {'corr dn':>10} | {'fixed bias':>22}")
    drift: Dict[str, dict] = {}
    for R in runs:
        p, sd = R["prop"], R["pool_sd"] or 1.0
        m = {(d, c): np.mean(cell(R["clean"], d, c, args.neutral_alpha, "delta") or [np.nan])
             for d in ("up", "down") for c in (False, True)}
        bu, bd = m[("up", True)] - m[("up", False)], m[("down", True)] - m[("down", False)]
        drift[p.key] = {"naive_up": float(m[("up", False)]), "naive_down": float(m[("down", False)]),
                        "corrected_up": float(m[("up", True)]), "corrected_down": float(m[("down", True)]),
                        "fixed_bias_up": float(bu), "fixed_bias_down": float(bd),
                        "fixed_bias_sd": float((bu + bd) / 2 / sd)}
        print(f"  {p.key:>14} {sd:>8.4f} | {m[('up', False)]:>+10.3f} {m[('down', False)]:>+10.3f} | "
              f"{m[('up', True)]:>+10.3f} {m[('down', True)]:>+10.3f} | "
              f"up {bu:>+8.3f} down {bd:>+8.3f} ({(bu + bd) / 2 / sd:+.2f} SD)")
    summary["alpha0_drift"] = drift
    tex_table(tables / "alpha0_drift.tex",
              r"\begin{tabular}{lrrrrr}" + "\n"
              r"property & naive up & naive down & corrected up & corrected down & fixed bias (SD) \\",
              [f"{k} & ${v['naive_up']:+.3f}$ & ${v['naive_down']:+.3f}$ & "
               f"${v['corrected_up']:+.3f}$ & ${v['corrected_down']:+.3f}$ & "
               f"${v['fixed_bias_sd']:+.2f}$ \\\\" for k, v in drift.items()],
              "Concept change with \\emph{no} steering applied ($\\alpha{=}0$), by arm and "
              "starting set. A fixed bias adds the same amount from both sets; mean "
              "reversion flips sign with the set and cancels in the corrected$-$naive "
              "difference, which is the last column.",
              "tab:steer-drift")

    # composition, which is what the drift actually is
    comp = {}
    for R in runs:
        sub = [r for r in R["clean"] if r["alpha"] == args.neutral_alpha and r["corrected"]
               and (r["direction"], r["init"]) in R["starts"]]
        if not sub:
            continue
        f0 = _aa_frequencies([R["starts"][(r["direction"], r["init"])] for r in sub])
        f1 = _aa_frequencies([r["seq"] for r in sub])
        comp[R["prop"].key] = {a: (f1[a] - f0[a]) * 100 for a in f0}
    if comp:
        any_key = next(iter(comp))
        top = sorted(comp[any_key], key=lambda a: -abs(comp[any_key][a]))[:6]
        print("\n  composition change at alpha=0 (corrected arm), percentage points:")
        for k, v in comp.items():
            print(f"    {k:>14}: " + "  ".join(f"{a} {v[a]:+.1f}" for a in top))
        summary["alpha0_composition_pp"] = comp

    # ---------------- figures ----------------
    style.apply_style()
    import matplotlib.pyplot as plt

    if "decoding_fixed_point" in summary:
        fig, ax = plt.subplots(figsize=(6.6, 5.2))
        lim = [0.15, 0.70]
        ax.plot(lim, lim, color=style.BASELINE, linestyle="--", linewidth=1.2)
        ax.annotate("no change", xy=(lim[1], lim[1]), xytext=(-4, -12),
                    textcoords="offset points", ha="right", fontsize=8.5, color=style.SECONDARY)
        xs = np.linspace(*lim, 50)
        ax.plot(xs, a + b * xs, color=style.PURPLE_MID, linewidth=2)
        ax.scatter(nat, post, s=64, color=style.PURPLE_DARK, zorder=3)
        for (n_, q_, lab) in pts:
            ax.annotate(lab, (n_, q_), xytext=(6, -3), textcoords="offset points",
                        fontsize=8, color=style.SECONDARY)
        ax.scatter([fixed], [fixed], marker="*", s=200, color=style.ORANGE_MID, zorder=4)
        ax.annotate(f"fixed point {fixed:.2f}", (fixed, fixed), xytext=(8, 8),
                    textcoords="offset points", fontsize=9, color=style.ORANGE_DARK)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("natural starting proteins (ESMFold2 pTM)")
        ax.set_ylabel("after masked-Gibbs decoding, no steering (pTM)")
        if not args.no_titles:
            ax.set_title("Foldability before and after the decoding loop", loc="left")
        print()
        print(f"  wrote {style.save_fig(fig, run.figures_dir / 'decoding_fixed_point.png')}")

    if drift:
        fig2, ax2 = plt.subplots(figsize=(7.6, 4.6))
        keys = list(drift)
        x = np.arange(len(keys)); w = 0.2
        sds = {R["prop"].key: (R["pool_sd"] or 1.0) for R in runs}
        series = [("naive_up", "naive, from low set", style.ORANGE_MID, "//"),
                  ("naive_down", "naive, from high set", style.ORANGE_LIGHT, "//"),
                  ("corrected_up", "corrected, from low set", style.PURPLE_DARK, None),
                  ("corrected_down", "corrected, from high set", style.PURPLE_LIGHT, None)]
        for i, (field, lab, col, hatch) in enumerate(series):
            ax2.bar(x + (i - 1.5) * w, [drift[k][field] / sds[k] for k in keys], w,
                    color=col, label=lab, edgecolor=style.SURFACE, linewidth=1, hatch=hatch)
        ax2.axhline(0, color=style.INK, linewidth=1)
        ax2.set_xticks(x); ax2.set_xticklabels(keys)
        ax2.set_ylabel("change with NO steering\n(natural SD units)")
        ax2.legend(fontsize=8.5, ncol=2)
        if not args.no_titles:
            ax2.set_title("What the correction does at zero steering strength", loc="left")
        print(f"  wrote {style.save_fig(fig2, run.figures_dir / 'alpha0_drift.png')}")

    run.log_summary(**summary)
    print(f"\n[cross-property] metrics -> {run.metrics_dir}/summary.json")


if __name__ == "__main__":
    main()
