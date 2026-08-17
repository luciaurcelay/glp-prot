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


def pass_rate_by_alpha(rows: List[dict], key: str, thresholds: List[float],
                       higher_is_better: bool = True) -> Dict[str, dict]:
    """Fraction of ALL generations in each (alpha, arm) cell that clear a confidence threshold.

    This is the lab-standard "filter at pLDDT>=70" idea turned into an OUTCOME instead of a
    filter, which matters because using it as a filter here does two bad things:

      1. It selects FOR the artefact. On this run, pLDDT>=70 keeps 87% of the degenerate
         poly-Leu/Val sequences and only 12% of the clean ones, taking a set that is 8% junk
         to 38% junk (73% at >=80). The conventional cutoff assumes pLDDT tracks quality; on
         these sequences it is inverted.
      2. The arms survive at different rates (naive ~17% vs corrected ~7.5% overall, 40% vs
         13% at alpha=2), so comparing survivors compares the best 13% of one arm against the
         best 40% of the other — survivor bias, the same trap the degenerate filter created
         at t_start=0.5.

    As a rate it has neither problem: the denominator is every sequence generated, and a
    sequence counts as passing only if it is BOTH non-degenerate AND above threshold — so the
    junk counts as failure rather than being quietly excluded. It answers the question the lab
    practice is really asking ("how often does this method produce something confidently
    foldable?") and generating more sequences buys precision with no selection effect.

    Reported at several thresholds because any single cutoff is arbitrary."""
    out: Dict[str, dict] = {}
    for a in sorted({r["alpha"] for r in rows}):
        entry: Dict[str, object] = {}
        for corrected, name in ((False, "naive"), (True, "corrected")):
            cell = [r for r in rows if r["alpha"] == a and r["corrected"] == corrected]
            if not cell:
                continue
            entry[f"{name}_n_generated"] = len(cell)
            for thr in thresholds:
                ok = 0
                for r in cell:
                    v = r.get(key)
                    if r["_degenerate"] or v is None:
                        continue   # degenerate counts as a failure, not an exclusion
                    ok += (v >= thr) if higher_is_better else (v <= thr)
                entry[f"{name}_pass_{thr:g}"] = ok / len(cell)
        if entry:
            out[str(a)] = entry
    return out


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
        # Test the verdict rather than reading it off two means: a bare inequality calls a
        # 0.278-vs-0.268 gap "gamed" exactly as loudly as an 81-vs-55 one, and the figure
        # headline is built from these verdicts.
        from scipy.stats import mannwhitneyu
        better_is_lower = (key == "pppl")
        try:
            pv = float(mannwhitneyu(d, c).pvalue)
        except ValueError:
            pv = float("nan")
        deg_better = (np.mean(d) < np.mean(c)) if better_is_lower else (np.mean(d) > np.mean(c))
        if pv == pv and pv < 0.05:
            verdict = "gamed" if deg_better else "discriminates"
        else:
            verdict = "indistinguishable"
        gate[key] = {"degenerate_mean": float(np.mean(d)), "clean_mean": float(np.mean(c)),
                     "n_degenerate": len(d), "n_clean": len(c), "mannwhitney_p": pv,
                     "verdict": verdict}
        blurb = {"gamed": "GAMED — scores known non-proteins better than our own output",
                 "discriminates": "discriminates — rejects non-proteins",
                 "indistinguishable": "cannot tell non-proteins from our output"}[verdict]
        print(f"{key:>10} {np.mean(d):>14.3f} {np.mean(c):>14.3f} "
              f"{'' if ref is None else f'{ref:>12.2f}'}  {blurb} (p={pv:.1e})")
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

    # The attainment-front and objective-space-path figures that used to be built here are
    # gone. alpha is a knob we control, so the achievable (naturalness, effect) set simply IS
    # the dose-response curve and a "front" is that curve with dominated alphas deleted —
    # optimisation vocabulary over a filtered line plot. And the trade-off they were drawn to
    # show does not exist here: structural confidence is flat across alpha and arm, so the
    # second axis carried noise and the figures were unreadable. The claims are made instead
    # by scripts/plot_steering_report.py (divergence, baseline drift, stacked dose-response),
    # where the flat naturalness panel IS the dominance statement.
    style.apply_style()
    import matplotlib.pyplot as plt

    arms = ((False, style.ACCENT, "naive"), (True, style.method_color("glp"), "GLP-corrected"))

    # ---- pass rate: the "filter at pLDDT>=70" practice as an unbiased outcome ----
    thr_map = {"ptm": [0.3, 0.4, 0.5], "plddt": [60.0, 70.0, 80.0], "pppl": [10.0, 8.0]}
    thresholds = thr_map[key]
    rates = pass_rate_by_alpha(rows, key, thresholds, higher_is_better)
    run.log_summary(**{f"pass_rate_{key}": {"thresholds": thresholds, "by_alpha": rates}})
    print(f"\n[naturalness] pass rate on {key} (non-degenerate AND past threshold; "
          "denominator = every sequence generated)")
    hdr = "  alpha | " + " | ".join(f"{'≥' if higher_is_better else '≤'}{t:g} nv/cor".rjust(16)
                                    for t in thresholds)
    print(hdr)
    for a, e in rates.items():
        cells = []
        for t in thresholds:
            nv, cr = e.get(f"naive_pass_{t:g}"), e.get(f"corrected_pass_{t:g}")
            cells.append(f"{'--' if nv is None else f'{nv:.0%}'}"
                         f" / {'--' if cr is None else f'{cr:.0%}'}".rjust(16))
        print(f"  {a:>5} | " + " | ".join(cells))

    fig3, ax3 = plt.subplots(figsize=(7.6, 5.0))
    mid = thresholds[len(thresholds) // 2]
    order = sorted(rates, key=float)
    # evenly-spaced categorical ticks: the alpha grid contains 0, which a log axis cannot show
    # and symlog only displays by putting jargon on the axis
    xi = {a: i for i, a in enumerate(order)}
    for corrected, colour, lab in arms:
        nm = "corrected" if corrected else "naive"
        xs, ys, es = [], [], []
        for a in order:
            e = rates[a]
            v, n = e.get(f"{nm}_pass_{mid:g}"), e.get(f"{nm}_n_generated")
            if v is None or not n:
                continue
            xs.append(xi[a]); ys.append(v * 100)
            # binomial standard error — without it the wiggles invite over-reading, and here
            # every difference between the arms sits inside about one SE
            es.append(100 * (v * (1 - v) / n) ** 0.5)
        if xs:
            ax3.errorbar(xs, ys, yerr=es, color=colour, marker="o", linewidth=2, ms=6,
                         capsize=2.5, elinewidth=1, label=lab)
    ax3.set_xticks(range(len(order)))
    ax3.set_xticklabels([f"{float(a):g}" for a in order])
    ax3.set_ylim(bottom=0)
    ax3.set_xlabel("steering strength α")
    ax3.set_ylabel(f"% of generated sequences that pass\n(not degenerate, and "
                   f"{'pTM' if key == 'ptm' else key} "
                   f"{'≥' if higher_is_better else '≤'} {mid:g})")
    ax3.set_title("How often steering yields a confidently-folded sequence\n"
                  "every generation counted, so degenerate output counts as a failure",
                  loc="left")
    ax3.legend(fontsize=9, loc="upper right")
    ax3.annotate("bars are binomial SE; both directions pooled",
                 xy=(0.99, 0.02), xycoords="axes fraction", ha="right", fontsize=8,
                 color=style.SECONDARY)
    p3 = style.save_fig(fig3, run.figures_dir / f"pass_rate_{key}.png")
    print(f"[naturalness] wrote {p3}")

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
    # Built from the tested verdicts, never hardcoded: the same figure is produced for runs
    # where an axis behaves differently, and a fixed headline would then contradict its own
    # panels (it did — pTM separated junk from clean output on one run and not on another).
    pretty = {"pppl": "pseudo-perplexity", "plddt": "pLDDT", "ptm": "pTM"}
    bad = [pretty[k] for k, v in gate.items() if v["verdict"] == "gamed"]
    meh = [pretty[k] for k, v in gate.items() if v["verdict"] == "indistinguishable"]
    ok = [pretty[k] for k, v in gate.items() if v["verdict"] == "discriminates"]
    bits = []
    if bad:
        bits.append(f"scored BETTER than our own output by: {', '.join(bad)}")
    if meh:
        bits.append(f"indistinguishable from it by: {', '.join(meh)}")
    if ok:
        bits.append(f"correctly rejected by: {', '.join(ok)}")
    fig2.suptitle("Sequences that are not proteins, " + "; ".join(bits),
                  x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    path2 = style.save_fig(fig2, run.figures_dir / "naturalness_axis_gate_check.png")
    print(f"[naturalness] wrote {path2}")


if __name__ == "__main__":
    main()
