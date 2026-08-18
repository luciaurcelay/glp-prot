#!/usr/bin/env python
"""Cross-property specificity: does steering property X move X, or does it move everything?

Pools several Exp B runs — one per steered property — and builds the matrix

    rows    = the property the run steered
    columns = the property measured on the SAME generated sequences
    cell    = steering-attributable divergence (up minus down), in units of the natural
              held-out proteins' own SD for that column

A concept direction should light up its diagonal. A vector that just makes proteins weirder
lights up whole rows. Neither can be told apart from a single-property experiment, which is
why every generation is scored on every property (glp_plm/eval/steering_properties.py).

**Off-diagonal cells are not automatically failures.** The properties are correlated in real
proteins — measured on our own UniRef50 pool, GRAVY x instability r = -0.35 — and a faithful
direction should reproduce that coupling. (GRAVY x charge is the pair that does NOT need the
correction: r = -0.03. Net charge per residue is near-orthogonal to hydrophobicity, because a
protein rich in both Lys and Glu is hydrophilic with near-zero net charge.) The natural correlation matrix from the
contrast build is therefore printed alongside, and the figure's second panel is the residual:
measured off-target minus what the on-target effect would produce through the natural
correlation alone. That residual, not the raw off-target number, is the evidence of an
unspecific vector.

Every quantity here is (a) steering-attributable — differenced against the same init's own
alpha=0 run in the same arm and direction, so the correction's own drift is removed — and
(b) computed within a direction before differencing, since pooling would cancel the effect.

Usage:
  python scripts/analyze_property_specificity.py \
      --run-dirs <gravy-run>,<charge-run>,<instability-run> \
      --out-run-dir <where figures/metrics go> [--alpha 2]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import _bootstrap  # noqa: F401
import numpy as np

from glp_plm.eval.steering_properties import (PROPERTY_KEYS, find_contrast_json, find_run_csv,
                                              get_property, run_property, score_all)
from glp_plm.experiments import RunLogger, style


def is_degenerate(row: dict) -> bool:
    """Mirrors run_steering.is_degenerate_sequence — this must analyse exactly the row set
    the run's own summary.json reported on."""
    seq = row.get("sequence") or ""
    if seq and max(Counter(seq).values()) / len(seq) >= 0.30:
        return True
    return (int(float(row["unique_aa_count"])) <= 8
            or int(float(row["max_homopolymer_run"])) >= 12)


def resolve_contrast(run_dir: Path, prop_key: str, override: Optional[str]) -> dict:
    """The contrast this run used, wherever it lives.

    A multi-property build writes one shared contrast per property outside any single run
    dir and the run is pointed at it with --contrast, so globbing the run dir is not
    enough. Order: explicit --contrasts override, then the path the run recorded in its own
    summary.json, then the run dir itself (single-property runs that built it inline).
    """
    if override:
        return json.loads(Path(override).read_text())
    summary = run_dir / "metrics" / "summary.json"
    if summary.exists():
        recorded = json.loads(summary.read_text()).get("contrast_path")
        if recorded and Path(recorded).exists():
            return json.loads(Path(recorded).read_text())
        # Runs before contrast_path was recorded may still carry the two fields directly.
        s = json.loads(summary.read_text())
        if s.get("contrast_pool_property_stats"):
            return {"pool_property_stats": s["contrast_pool_property_stats"],
                    "natural_correlations": s.get("contrast_natural_correlations")}
    local = find_contrast_json(run_dir, prop_key)
    if local:
        return json.loads(local.read_text())
    print(f"[specificity] WARNING: no contrast found for {run_dir.name} — pass "
          f"--contrasts {prop_key}=<path>. Without it this run cannot be standardized.")
    return {}


def load_run(run_dir: Path, neutral_alpha: float, override: Optional[str] = None
             ) -> Tuple[str, List[dict], dict]:
    csv_path = find_run_csv(run_dir)
    raw = list(csv.DictReader(open(csv_path)))
    prop = run_property(raw, csv_path)
    missing = [k for k in PROPERTY_KEYS if f"delta_{k}" not in (raw[0] if raw else {})]
    if missing:
        raise SystemExit(
            f"{csv_path} has no delta_{{{','.join(missing)}}} column — it predates the "
            "all-property scoring, so its off-target effects were never recorded and cannot "
            "be recovered without regenerating. Re-run it with the current run_steering.py.")

    rows = []
    for r in raw:
        if is_degenerate(r):
            continue
        row = {"alpha": float(r["alpha"]), "corrected": r["corrected"] == "True",
               "direction": r.get("direction") or "up", "init": r["init_sequence_id"]}
        for k in PROPERTY_KEYS:
            v = r.get(f"delta_{k}")
            row[k] = float(v) if v not in (None, "") else None
        rows.append(row)

    # steering-attributable: subtract this init's own alpha=0 run, same arm and direction
    base = {(r["direction"], r["corrected"], r["init"]): r
            for r in rows if r["alpha"] == neutral_alpha}
    for r in rows:
        b = base.get((r["direction"], r["corrected"], r["init"]))
        for k in PROPERTY_KEYS:
            r[f"attr_{k}"] = (None if b is None or r[k] is None or b[k] is None
                              else r[k] - b[k])

    contrast = resolve_contrast(run_dir, prop.key, override)
    print(f"[specificity] {run_dir.name}: steered {prop.key}, {len(rows)} clean rows")
    return prop.key, rows, contrast


def natural_sd(contrast: dict) -> Dict[str, float]:
    """SD of each property across natural proteins — the scale that makes an effect of
    +0.5 GRAVY and one of +0.02 charge/residue commensurable.

    It must come from the **UniRef50 pool**, not from the run's held-out starts. The two
    held-out sets are drawn from opposite quartiles of the *steered* property, so their
    pooled spread is bimodal and inflated for that one property while the off-target
    properties keep a roughly natural spread. Standardizing by it would shrink precisely
    the diagonal of this matrix — i.e. systematically understate on-target specificity.
    """
    stats = contrast.get("pool_property_stats")
    if stats:
        return {k: stats[k]["sd"] for k in PROPERTY_KEYS if k in stats}
    # Contrast files built before pool stats were recorded: fall back to the held-out
    # starts, and say so, because the diagonal is then biased low.
    seqs = [h["sequence"] for h in
            (contrast.get("heldout") or []) + (contrast.get("heldout_high") or [])]
    if not seqs:
        return {}
    print("[specificity] WARNING: this contrast predates `pool_property_stats`; "
          "standardizing by the held-out starts instead, which understates the diagonal. "
          "Rebuild the contrast with --restart for the correct scale.")
    scored = [score_all(s) for s in seqs]
    return {k: float(np.std([s[k] for s in scored], ddof=1)) for k in PROPERTY_KEYS}


def divergence(rows: List[dict], key: str, alpha: float, corrected: bool
               ) -> Optional[Tuple[float, float, int]]:
    """up-minus-down of the attributable change in `key`, with an unpaired test.

    Unpaired because the two directions start from different held-out proteins."""
    from scipy.stats import mannwhitneyu

    def side(d):
        return [r[f"attr_{key}"] for r in rows
                if r["direction"] == d and r["corrected"] == corrected
                and r["alpha"] == alpha and r[f"attr_{key}"] is not None]

    u, d = side("up"), side("down")
    if len(u) < 3 or len(d) < 3:
        return None
    try:
        pv = float(mannwhitneyu(u, d).pvalue)
    except ValueError:
        pv = float("nan")
    return float(np.mean(u) - np.mean(d)), pv, min(len(u), len(d))


def _override_for(run_dir: Path, overrides: Dict[str, str]) -> Optional[str]:
    """Match a `<property>=<path>` override to this run by the property in its CSV name."""
    if not overrides:
        return None
    name = find_run_csv(run_dir).name
    for key, path in overrides.items():
        if name == f"expB_diffmean_{key}.csv":
            return path
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", required=True,
                    help="comma-separated Exp B run dirs, one per steered property")
    ap.add_argument("--out-run-dir", required=True, help="where the figure and metrics go")
    ap.add_argument("--contrasts", default="",
                    help="comma-separated <property>=<path to its _contrast.json>, for runs "
                         "whose contrast lives outside the run dir and predates the "
                         "recording of `contrast_path` in summary.json")
    ap.add_argument("--match-on", choices=["alpha", "effect"], default="effect",
                    help="what to hold constant across the three properties. 'alpha' reads "
                         "every property at --alpha, i.e. matched PERTURBATION: it answers "
                         "'at the same push, which concept moves most'. 'effect' (default) "
                         "picks each property's own alpha so the ON-TARGET divergence is "
                         "closest to --target-effect, i.e. matched EFFECT: it answers 'at "
                         "the same on-target gain, what does each direction cost off "
                         "target'. The second is the fair reading of an off-target number — "
                         "otherwise a property that was simply pushed harder looks less "
                         "specific. Both need --alpha-mode relative runs to be meaningful.")
    ap.add_argument("--target-effect", type=float, default=1.0,
                    help="[--match-on effect] on-target divergence, in natural SD units, to "
                         "match the three properties at")
    ap.add_argument("--match-tolerance", type=float, default=0.35,
                    help="[--match-on effect] how far, in SD, a property's best alpha may "
                         "sit from --target-effect before the run is flagged as unmatched. "
                         "A flagged row is NOT comparable to the others and says the alpha "
                         "grid needs another point, not that the direction is unspecific.")
    ap.add_argument("--alpha", type=float, default=2.0,
                    help="[--match-on alpha] steering strength the matrix is evaluated at. "
                         "2.0 is where the hydrophobicity run separates the arms most "
                         "cleanly while staying well below the degenerate regime")
    ap.add_argument("--neutral-alpha", type=float, default=0.0)
    ap.add_argument("--arm", choices=["corrected", "naive", "both"], default="corrected")
    ap.add_argument("--no-titles", action="store_true")
    args = ap.parse_args()

    overrides = dict(kv.split("=", 1) for kv in args.contrasts.split(",") if "=" in kv)
    runs = {}
    contrasts = {}
    for d in args.run_dirs.split(","):
        d = Path(d.strip())
        # the override is keyed by property, which we only learn inside load_run — so try
        # each override path against this run and let load_run pick by key
        key, rows, contrast = load_run(d, args.neutral_alpha,
                                       override=_override_for(d, overrides))
        if key in runs:
            raise SystemExit(f"two runs both steer {key!r} — pass one run per property")
        runs[key], contrasts[key] = rows, contrast

    steered_keys = [k for k in PROPERTY_KEYS if k in runs]
    arms = ([True] if args.arm == "corrected" else [False] if args.arm == "naive"
            else [False, True])

    # ---------------- the matrix ----------------
    results: Dict[str, dict] = {}
    alpha_used: Dict[str, Dict[str, float]] = {}
    for corrected in arms:
        arm_name = "corrected" if corrected else "naive"
        cells: Dict[str, Dict[str, dict]] = {}
        sds = {k: natural_sd(contrasts[k]) for k in steered_keys}
        chosen: Dict[str, float] = {}

        for sk in steered_keys:
            if args.match_on == "alpha":
                chosen[sk] = args.alpha
                continue
            # matched effect: this property's own alpha whose ON-TARGET standardized
            # divergence lands closest to the target. Comparing off-target numbers taken at
            # different on-target strengths would penalise whichever property happened to be
            # pushed hardest, which is a property of the grid, not of the direction.
            sd = sds[sk].get(sk)
            best, best_gap, best_eff = None, float("inf"), None
            for a in sorted({r["alpha"] for r in runs[sk]}):
                # The neutral alpha is the NO-STEERING control, not a weak steering setting.
                # Leaving it in the candidate set lets a property whose grid overshoots the
                # target "match" at zero steering, which reads as perfect specificity
                # (all-zero row) — it happened on the first run of this code.
                if a == args.neutral_alpha:
                    continue
                dv = divergence(runs[sk], sk, a, corrected)
                if dv is None or not sd:
                    continue
                eff = dv[0] / sd
                if abs(eff - args.target_effect) < best_gap:
                    best, best_gap, best_eff = a, abs(eff - args.target_effect), eff
            if best is None:
                raise SystemExit(f"no usable steered alpha for {sk} in the {arm_name} arm — "
                                 "run it with more sequences per cell, or use "
                                 "--match-on alpha")
            chosen[sk] = best
            note = (f"[specificity] {arm_name}/{sk}: alpha={best:g}, on-target "
                    f"{best_eff:+.2f} SD (target {args.target_effect:+.2f})")
            if best_gap > args.match_tolerance:
                note += ("  <-- NOT MATCHED: the alpha grid does not bracket the target, so "
                         "this row is read at a different on-target strength than the others "
                         "and its off-target numbers are not comparable to them")
            print(note)

        for sk in steered_keys:
            cells[sk] = {}
            for mk in PROPERTY_KEYS:
                dv = divergence(runs[sk], mk, chosen[sk], corrected)
                if dv is None:
                    continue
                raw_div, pv, n = dv
                sd = sds[sk].get(mk)
                cells[sk][mk] = {"divergence": raw_div, "p": pv, "n": n,
                                 "natural_sd": sd, "alpha": chosen[sk],
                                 "standardized": raw_div / sd if sd else None,
                                 "on_target": mk == sk}
        results[arm_name] = cells
        alpha_used[arm_name] = chosen

        where = (f"alpha={args.alpha:g}" if args.match_on == "alpha"
                 else f"matched on-target effect ≈ {args.target_effect:+.2f} SD, "
                      "per-property alpha")
        print(f"\n=== {arm_name} arm, {where}: steering-attributable divergence "
              "(up - down), in natural SD units ===")
        head = "  steered \\ measured  " + "".join(f"{k:>16}" for k in PROPERTY_KEYS)
        print(head)
        for sk in steered_keys:
            line = f"  {sk:>18}  "
            for mk in PROPERTY_KEYS:
                c = cells[sk].get(mk)
                line += ("            --  " if not c or c["standardized"] is None
                         else f"{c['standardized']:>+11.2f}{'*' if c['p'] < 0.01 else ' '}   ")
            print(line)
        print("  (* p < 0.01, unpaired Mann-Whitney; raw units in metrics/summary.json)")

    # ---------------- the natural-coupling reference ----------------
    # An off-target effect is only evidence of an unspecific vector once it exceeds what the
    # on-target effect would drag along through the properties' natural correlation.
    # Each run's own contrast, not the first one that happens to have it: the three are
    # built from the same UniRef50 pool so they should agree, and a disagreement means the
    # builds used different pools — which the residual must not paper over.
    corr_by_run = {sk: contrasts[sk].get("natural_correlations") for sk in steered_keys}
    corr = next((c for c in corr_by_run.values() if c), {})
    spread = [abs(a[x][y] - b[x][y]) for a in corr_by_run.values() if a
              for b in corr_by_run.values() if b
              for x in PROPERTY_KEYS for y in PROPERTY_KEYS]
    if spread and max(spread) > 0.05:
        print(f"[specificity] WARNING: the runs' contrast files disagree on the natural "
              f"correlations by up to {max(spread):.2f} — they were built from different "
              "UniRef50 pools (different seed or --n-raw-seqs). The residual panel assumes "
              "one common reference.")
    if corr:
        print("\n=== natural correlations between the properties (UniRef50 pool, "
              "the coupling a faithful direction SHOULD reproduce) ===")
        print("  " + " " * 16 + "".join(f"{k:>16}" for k in PROPERTY_KEYS))
        for a in PROPERTY_KEYS:
            print(f"  {a:>14}  " + "".join(f"{corr[a][b]:>+16.2f}" for b in PROPERTY_KEYS))
    else:
        print("\n[specificity] no `natural_correlations` in any contrast file (they predate "
              "it) — the residual panel is skipped; rebuild a contrast with --restart to "
              "recover it")

    residual: Dict[str, Dict[str, Optional[float]]] = {}
    arm_for_fig = "corrected" if "corrected" in results else list(results)[0]
    cells = results[arm_for_fig]
    for sk in steered_keys:
        residual[sk] = {}
        on = cells[sk].get(sk, {}).get("standardized")
        for mk in PROPERTY_KEYS:
            got = cells[sk].get(mk, {}).get("standardized")
            ref = corr_by_run.get(sk) or corr
            if mk == sk:
                # r = 1 on the diagonal, so the residual there is identically zero. Printing
                # "+0.00" in the on-target cell invites reading it as "no on-target effect",
                # which is the opposite of what the panel shows — leave it blank.
                residual[sk][mk] = None
            elif got is None or on is None or not ref:
                residual[sk][mk] = None
            else:
                # In SD units, a coupling of r predicts an off-target move of r x on-target.
                residual[sk][mk] = got - ref[sk][mk] * on

    run = RunLogger(Path(args.out_run_dir))
    run.log_summary(specificity_match_on=args.match_on, specificity_alpha=args.alpha,
                    specificity_target_effect=args.target_effect,
                    specificity_alpha_used=alpha_used, specificity=results,
                    specificity_residual=residual, natural_correlations=corr or None)

    # ---------------- figure ----------------
    style.apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    # Diverging: the quantity is signed (steering can push either way), so two hues with a
    # NEUTRAL midpoint — never a hue at zero, which would read as a third category.
    cmap = LinearSegmentedColormap.from_list(
        "steer_div", [style.ORANGE_DARK, "#f6f5f0", style.PURPLE_DARK])

    panels = [("measured effect", {sk: {mk: cells[sk].get(mk, {}).get("standardized")
                                        for mk in PROPERTY_KEYS} for sk in steered_keys})]
    if corr:
        panels.append(("minus natural coupling", residual))

    # the colorbar gets its own gridspec slot rather than stealing space from a panel —
    # save_fig() runs tight_layout, which cannot place a figure-level colorbar
    fig, axs = plt.subplots(1, len(panels) + 1, figsize=(5.6 * len(panels) + 1.3, 4.2),
                            squeeze=False,
                            gridspec_kw={"width_ratios": [1] * len(panels) + [0.045]})
    cax = axs[0][-1]
    vmax = max((abs(v) for _, m in panels for row in m.values() for v in row.values()
                if v is not None), default=1.0) or 1.0     # TwoSlopeNorm rejects vmax == 0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    for ax, (name, mat) in zip(axs[0][:-1], panels):
        ax.grid(False)          # the shared style draws a hairline grid; not on a heatmap
        grid = np.array([[np.nan if mat[sk][mk] is None else mat[sk][mk]
                          for mk in PROPERTY_KEYS] for sk in steered_keys], dtype=float)
        ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(len(PROPERTY_KEYS)))
        ax.set_xticklabels([get_property(k).unit for k in PROPERTY_KEYS], fontsize=9)
        ax.set_yticks(range(len(steered_keys)))
        ax.set_yticklabels([get_property(k).key for k in steered_keys], fontsize=9)
        ax.set_xlabel("property measured")
        if ax is axs[0][0]:
            ax.set_ylabel("property steered")
        else:
            ax.set_yticklabels([])
        for i, sk in enumerate(steered_keys):
            for j, mk in enumerate(PROPERTY_KEYS):
                v = mat[sk][mk]
                if v is None:
                    continue
                # text never takes the series colour: ink on light, surface on saturated
                ink = style.INK if abs(v) < 0.55 * vmax else style.SURFACE
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=10, color=ink)
        # the diagonal is the on-target claim; mark it rather than relying on the reader
        for i, sk in enumerate(steered_keys):
            if sk in PROPERTY_KEYS:
                j = PROPERTY_KEYS.index(sk)
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor=style.INK, linewidth=1.8))
                if mat[sk][sk] is None:      # residual panel: undefined on the diagonal
                    ax.text(j, i, "on-target", ha="center", va="center", fontsize=8,
                            style="italic", color=style.MUTED)
        if not args.no_titles:
            ax.set_title(name, loc="left")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                 label=("divergence (natural SD units)\n"
                        + (f"at α={args.alpha:g}" if args.match_on == "alpha"
                           else f"at matched on-target ≈{args.target_effect:+.2f} SD")))
    p = style.save_fig(fig, run.figures_dir / "property_specificity.png")
    print(f"\n[specificity] wrote {p}")


if __name__ == "__main__":
    main()
