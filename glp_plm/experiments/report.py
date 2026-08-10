"""Render all applicable figures for a run (and cross-run comparisons).

`make_report(run_dir)` is idempotent and additive: it renders every registered figure
whose declared inputs exist, so it can be called after each pipeline phase (train, eval,
probing, …) and each call fills in whatever new figures the latest data enables.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from glp_plm.experiments import figures as _figs
from glp_plm.experiments import style
from glp_plm.experiments.metrics import RunLogger
from glp_plm.experiments.summary import write_run_summary


def make_report(run_dir: str | Path, verbose: bool = True) -> List[str]:
    style.apply_style()
    run = RunLogger(run_dir)
    made: List[str] = []
    for fig in _figs.REGISTRY.values():
        if not all((run.dir / need).exists() for need in fig.needs):
            continue
        try:
            out = fig.fn(run)
        except Exception as e:  # a broken figure must not sink the whole report
            if verbose:
                print(f"[report] {fig.name}: FAILED ({type(e).__name__}: {e})")
            continue
        if out:
            made.append(out)
            if verbose:
                print(f"[report] {fig.name} -> {out}")
    # write an index for quick browsing
    if made:
        idx = run.figures_dir / "INDEX.md"
        idx.write_text("# Figures\n\n" + "\n".join(f"- ![{Path(m).stem}]({Path(m).name})" for m in made) + "\n")
    if verbose:
        print(f"[report] {len(made)} figure(s) in {run.figures_dir}")
    write_run_summary(run_dir, verbose=verbose)
    return made


def compare_runs(run_dirs: List[str | Path], summary_key: str, x_key: str,
                 out_path: str | Path, xlabel: str = "", ylabel: str = "") -> Optional[str]:
    """Cross-run line: `summary_key` vs `x_key` across runs (W2 FD-vs-budget, scaling curve).

    Reads each run's metrics/summary.json; skips runs missing either key.
    """
    import matplotlib.pyplot as plt

    style.apply_style()
    pts = []
    for rd in run_dirs:
        s = RunLogger(rd).load_summary()
        if summary_key in s and x_key in s:
            pts.append((s[x_key], s[summary_key]))
    if not pts:
        return None
    pts.sort()
    xs, ys = zip(*pts)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o", markersize=7, color=style.method_color("glp"))
    ax.set_xlabel(xlabel or x_key)
    ax.set_ylabel(ylabel or summary_key)
    ax.set_title(f"{ylabel or summary_key} vs {xlabel or x_key}")
    return str(style.save_fig(fig, out_path))
