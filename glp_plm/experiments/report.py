"""Render all applicable figures for a single run.

`make_report(run_dir)` is idempotent and additive: it renders every registered figure
whose declared inputs exist, so it can be called after each pipeline phase (train, eval,
probing, …) and each call fills in whatever new figures the latest data enables.

For comparing MULTIPLE already-finished runs against each other, see
`glp_plm.experiments.compare` instead — a separate, standalone module (not part of the
per-run pipeline this file serves).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

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
