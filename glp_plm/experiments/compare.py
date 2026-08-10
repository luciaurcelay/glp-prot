"""Cross-run comparison figures — deliberately NOT part of the per-run pipeline.

`make_report` (experiments.report) renders figures for ONE finished run from its own
metrics/. This module compares MULTIPLE already-finished runs against each other —
e.g. "does probing AUC / generation FD improve with more training activations" — which
by definition needs runs that already exist. So it's invoked by hand
(`scripts/compare_runs.py`) after the fact, never from `submit_pipeline.sh`, and it
never writes into any compared run's own `metrics/` or `figures/`.

Read-only by design: `_load_run` reads each run's `metrics/summary.json`,
`metrics/train.jsonl`, `metrics/probing.json`, and (for the x-axis) the activation
store's `train_store/meta.json` — but never constructs a `RunLogger`, whose
constructor stamps a fresh git_commit/git_dirty into `summary.json` as a side effect
(appropriate for a run producing its own metrics, wrong for a tool that's only
comparing runs someone else's job may still be writing to).

Each `compare_*` figure follows the same defensive contract as `experiments.figures`:
returns None (never raises) when fewer than 2 of the given runs have the metric it
needs — runs reach pipeline stages at different times (e.g. as of 2026-08,
uniref50_500m has training metrics only; eval_generation/probing haven't run on it
yet), so a comparison only makes sense once >=2 runs have actually gotten there.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from glp_plm.experiments import style
from glp_plm.experiments.figures import _downsample
from glp_plm.utils import resolve_storage_dir

# Run identity (which RUN a line is, not which METHOD) is a different semantic axis
# than glp/sae/raw_layer/raw_mlp, so it must not reuse those hexes even though
# nothing else currently calls style.color() — a reader who has just seen glp=dark
# orange in probing_auc.png would misread the same hex here as "this is GLP's line".
# The two mid shades are exactly the "spare accent, outside the method identity set"
# role style.py already reserves them for (see its module docstring); validated PASS
# adjacent (ΔE 19.7, dataviz skill's validate_palette.js). A 3rd+ run needs a
# re-validated extension, not an unchecked append.
RUN_COLORS = [style.ORANGE_MID, style.PURPLE_MID]


def _human_count(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:g}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:g}M"
    if n >= 1_000:
        return f"{n / 1_000:g}K"
    return str(n)


def _run_color(i: int) -> str:
    return RUN_COLORS[i % len(RUN_COLORS)]


@dataclass
class RunInfo:
    dir: Path
    label: str
    summary: Dict[str, Any] = field(default_factory=dict)
    train: List[Dict[str, Any]] = field(default_factory=list)
    probing: Optional[Dict[str, Any]] = None
    n_activations: Optional[int] = None


def _load_run(run_dir: str | Path, label: Optional[str] = None) -> RunInfo:
    d = Path(run_dir)
    metrics_dir = d / "metrics"  # already resolved by whichever pipeline stage produced it

    summary: Dict[str, Any] = {}
    p = metrics_dir / "summary.json"
    if p.exists():
        summary = json.loads(p.read_text())

    train: List[Dict[str, Any]] = []
    p = metrics_dir / "train.jsonl"
    if p.exists():
        train = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    probing = None
    p = metrics_dir / "probing.json"
    if p.exists():
        probing = json.loads(p.read_text())

    # train_store is data-sized and NOT symlinked under runs/ (scripts/cluster/README.md
    # "Storage layout") — resolve_storage_dir(symlink=False) finds it either way, on
    # cluster or the retired local Mac PoC layout, without creating a symlink itself.
    n_activations = None
    store_meta = resolve_storage_dir(d, "train_store", "GLP_DATA_ROOT", symlink=False) / "meta.json"
    if store_meta.exists():
        n_activations = json.loads(store_meta.read_text()).get("n")

    return RunInfo(
        dir=d,
        label=label or (_human_count(n_activations) if n_activations else d.name),
        summary=summary, train=train, probing=probing, n_activations=n_activations,
    )


def compare_training_loss(runs: List[RunInfo], out_dir: Path) -> Optional[str]:
    """Loss-curve overlay, one line per run. The only comparison that only needs
    train.jsonl — usually the first (and, early on, only) one with enough data."""
    import matplotlib.pyplot as plt

    qualifying = [r for r in runs if r.train]
    if len(qualifying) < 2:
        return None

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, r in enumerate(qualifying):
        steps = [row["step"] for row in r.train]
        loss = [row["loss"] for row in r.train]
        xs, ys = _downsample(steps, loss)
        ax.plot(xs, ys, color=_run_color(i), label=r.label, lw=1.6, alpha=0.9)
    ax.set_xlabel("training step")
    ax.set_ylabel("flow-matching MSE")
    ax.set_title("Training loss across runs")
    ax.legend()
    return str(style.save_fig(fig, out_dir / "scaling_loss.png"))


def compare_scaling_auc(runs: List[RunInfo], out_dir: Path) -> Optional[str]:
    """Mean test AUC per method vs training scale (n activations) — the headline
    "does GLP's edge hold/grow with more data" curve. One line per method, method
    identity colors (same as probing_auc.png) since method IS the entity here."""
    import matplotlib.pyplot as plt

    qualifying = sorted(
        (r for r in runs if r.probing and r.probing.get("aggregate") and r.n_activations),
        key=lambda r: r.n_activations,
    )
    if len(qualifying) < 2:
        return None

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in style.METHOD_ORDER:
        pts = [(r.n_activations, r.probing["aggregate"][m]["mean_test_auc"])
               for r in qualifying if m in r.probing["aggregate"]]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", markersize=8, color=style.method_color(m), label=m)
    ax.set_xscale("log")
    ax.set_xlabel("training activations")
    ax.set_ylabel("mean test AUC")
    ax.set_title("1-D probing AUC vs training scale")
    ax.legend()
    return str(style.save_fig(fig, out_dir / "scaling_auc.png"))


def compare_scaling_fd(runs: List[RunInfo], out_dir: Path) -> Optional[str]:
    """Generation-quality FD vs training scale, with the real-vs-real lower bound
    anchor alongside it (same pairing as fd_vs_steps.png / generation_fd.png)."""
    import matplotlib.pyplot as plt

    qualifying = sorted(
        (r for r in runs if "fd_generated_vs_real" in r.summary and r.n_activations),
        key=lambda r: r.n_activations,
    )
    if len(qualifying) < 2:
        return None

    xs = [r.n_activations for r in qualifying]
    fd = [r.summary["fd_generated_vs_real"] for r in qualifying]
    lb = [r.summary.get("fd_lower_bound") for r in qualifying]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, fd, marker="o", markersize=8, color=style.ACCENT, label="FD(generated, real)")
    if all(v is not None for v in lb):
        ax.plot(xs, lb, marker="o", markersize=6, color=style.BASELINE, ls="--",
                label="FD(train, val) lower bound")
    ax.set_xscale("log")
    ax.set_xlabel("training activations")
    ax.set_ylabel("Fréchet distance")
    ax.set_title("Generation quality vs training scale")
    ax.legend()
    return str(style.save_fig(fig, out_dir / "scaling_fd.png"))


# Registry order = render order = summary-table column emphasis. Add a new
# compare_* here (same needs-based skip contract) to extend the report.
COMPARISONS = [compare_training_loss, compare_scaling_auc, compare_scaling_fd]


def _summary_table(runs: List[RunInfo]) -> str:
    lines = ["| run | activations | final loss | FD(gen,real) | glp AUC | sae AUC |",
             "|---|---|---|---|---|---|"]
    for r in runs:
        acts = _human_count(r.n_activations) if r.n_activations else "—"
        final_loss = f"{r.summary['final_loss']:.4f}" if "final_loss" in r.summary else "—"
        fd = f"{r.summary['fd_generated_vs_real']:.3f}" if "fd_generated_vs_real" in r.summary else "—"
        agg = (r.probing or {}).get("aggregate", {})
        glp_auc = f"{agg['glp']['mean_test_auc']:.3f}" if "glp" in agg else "—"
        sae_auc = f"{agg['sae']['mean_test_auc']:.3f}" if "sae" in agg else "—"
        lines.append(f"| {r.label} | {acts} | {final_loss} | {fd} | {glp_auc} | {sae_auc} |")
    return "\n".join(lines)


def compare_runs(run_dirs: List[str | Path], labels: Optional[List[str]] = None,
                 name: Optional[str] = None, out_root: str | Path = "runs/comparisons",
                 verbose: bool = True) -> Dict[str, Any]:
    """Render every registered cross-run comparison with enough qualifying runs.

    Writes to `<out_root>/<name>/{figures/*.png, comparison_summary.md}` — a sibling
    of individual runs' `runs/<run>/`, never inside one of them, so a comparison can
    never be mistaken for one run's own output (or clobber one).
    """
    style.apply_style()
    labels = labels or [None] * len(run_dirs)
    runs = [_load_run(d, l) for d, l in zip(run_dirs, labels)]

    name = name or "_vs_".join(r.label for r in runs)
    out_dir = Path(out_root) / name
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    made: List[str] = []
    for fn in COMPARISONS:
        try:
            out = fn(runs, figures_dir)
        except Exception as e:  # a broken comparison must not sink the rest
            if verbose:
                print(f"[compare] {fn.__name__}: FAILED ({type(e).__name__}: {e})")
            continue
        if out:
            made.append(out)
            if verbose:
                print(f"[compare] {fn.__name__} -> {out}")
        elif verbose:
            print(f"[compare] {fn.__name__}: skipped (< 2 runs have this metric yet)")

    md = [f"# Comparison: {' vs '.join(r.label for r in runs)}", "", _summary_table(runs)]
    if made:
        md += ["", "## Figures", ""]
        md += [f"![{Path(p).stem}](figures/{Path(p).name})" for p in made]
    summary_path = out_dir / "comparison_summary.md"
    summary_path.write_text("\n".join(md) + "\n")

    if verbose:
        print(f"[compare] {len(made)} figure(s) -> {figures_dir}")
        print(f"[compare] summary -> {summary_path}")
    return {"out_dir": str(out_dir), "figures": made, "summary": str(summary_path)}
