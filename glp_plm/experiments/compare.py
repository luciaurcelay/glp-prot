"""Cross-run scaling comparison — the paper's depth x data-scale figures.

Compares MULTIPLE already-finished runs against each other (a depth grid at two data
scales), on generation quality only: FLOPs-vs-loss, FD-vs-sampler-steps, PCA-vs-steps,
and a params/FD table. Deliberately NOT part of the per-run pipeline and NEVER touches
probing — the block scan follows GLP's own protocol of comparing architectures by
Fréchet Distance.

Read-only by design: `_load_run` reads each run's `metrics/*.json(l)`, `config.yaml`
(for n_train_activations, since store_dir-reuse runs have no per-run train_store), and
an optional `pca_proj_vs_steps.npz` — but never constructs a `RunLogger` (whose
constructor would stamp git provenance into a run someone else may still be writing).

FLOPs follow Kaplan et al. (2020): C = 6 N D, N = params, D = tokens = activations seen
(n_train_activations x epochs). Loss is smoothed (mean of the last ~500 steps) because
the flow-matching loss is single-step noisy (random t~U(0,1), eps per step) at a
magnitude comparable to the inter-run gaps.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from glp_plm.experiments import style

# GLP paper's LLM-activation compute-scaling fit, for reference overlay (their units).
GLP_REF = {"E": 0.52, "A": 435.1, "alpha": 0.169}

LOSS_SMOOTH_WINDOW = 500   # steps averaged for the "final" loss (kills single-step noise)


def _human_count(n: Optional[int]) -> str:
    if not n:
        return "—"
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:g}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:g}M"
    if n >= 1_000:
        return f"{n / 1_000:g}K"
    return str(n)


@dataclass
class RunInfo:
    dir: Path
    label: str
    n_blocks: Optional[int] = None
    n_params: Optional[int] = None
    n_activations: Optional[int] = None       # tokens D (from config.yaml)
    smoothed_loss: Optional[float] = None
    summary: Dict[str, Any] = field(default_factory=dict)
    fd_sweep: List[Dict[str, Any]] = field(default_factory=list)   # [{steps,fd,ratio,pc_std_ratio}]
    fd_lower_bound: Optional[float] = None
    pca_steps: Optional[Dict[str, Any]] = None                     # {real, gen_<step>...} if npz present

    @property
    def flops(self) -> Optional[float]:
        if self.n_params and self.n_activations:
            return 6.0 * self.n_params * self.n_activations
        return None

    @property
    def scale(self) -> str:
        return _human_count(self.n_activations)


def _load_run(run_dir: str | Path, label: Optional[str] = None) -> RunInfo:
    d = Path(run_dir)
    m = d / "metrics"

    summary: Dict[str, Any] = {}
    if (m / "summary.json").exists():
        summary = json.loads((m / "summary.json").read_text())

    # tokens D: n_train_activations from the run's own config.yaml (log_config). The
    # per-run train_store/meta.json is absent when the run reused a shared store_dir,
    # so config.yaml is the reliable source; fall back to the store meta (works on
    # cluster) for older runs that predate log_config.
    n_activations = None
    cfg_path = d / "config.yaml"
    if cfg_path.exists():
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text())
        n_activations = (cfg.get("data") or {}).get("n_train_activations")
    if n_activations is None:
        try:
            from glp_plm.utils import resolve_storage_dir
            meta = resolve_storage_dir(d, "train_store", "GLP_DATA_ROOT", symlink=False) / "meta.json"
            if meta.exists():
                n_activations = json.loads(meta.read_text()).get("n")
        except Exception:
            pass

    # smoothed final loss
    smoothed_loss = None
    if (m / "train.jsonl").exists():
        losses = [json.loads(l)["loss"] for l in (m / "train.jsonl").read_text().splitlines() if l.strip()]
        if losses:
            tail = losses[-LOSS_SMOOTH_WINDOW:]
            smoothed_loss = sum(tail) / len(tail)

    fd_sweep: List[Dict[str, Any]] = []
    fd_lb = None
    if (m / "fd_vs_steps.json").exists():
        d_fd = json.loads((m / "fd_vs_steps.json").read_text())
        fd_sweep = sorted(d_fd.get("sweep", []), key=lambda r: r["steps"])
        fd_lb = d_fd.get("lower_bound")

    pca_steps = None
    if (m / "pca_proj_vs_steps.npz").exists():
        import numpy as np
        z = np.load(m / "pca_proj_vs_steps.npz")
        pca_steps = {k: z[k] for k in z.files}

    return RunInfo(
        dir=d, label=label or d.name,
        n_blocks=summary.get("n_blocks"), n_params=summary.get("n_params"),
        n_activations=n_activations, smoothed_loss=smoothed_loss,
        summary=summary, fd_sweep=fd_sweep, fd_lower_bound=fd_lb, pca_steps=pca_steps,
    )


# --- shared visual encoding -------------------------------------------------
def _depth_colors(runs: List[RunInfo]) -> Dict[int, str]:
    """Fixed depth -> color, sorted shallow->deep, from the documented palette."""
    depths = sorted({r.n_blocks for r in runs if r.n_blocks is not None})
    return {nb: style.color(i) for i, nb in enumerate(depths)}


def _scale_style(runs: List[RunInfo]) -> Dict[int, str]:
    """Data scale -> linestyle: solid for the larger scale, dashed for the smaller."""
    scales = sorted({r.n_activations for r in runs if r.n_activations})
    # smallest dashed, then denser -> solid (only 2 in practice)
    order = ["--", "-", "-."]
    return {n: order[min(i, len(order) - 1)] for i, n in enumerate(scales)}


def _scale_colors(runs: List[RunInfo]) -> Dict[int, str]:
    """Data scale -> color (for plots keyed on scale rather than depth)."""
    scales = sorted({r.n_activations for r in runs if r.n_activations})
    palette = [style.ORANGE_MID, style.PURPLE_MID, style.PURPLE_DARK]
    return {n: palette[min(i, len(palette) - 1)] for i, n in enumerate(scales)}


# --- figures ----------------------------------------------------------------
def flops_vs_loss(runs: List[RunInfo], out_dir: Path) -> Optional[str]:
    """Compute-scaling: smoothed diffusion loss vs FLOPs (C=6ND), points colored by
    data scale, with a power-law fit L(C)=E+A*C^-a and the GLP reference overlaid."""
    import matplotlib.pyplot as plt
    import numpy as np

    pts = [r for r in runs if r.flops and r.smoothed_loss is not None]
    if len(pts) < 2:
        return None
    C = np.array([r.flops for r in pts], float)
    L = np.array([r.smoothed_loss for r in pts], float)
    scol = _scale_colors(pts)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    # points, one color per data scale; connect within a scale (ordered by depth)
    for n in sorted({r.n_activations for r in pts}):
        grp = sorted([r for r in pts if r.n_activations == n], key=lambda r: r.flops)
        ax.plot([r.flops for r in grp], [r.smoothed_loss for r in grp], marker="o",
                markersize=8, color=scol[n], lw=1.4, label=f"{_human_count(n)} activations", zorder=3)
        for r in grp:  # annotate depth
            ax.annotate(f"{r.n_blocks}b", (r.flops, r.smoothed_loss),
                        textcoords="offset points", xytext=(6, 5), fontsize=8, color=style.SECONDARY)

    # power-law fit L = E + A*(C/C0)^-a  (C0 = geo-mean for numerical stability), then
    # report A back in raw-C units so alpha/E/A are comparable to the GLP reference.
    fit_txt = ""
    try:
        from scipy.optimize import curve_fit
        C0 = float(np.exp(np.mean(np.log(C))))
        f = lambda c, E, A, a: E + A * (c / C0) ** (-a)
        p0 = [float(L.min()) * 0.9, float(L.max() - L.min()), 0.17]
        (E, A, a), _ = curve_fit(f, C, L, p0=p0, maxfev=20000,
                                 bounds=([0, 0, 0.0], [L.min(), np.inf, 2.0]))
        A_raw = A * C0 ** a
        xs = np.logspace(np.log10(C.min()) - 0.15, np.log10(C.max()) + 0.15, 200)
        ax.plot(xs, E + A_raw * xs ** (-a), color=style.INK, lw=1.6, ls="-",
                label=f"power-law fit (α={a:.3f})", zorder=2)
        e_note = "" if E > 0.01 else "  [E unconstrained over this C range → ~0; effective L≈A·C^−α]"
        fit_txt = (f"our fit: L = {E:.3f} + {A_raw:.3g}·C^(−{a:.3f})  vs GLP α={GLP_REF['alpha']}." + e_note)
    except Exception as e:  # thin data / no scipy: show points without a fit
        fit_txt = f"(no power-law fit: {type(e).__name__})"

    # GLP reference curve (their LLM-activation fit), over our C range
    xs = np.logspace(np.log10(C.min()) - 0.15, np.log10(C.max()) + 0.15, 200)
    ax.plot(xs, GLP_REF["E"] + GLP_REF["A"] * xs ** (-GLP_REF["alpha"]),
            color=style.BASELINE, lw=1.4, ls=":", label=f"GLP ref (α={GLP_REF['alpha']})", zorder=1)

    ax.set_xscale("log")
    ax.set_xlabel("compute  C = 6ND  (FLOPs)")
    ax.set_ylabel("diffusion loss (last-500-step mean)")
    ax.set_title("Compute scaling: diffusion loss vs FLOPs")
    ax.legend(fontsize=8)
    print(f"[compare] {fit_txt}")
    return str(style.save_fig(fig, out_dir / "flops_vs_loss.png"))


def fd_vs_steps(runs: List[RunInfo], out_dir: Path) -> Optional[str]:
    """FD vs sampler steps, faceted by data scale (one subplot per scale, shared y),
    one line per depth + the real-vs-real lower bound. FD must be read with its step
    count, so this doubles as the 'always quote steps' caveat figure."""
    import matplotlib.pyplot as plt

    pts = [r for r in runs if r.fd_sweep and r.n_activations]
    if len(pts) < 2:
        return None
    scales = sorted({r.n_activations for r in pts})
    dcol = _depth_colors(pts)

    fig, axes = plt.subplots(1, len(scales), figsize=(5.2 * len(scales), 4.3), sharey=True, squeeze=False)
    for ax, n in zip(axes[0], scales):
        grp = sorted([r for r in pts if r.n_activations == n], key=lambda r: r.n_blocks or 0)
        for r in grp:
            xs = [s["steps"] for s in r.fd_sweep]
            ys = [s["fd"] for s in r.fd_sweep]
            ax.plot(xs, ys, marker="o", markersize=6, color=dcol[r.n_blocks], label=f"{r.n_blocks} blocks")
        lb = next((r.fd_lower_bound for r in grp if r.fd_lower_bound is not None), None)
        if lb is not None:
            ax.axhline(lb, color=style.BASELINE, lw=1.3, ls="--", label=f"lower bound {lb:.2f}")
        ax.set_xscale("log")
        ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
        ax.set_xlabel("sampler steps")
        ax.set_title(f"{_human_count(n)} activations")
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("Fréchet distance")
    fig.suptitle("FD vs sampler steps, by depth and data scale", y=1.0)
    return str(style.save_fig(fig, out_dir / "fd_vs_steps.png"))


def pca_vs_steps(runs: List[RunInfo], out_dir: Path) -> Optional[str]:
    """4-panel PCA overlay (real vs generated on real's top-2 PCs) at increasing
    sampler steps for the single run that has pca_proj_vs_steps.npz — shows the
    diffusion sampler refining from near-noise to matched. Needs the regen step
    (scripts/pca_vs_steps.py)."""
    import matplotlib.pyplot as plt

    r = next((r for r in runs if r.pca_steps), None)
    if r is None:
        return None
    z = r.pca_steps
    real = z["real"]
    step_keys = sorted((k for k in z if k.startswith("gen_")), key=lambda k: int(k.split("_")[1]))
    if not step_keys:
        return None

    fig, axes = plt.subplots(1, len(step_keys), figsize=(3.5 * len(step_keys), 3.7), squeeze=False)
    for ax, k in zip(axes[0], step_keys):
        gen = z[k]
        ax.scatter(real[:, 0], real[:, 1], s=5, alpha=0.35, color=style.REAL_COLOR,
                   edgecolors="none", label="real")
        ax.scatter(gen[:, 0], gen[:, 1], s=5, alpha=0.35, color=style.GENERATED_COLOR,
                   edgecolors="none", label="generated")
        ax.set_title(f"{int(k.split('_')[1])} steps")
        ax.set_xlabel("PC1")
    axes[0][0].set_ylabel("PC2")
    axes[0][0].legend(markerscale=3, fontsize=8)
    fig.suptitle(f"Sampler convergence in PCA space — {r.label}", y=1.0)
    return str(style.save_fig(fig, out_dir / "pca_vs_steps.png"))


def scaling_table(runs: List[RunInfo], out_dir: Path) -> str:
    """Params + FD table for every run (markdown returned, CSV written alongside)."""
    ordered = sorted(runs, key=lambda r: (r.n_activations or 0, r.n_blocks or 0))
    header = ["data", "blocks", "params", "FLOPs (C=6ND)", "FD@1000", "FD ratio", "loss (smoothed)"]
    rows = []
    for r in ordered:
        rows.append([
            r.scale,
            str(r.n_blocks) if r.n_blocks is not None else "—",
            _human_count(r.n_params),
            f"{r.flops:.3g}" if r.flops else "—",
            f"{r.summary['fd_generated_vs_real']:.3f}" if "fd_generated_vs_real" in r.summary else "—",
            f"{r.summary['fd_ratio']:.2f}" if "fd_ratio" in r.summary else "—",
            f"{r.smoothed_loss:.4f}" if r.smoothed_loss is not None else "—",
        ])
    with open(out_dir / "scaling_table.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(md)


FIGURES = [flops_vs_loss, fd_vs_steps, pca_vs_steps]


def compare_runs(run_dirs: List[str | Path], labels: Optional[List[str]] = None,
                 name: str = "depth_scaling", out_root: str | Path = "runs/comparisons",
                 verbose: bool = True) -> Dict[str, Any]:
    """Render the depth x data-scale generation-quality report from finished runs.

    Writes `<out_root>/<name>/{figures/*.png, scaling_table.csv, comparison_summary.md}`
    — a sibling of the individual runs, never inside one of them.
    """
    style.apply_style()
    name = name or "depth_scaling"
    labels = labels or [None] * len(run_dirs)
    runs = [_load_run(d, l) for d, l in zip(run_dirs, labels)]

    out_dir = Path(out_root) / name
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    made: List[str] = []
    for fn in FIGURES:
        try:
            out = fn(runs, figures_dir)
        except Exception as e:  # a broken figure must not sink the rest
            if verbose:
                print(f"[compare] {fn.__name__}: FAILED ({type(e).__name__}: {e})")
            continue
        if out:
            made.append(out)
            if verbose:
                print(f"[compare] {fn.__name__} -> {out}")
        elif verbose:
            print(f"[compare] {fn.__name__}: skipped (inputs missing)")

    table = scaling_table(runs, figures_dir.parent)
    md = ["# Depth × data-scale generation-quality comparison", "",
          "Generation only (FD / PCA / loss) — the block scan follows GLP's protocol of "
          "comparing architectures by Fréchet Distance. No probing.", "", table]
    if made:
        md += ["", "## Figures", ""]
        md += [f"![{Path(p).stem}](figures/{Path(p).name})" for p in made]
    summary_path = out_dir / "comparison_summary.md"
    summary_path.write_text("\n".join(md) + "\n")

    if verbose:
        print(f"[compare] {len(made)} figure(s) + table -> {out_dir}")
    return {"out_dir": str(out_dir), "figures": made, "summary": str(summary_path)}
