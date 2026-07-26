"""Figure registry — each figure declares the metric artifacts it needs.

`make_report` (see experiments.report) renders every registered figure whose inputs
exist under the run dir, so figures appear as the pipeline produces their data:
  W1  -> training_loss, lr_schedule, generation_pca, generation_fd
  W2+ -> loss_recovered, fd_vs_budget, probing_auc, steering_pareto, scaling_curve …
Adding a figure = write one `fig_*` function and `@register(...)` it. Nothing else.

Each function takes a `RunLogger` and returns the saved path (or None if it no-ops).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from glp_plm.experiments import style
from glp_plm.experiments.metrics import RunLogger


@dataclass
class Figure:
    name: str
    fn: Callable[[RunLogger], Optional[str]]
    needs: List[str]          # metric artifact paths (relative to run dir) required


REGISTRY: Dict[str, Figure] = {}


def register(name: str, needs: List[str]):
    def deco(fn):
        REGISTRY[name] = Figure(name, fn, needs)
        return fn
    return deco


def _downsample(xs, ys, max_points: int = 3000):
    if len(xs) <= max_points:
        return xs, ys
    k = len(xs) // max_points + 1
    return xs[::k], ys[::k]


# ----------------------------- W1 figures ----------------------------------- #
@register("training_loss", needs=["metrics/train.jsonl"])
def fig_training_loss(run: RunLogger) -> Optional[str]:
    import matplotlib.pyplot as plt

    rows = run.load_series("train")
    if not rows:
        return None
    steps = [r["step"] for r in rows]
    loss = [r["loss"] for r in rows]
    xs, ys = _downsample(steps, loss)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, color=style.color(0), label="diffusion loss (MSE)")
    # trivial-predictor reference (predict 0): E||eps - z0||^2 / d ≈ 2 for standardized z0
    ax.axhline(2.0, color=style.CRITICAL, lw=1.5, ls="--", label="trivial baseline ≈ 2.0")
    ax.set_xlabel("training step")
    ax.set_ylabel("flow-matching MSE")
    ax.set_title("GLP training loss")
    ax.legend()
    return str(style.save_fig(fig, run.figures_dir / "training_loss.png"))


@register("lr_schedule", needs=["metrics/train.jsonl"])
def fig_lr_schedule(run: RunLogger) -> Optional[str]:
    import matplotlib.pyplot as plt

    rows = run.load_series("train")
    if not rows or "lr" not in rows[0]:
        return None
    steps = [r["step"] for r in rows]
    lr = [r["lr"] for r in rows]
    xs, ys = _downsample(steps, lr)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(xs, ys, color=style.color(6))
    ax.set_xlabel("training step")
    ax.set_ylabel("learning rate")
    ax.set_title("LR schedule (cosine + warmup)")
    return str(style.save_fig(fig, run.figures_dir / "lr_schedule.png"))


@register("generation_pca", needs=["metrics/pca_proj.npz"])
def fig_generation_pca(run: RunLogger) -> Optional[str]:
    import matplotlib.pyplot as plt
    import numpy as np

    d = np.load(run.metrics_dir / "pca_proj.npz")
    R, G = d["real"], d["gen"]

    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(R[:, 0], R[:, 1], s=6, alpha=0.35, color=style.color(0),
               edgecolors="none", label="real")
    ax.scatter(G[:, 0], G[:, 1], s=6, alpha=0.35, color=style.color(1),
               edgecolors="none", label="generated")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Real vs generated (real's top-2 PCs)")
    ax.legend(markerscale=2)
    return str(style.save_fig(fig, run.figures_dir / "generation_pca.png"))


# ----------------------------- W3 figures ----------------------------------- #
@register("probing_auc", needs=["metrics/probing.json"])
def fig_probing_auc(run: RunLogger) -> Optional[str]:
    """Aggregate test AUC per method with 95% bootstrap CIs (PLAN §7 level (b))."""
    import json

    import matplotlib.pyplot as plt

    d = json.loads((run.metrics_dir / "probing.json").read_text())
    agg = d.get("aggregate", {})
    if not agg:
        return None
    methods = list(agg.keys())
    means = [agg[m]["mean_test_auc"] for m in methods]
    lo = [agg[m]["mean_test_auc"] - agg[m]["ci_low"] for m in methods]
    hi = [agg[m]["ci_high"] - agg[m]["mean_test_auc"] for m in methods]

    fig, ax = plt.subplots(figsize=(1.5 * len(methods) + 2, 4))
    bars = ax.bar(methods, means, color=[style.color(i) for i in range(len(methods))],
                  width=0.6, yerr=[lo, hi], capsize=5,
                  error_kw={"ecolor": style.SECONDARY, "lw": 1.5})
    ax.bar_label(bars, fmt="%.3f", padding=10, color=style.SECONDARY)
    ax.axhline(0.5, color=style.BASELINE, lw=1.2, ls="--")
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("mean test AUC")
    n = agg[methods[0]]["n_concepts"]
    ax.set_title(f"1-D probing: mean test AUC over {n} concepts (95% CI)")
    return str(style.save_fig(fig, run.figures_dir / "probing_auc.png"))


@register("probing_per_concept", needs=["metrics/probing.json"])
def fig_probing_per_concept(run: RunLogger) -> Optional[str]:
    """Per-concept GLP-vs-SAE scatter: above the diagonal = GLP isolates it better."""
    import json

    import matplotlib.pyplot as plt

    d = json.loads((run.metrics_dir / "probing.json").read_text())
    rows = d.get("results", [])
    glp = {r["concept"]: r["test_auc"] for r in rows if r["method"] == "glp"}
    sae = {r["concept"]: r["test_auc"] for r in rows if r["method"] == "sae"}
    shared = sorted(set(glp) & set(sae))
    if not shared:
        return None
    xs = [sae[c] for c in shared]
    ys = [glp[c] for c in shared]

    fig, ax = plt.subplots(figsize=(5.2, 5))
    lim = [min(0.5, min(xs + ys)) - 0.02, 1.005]
    ax.plot(lim, lim, color=style.BASELINE, lw=1.2, ls="--", zorder=1)
    ax.scatter(xs, ys, s=42, color=style.color(0), alpha=0.75,
               edgecolors=style.SURFACE, linewidths=1.5, zorder=2)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("SAE feature — test AUC")
    ax.set_ylabel("GLP meta-neuron — test AUC")
    n_win = sum(1 for a, b in zip(xs, ys) if b > a)
    ax.set_title(f"Per-concept: GLP vs SAE ({n_win}/{len(shared)} above diagonal)")
    return str(style.save_fig(fig, run.figures_dir / "probing_per_concept.png"))


@register("generation_fd", needs=["metrics/summary.json"])
def fig_generation_fd(run: RunLogger) -> Optional[str]:
    import matplotlib.pyplot as plt

    s = run.load_summary()
    if "fd_generated_vs_real" not in s:
        return None
    labels, vals, colors = [], [], []
    labels.append("FD(gen, real)"); vals.append(s["fd_generated_vs_real"]); colors.append(style.color(0))
    if "fd_lower_bound" in s:
        labels.append("FD(train, val)\nlower bound"); vals.append(s["fd_lower_bound"]); colors.append(style.BASELINE)
    if "fd_sae_recon" in s:  # W2 anchor when available
        labels.append("FD(SAE recon)"); vals.append(s["fd_sae_recon"]); colors.append(style.color(1))

    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 1.5, 4))
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    ax.bar_label(bars, fmt="%.2f", padding=3, color=style.SECONDARY)
    ax.set_ylabel("Fréchet distance")
    ax.set_title("Generation quality (lower is better)")
    ax.margins(y=0.15)
    return str(style.save_fig(fig, run.figures_dir / "generation_fd.png"))
