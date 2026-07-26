"""Shared matplotlib style so every figure reads as one system.

Palette + ink/grid from the dataviz reference palette (light surface). Categorical
hues are assigned in FIXED order, never cycled (dataviz non-negotiable): slot i is
always the same entity across figures. For scatter/all-pairs forms use at most the
first 3 slots (validated all-pairs); bar/line adjacent forms may use more.
"""

from __future__ import annotations

from pathlib import Path

# Categorical palette (light), fixed slot order: blue, orange, aqua, yellow,
# magenta, green, violet, red. Validated colorblind-safe on the light surface.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

SURFACE = "#fcfcfb"      # chart surface
INK = "#0b0b0b"          # primary text
SECONDARY = "#52514e"    # secondary text
MUTED = "#898781"        # axis / tick labels
GRID = "#e1e0d9"         # hairline gridline
BASELINE = "#c3c2b7"     # axis / baseline
GOOD = "#0ca30c"         # status: good
CRITICAL = "#d03b3b"     # status: critical (e.g. baselines/limits)


def color(i: int) -> str:
    """Categorical color for slot i (0-based), fixed order."""
    return PALETTE[i % len(PALETTE)]


def apply_style() -> None:
    """Set global rcParams: sans, recessive grid/axes, surface background."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["system-ui", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 1.0,
        "axes.labelcolor": SECONDARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": SECONDARY,
        "ytick.labelcolor": SECONDARY,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "lines.linewidth": 2.0,
        "figure.dpi": 120,
    })


def save_fig(fig, path: str | Path) -> Path:
    """Save a figure as PNG (+ PDF sibling) and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path
