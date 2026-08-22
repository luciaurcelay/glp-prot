#!/usr/bin/env python
"""Metric-A figure: GLP FD lower-bound ratio (FD@1000 / intrinsic floor) vs depth,
ESM-2-8M vs ESM-2-650M, on the matched 100M-activation UniRef50 depth grids.

Absolute FD is not comparable across PLMs (different dims/scales), so we plot the
lower-bound ratio, which normalizes each PLM by its own intrinsic FD floor. Lower = the
GLP generates activations closer to real. Numbers from
runs/{uniref50_100m_4096_b*,esm650m_100m_4096_d*}/metrics/fd_vs_steps.json (2026-08-22).

Saves docs/figures/metric_a_ratio_vs_depth.png (+ .pdf). Data is inlined (4 points/series)
so this runs anywhere without the cluster.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter

DEPTHS = [3, 6, 12, 24]
RATIO_8M = [4.4667, 4.4701, 4.1477, 4.0579]
RATIO_650M = [4.2363, 4.2344, 3.9740, 3.8610]

# Okabe-Ito: colorblind-safe across all CVD types.
BLUE, VERM = "#0072B2", "#D55E00"
INK, MUTED = "#1a1a1a", "#6b6b6b"

fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=200)

for ratios, color, label in [(RATIO_8M, BLUE, "ESM-2-8M  (d=320)"),
                             (RATIO_650M, VERM, "ESM-2-650M  (d=1280, layer 18)")]:
    ax.plot(DEPTHS, ratios, "-o", color=color, lw=2.0, ms=8, mfc=color,
            mec="white", mew=1.2, label=label, zorder=3, clip_on=False)

# Direct end-labels (identity not by color alone).
ax.annotate("8M", (DEPTHS[-1], RATIO_8M[-1]), xytext=(10, 0), textcoords="offset points",
            va="center", color=BLUE, fontsize=11, fontweight="bold")
ax.annotate("650M", (DEPTHS[-1], RATIO_650M[-1]), xytext=(10, 0), textcoords="offset points",
            va="center", color=VERM, fontsize=11, fontweight="bold")

# log2 x so the doubling grid (3,6,12,24) is evenly spaced.
ax.set_xscale("log", base=2)
ax.xaxis.set_major_locator(FixedLocator(DEPTHS))
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.set_xlim(2.7, 30)
ax.set_ylim(3.7, 4.55)

ax.set_xlabel("GLP depth (number of blocks)", fontsize=11, color=INK)
ax.set_ylabel("FD lower-bound ratio  (FD@1000 / floor)", fontsize=11, color=INK)
ax.set_title("GLP depth scaling: Fréchet-distance ratio vs blocks\n"
             "matched 100M-activation UniRef50 grids", fontsize=12, color=INK, pad=10)

# "lower is better" cue, since the axis direction isn't self-evident.
ax.annotate("lower = closer to real", xy=(0.02, 0.04), xycoords="axes fraction",
            fontsize=9, color=MUTED, style="italic")

ax.grid(True, which="major", axis="both", color="#e6e6e6", lw=0.8, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#cccccc")
ax.tick_params(colors=MUTED, labelsize=10)
ax.legend(frameon=False, fontsize=10, loc="upper right")

fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "docs" / "figures" / "metric_a_ratio_vs_depth.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
print("saved", out)
print("saved", out.with_suffix(".pdf"))
