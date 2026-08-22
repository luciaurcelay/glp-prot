#!/usr/bin/env python
"""Metric-A figure: GLP FD lower-bound ratio (FD@1000 / intrinsic floor) vs depth,
ESM-2-8M vs ESM-2-650M, on the matched 100M-activation UniRef50 depth grids.

Absolute FD is not comparable across PLMs (different dims/scales), so we plot the
lower-bound ratio, which normalizes each PLM by its own intrinsic FD floor. Lower = the
GLP generates activations closer to real. Numbers from
runs/{uniref50_100m_4096_b*,esm650m_100m_4096_d*}/metrics/fd_vs_steps.json (2026-08-22).

Uses the project palette (glp_plm.experiments.style): a two-series cross-model chart, so
the two curves take the generic categorical slots color(0)/color(1) (fixed order, validated
adjacent). Saves docs/figures/metric_a_ratio_vs_depth.png. Data is inlined (4 points/series)
so this runs anywhere without the cluster.
"""
import _bootstrap  # noqa: F401  (puts repo root on sys.path)

from matplotlib.ticker import FixedLocator, ScalarFormatter

from glp_plm.experiments import style

DEPTHS = [3, 6, 12, 24]
RATIO_8M = [4.4667, 4.4701, 4.1477, 4.0579]
RATIO_650M = [4.2363, 4.2344, 3.9740, 3.8610]

style.apply_style()
import matplotlib.pyplot as plt

C_8M, C_650M = style.color(0), style.color(1)

fig, ax = plt.subplots(figsize=(6.2, 4.4))

for ratios, c, label, tag in [(RATIO_8M, C_8M, "ESM-2-8M  (d=320)", "8M"),
                              (RATIO_650M, C_650M, "ESM-2-650M  (d=1280, layer 18)", "650M")]:
    ax.plot(DEPTHS, ratios, "-o", color=c, lw=2.0, ms=8, mfc=c,
            mec=style.SURFACE, mew=1.2, label=label, zorder=3, clip_on=False)
    ax.annotate(tag, (DEPTHS[-1], ratios[-1]), xytext=(10, 0), textcoords="offset points",
                va="center", color=c, fontsize=11, fontweight="bold")

# log2 x so the doubling grid (3,6,12,24) is evenly spaced.
ax.set_xscale("log", base=2)
ax.xaxis.set_major_locator(FixedLocator(DEPTHS))
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.set_xlim(2.7, 30)
ax.set_ylim(3.7, 4.55)

ax.set_xlabel("GLP depth (number of blocks)")
ax.set_ylabel("FD lower-bound ratio  (FD@1000 / floor)")
ax.set_title("GLP depth scaling: Fréchet-distance ratio vs blocks\n"
             "matched 100M-activation UniRef50 grids")

# "lower is better" cue, since the axis direction isn't self-evident.
ax.annotate("lower = closer to real", xy=(0.02, 0.04), xycoords="axes fraction",
            fontsize=9, color=style.MUTED, style="italic")

ax.legend(loc="upper right")

out = style.save_fig(fig, "docs/figures/metric_a_ratio_vs_depth.png")
print("saved", out)
