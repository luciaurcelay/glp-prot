#!/usr/bin/env python
"""Metric-B figure: 1-D probing mean test AUC by method, ESM-2-8M vs ESM-2-650M.

Grouped bars — color = method identity (glp_plm.experiments.style.method_color), texture =
host (8M solid, 650M hatched). Error bars are 95% bootstrap CIs. The two hosts are probed on
different concept sets (8M: 104, from paper_draft.tex Tab. 2, 24-block/500M; 650M: 166, d24,
metrics/probing.json 2026-08-23), so absolute AUCs are not directly comparable across hosts —
the within-host GLP-over-baseline ordering is the readable quantity. Data inlined; runs
anywhere. Saves docs/figures/probing_auc_8m_vs_650m.png.
"""
import _bootstrap  # noqa: F401

import numpy as np

from glp_plm.experiments import style

# order matches paper_draft.tex Table 2 columns
METHODS = ["glp", "sae", "raw_mlp", "raw_layer"]
LABELS = {"glp": "GLP\nmeta-neurons", "sae": "SAE\n(InterPLM)",
          "raw_mlp": "ESM FFN\nneurons", "raw_layer": "Raw\nresidual"}
# mean, ci_low, ci_high
AUC_8M = {"glp": (0.897, 0.879, 0.914), "sae": (0.842, 0.821, 0.862),
          "raw_mlp": (0.876, 0.858, 0.894), "raw_layer": (0.827, 0.808, 0.846)}
AUC_650M = {"glp": (0.9379, 0.9247, 0.9500), "sae": (0.8899, 0.8716, 0.9073),
            "raw_mlp": (0.8415, 0.8244, 0.8583), "raw_layer": (0.8509, 0.8334, 0.8677)}

style.apply_style()
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

x = np.arange(len(METHODS))
w = 0.38

fig, ax = plt.subplots(figsize=(7.2, 4.4))

for host, data, off, hatch in [("8M", AUC_8M, -w / 2, None),
                               ("650M", AUC_650M, +w / 2, "///")]:
    means = [data[m][0] for m in METHODS]
    lo = [data[m][0] - data[m][1] for m in METHODS]
    hi = [data[m][2] - data[m][0] for m in METHODS]
    colors = [style.method_color(m) for m in METHODS]
    bars = ax.bar(x + off, means, w, color=colors, hatch=hatch,
                  edgecolor=style.SURFACE, linewidth=0 if hatch is None else 0.0,
                  yerr=[lo, hi], capsize=4,
                  error_kw={"ecolor": style.SECONDARY, "lw": 1.2}, zorder=3)
    ax.bar_label(bars, fmt="%.3f", padding=6, fontsize=8, color=style.SECONDARY)

ax.axhline(0.5, color=style.BASELINE, lw=1.2, ls="--", zorder=1)
ax.set_xticks(x)
ax.set_xticklabels([LABELS[m] for m in METHODS])
ax.set_ylim(0.4, 1.0)
ax.set_ylabel("mean test AUC")
ax.set_title("1-D probing: mean test AUC by method and host")

# texture legend (host); color already carries method on the x-axis
host_legend = [Patch(facecolor=style.MUTED, edgecolor=style.SURFACE, label="ESM-2-8M  (104 concepts)"),
               Patch(facecolor=style.MUTED, edgecolor=style.SURFACE, hatch="///",
                     label="ESM-2-650M  (166 concepts)")]
ax.legend(handles=host_legend, loc="upper right", fontsize=9)

out = style.save_fig(fig, "docs/figures/probing_auc_8m_vs_650m.png")
print("saved", out)
