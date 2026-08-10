"""Shared matplotlib style so every figure reads as one system.

Brand palette, two hue families, three shades each:
  orange = GLP / model-generated   purple = reference / real / SAE
Dark shade = the headline entity in its family; light shade = that family's
"raw"/naive baseline sibling; mid shade = a spare accent for charts that aren't
part of the glp/sae/raw_layer/raw_mlp identity set (e.g. precision/recall).

Method identity (glp/sae/raw_layer/raw_mlp) is assigned by NAME, never by
position. `glp_plm.eval.baselines.METHODS` and a script's own method list don't
always iterate in the same order (probing_f1's did not, pre-fix), and "color
follows the entity, never its rank" means position-based indexing would
silently swap identities between figures whenever iteration order differs.

Every adjacency actually used in a chart (`METHOD_ORDER`'s bar order, the 2-way
real/generated split, the full PALETTE as a Hamiltonian path) is validated with
the dataviz skill's `scripts/validate_palette.js` (CVD simulation + normal-vision
floor, target/floor ΔE 8/15 in OKLab). Two pairs the brand hues do NOT clear
adjacent: GLP's dark orange next to either the dark or mid purple (14.1-14.6 dE,
below the 15 normal-vision floor) — this is why `sae` sits at the *light* purple
and `raw_layer` at the *dark* one rather than the more obvious dark-dark pairing.
Light orange/light purple next to their own mid shade also fails (5.3-7.6 dE) —
those two only ever appear non-adjacent (bar charts with a 3rd color between
them, or one used and the other not) in this codebase; don't add a chart that
puts them side by side without re-validating.
"""

from __future__ import annotations

from pathlib import Path

# --- Brand hues --------------------------------------------------------------
ORANGE_DARK = "#d63226"
ORANGE_MID = "#f17628"
ORANGE_LIGHT = "#f79858"
PURPLE_DARK = "#a32265"
PURPLE_MID = "#ba5895"
PURPLE_LIGHT = "#d162a4"

# Method identity: fixed, named, reused across every figure. Validated adjacent
# in this exact order (the order bars are actually drawn in, matching
# glp_plm.eval.baselines.METHODS) — glp-sae 15.9dE, sae-raw_layer 16.8dE,
# raw_layer-raw_mlp 32.1dE; all clear the 15 floor.
METHOD_ORDER = ["glp", "sae", "raw_layer", "raw_mlp"]
METHOD_COLORS = {
    "glp": ORANGE_DARK,
    "sae": PURPLE_LIGHT,
    "raw_layer": PURPLE_DARK,
    "raw_mlp": ORANGE_LIGHT,
}

# generation_pca (all-pairs scatter, 2 categories): reuses the same validated
# glp/sae pair so "orange = model output" and "purple = ground truth" reads the
# same way in every figure that draws the distinction.
GENERATED_COLOR = ORANGE_DARK
REAL_COLOR = PURPLE_LIGHT

# Generic categorical fallback for charts with no fixed entity set. Fixed order
# (never cycled), validated end-to-end as one adjacent Hamiltonian path.
PALETTE = [ORANGE_DARK, PURPLE_LIGHT, PURPLE_DARK, ORANGE_MID, PURPLE_MID, ORANGE_LIGHT]

SURFACE = "#fcfcfb"      # chart surface
INK = "#0b0b0b"          # primary text
SECONDARY = "#52514e"    # secondary text
MUTED = "#898781"        # axis / tick labels
GRID = "#e1e0d9"         # hairline gridline
BASELINE = "#c3c2b7"     # reference lines: anchors, diagonals, thresholds — never a brand hue
GOOD = "#0ca30c"         # status: good (reserved — not currently used)
CRITICAL = "#d03b3b"     # status: critical (reserved — not currently used)


def color(i: int) -> str:
    """Categorical color for slot i (0-based), fixed order. Prefer method_color()
    for glp/sae/raw_layer/raw_mlp — this is for charts with no fixed entity set."""
    return PALETTE[i % len(PALETTE)]


def method_color(name: str) -> str:
    """Color for a fixed method identity (glp/sae/raw_layer/raw_mlp) — same
    entity, same color in every figure, regardless of that figure's own
    dict/JSON key order."""
    return METHOD_COLORS.get(name, MUTED)


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
    """Save a figure as PNG and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path
