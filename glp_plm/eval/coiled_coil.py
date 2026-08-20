"""Sequence-only coiled-coil propensity score.

DSSP has no coiled-coil class -- it only reports local backbone shape (H/G/I helix,
E/B strand, ...), so a genuine coiled coil (two or more helices super-coiling around
each other with a repeating 7-residue "heptad" register: hydrophobic core at
positions a/d, often charged at e/g for inter-helix salt bridges) is indistinguishable
from an ordinary long helix by DSSP alone -- `fold_steering_outputs.py --target-ss
helix` says so explicitly. This module scores the PRIMARY SEQUENCE instead: a rough,
from-scratch reimplementation of the classic Lupas/COILS heptad idea using a
physicochemical propensity model (Kyte-Doolittle hydrophobicity for the core, a
charged-residue indicator for the flanking positions) rather than COILS's own trained
MTIDK substitution matrix, which isn't available here. Needs no fold, no GPU, no SAE --
just the amino acid string.

Calibrated empirically against real Swiss-Prot "Coiled coil" labels by
`scripts/calibrate_coiled_coil.py` (same val-then-test F1 protocol as every other
concept detector in this repo, `glp_plm.eval.probing_f1.run_concept_f1`), not by eye.
Treat this as a best-effort proxy, not ground truth -- it has not been validated
against a real coiled-coil predictor (Marcoil/DeepCoil/paircoil2) or a genuine
structure-based check (e.g. SOCKET on a multi-chain fold).
"""
from __future__ import annotations

import numpy as np

# Kyte & Doolittle (1982) hydrophobicity scale.
KD_HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
CHARGED = set("DEKR")

MIN_WINDOW = 14   # 2 heptads -- the shortest register scan that still means anything
DEFAULT_WINDOW = 28  # 4 heptads, matches classic COILS


def coiled_coil_profile(seq: str, window: int = DEFAULT_WINDOW) -> np.ndarray:
    """Per-residue coiled-coil propensity, same length as `seq`. NaN where no window
    of at least `MIN_WINDOW` residues fits (only possible for sequences shorter than
    MIN_WINDOW itself)."""
    n = len(seq)
    prof = np.full(n, np.nan)
    if n < MIN_WINDOW:
        return prof

    hydro = np.array([KD_HYDROPHOBICITY.get(aa, 0.0) for aa in seq])
    charged = np.array([1.0 if aa in CHARGED else 0.0 for aa in seq])
    w = min(window, n)

    def windowed_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        cs_val = np.concatenate([[0.0], np.cumsum(values * mask)])
        cs_cnt = np.concatenate([[0.0], np.cumsum(mask.astype(np.float64))])
        val_sum = cs_val[w:] - cs_val[:-w]
        cnt = cs_cnt[w:] - cs_cnt[:-w]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cnt > 0, val_sum / np.maximum(cnt, 1), 0.0)

    best_window_score = None  # (n - w + 1,)
    for register in range(7):
        pos = (np.arange(n) + register) % 7
        core_mask = (pos == 0) | (pos == 3)
        flank_mask = (pos == 4) | (pos == 6)
        surface_mask = ~core_mask & ~flank_mask

        core_mean = windowed_mean(hydro, core_mask)
        flank_mean = windowed_mean(charged, flank_mask)
        surface_mean = windowed_mean(hydro, surface_mask)
        score = core_mean + 0.5 * flank_mean - 0.5 * surface_mean

        best_window_score = score if best_window_score is None else np.maximum(best_window_score, score)

    # Window starts are 0..n-w, so their centers (start + half) are a CONTIGUOUS run
    # of integers `half .. half + (n - w)`. Residues before/after that run (within
    # half a window of either sequence end, whose own window would run off the
    # sequence) inherit the nearest window's score -- i.e. simple edge padding.
    half = w // 2
    prof[half:half + len(best_window_score)] = best_window_score
    prof[:half] = best_window_score[0]
    prof[half + len(best_window_score):] = best_window_score[-1]
    return prof


def coiled_coil_score(seq: str, window: int = DEFAULT_WINDOW) -> float:
    """Single sequence-level summary: the max per-residue propensity."""
    prof = coiled_coil_profile(seq, window)
    valid = prof[~np.isnan(prof)]
    return float(np.max(valid)) if len(valid) else float("nan")
