"""Top-2 PCA overlap of real vs generated activations (PLAN §6.4; G1 sanity).

Fit PCA on the REAL set, project both real and generated into the top-2 PCs, and:
  - report a scalar overlap (ratio of generated variance captured by real's PCs, and
    the mean/cov gap in PC space) so G1 has a number, and
  - optionally save a scatter (the visual "non-garbage" check).

Pure numpy/torch (+ optional matplotlib) — no InterPLM dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch


def _np(x):
    x = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    return x.reshape(-1, x.shape[-1]).astype(np.float64)


def pca_overlap(real, generated, n_components: int = 2, save_path: Optional[str | Path] = None) -> dict:
    """Fit PCA on `real`, project both, return overlap metrics (+ optional scatter)."""
    R = _np(real)
    G = _np(generated)

    mu = R.mean(axis=0)
    Rc = R - mu
    # top PCs from real (SVD on centered real). errstate silences spurious
    # Apple-Accelerate BLAS fp warnings on finite data.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        _, _, vt = np.linalg.svd(Rc, full_matrices=False)
        comps = vt[:n_components]                   # (k, d)
        R_proj = Rc @ comps.T                       # (Nr, k)
        G_proj = (G - mu) @ comps.T                 # (Ng, k)

    # fraction of each set's total variance captured by REAL's top-k PCs.
    # Compare gen's fraction to real's own (same PCs) — apples to apples.
    real_frac = (R_proj ** 2).sum() / (Rc ** 2).sum()
    gen_frac = (G_proj ** 2).sum() / (((G - mu) ** 2).sum() + 1e-12)
    # PC-space mean/std gaps, normalized by real's PC std so they're scale-free.
    r_std = R_proj.std(0)
    metrics = {
        "real_var_captured_frac": float(real_frac),   # baseline for gen_var_captured_frac
        "gen_var_captured_frac": float(gen_frac),
        "pc_mean_gap_norm": float(np.linalg.norm((R_proj.mean(0) - G_proj.mean(0)) / (r_std + 1e-12))),
        "pc_std_ratio": float(np.linalg.norm(G_proj.std(0)) / (np.linalg.norm(r_std) + 1e-12)),
    }

    if save_path is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5, 5))
            n = min(5000, R_proj.shape[0], G_proj.shape[0])
            ax.scatter(R_proj[:n, 0], R_proj[:n, 1], s=4, alpha=0.3, label="real")
            ax.scatter(G_proj[:n, 0], G_proj[:n, 1], s=4, alpha=0.3, label="generated")
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend()
            ax.set_title("Real vs generated (real's top-2 PCs)")
            fig.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)
            metrics["scatter_path"] = str(save_path)
        except ImportError:
            pass  # matplotlib optional; metrics still returned

    return metrics
