"""Top-2 PCA overlap of real vs generated activations (PLAN §6.4; G1 sanity).

Fit PCA on the REAL set, project both real and generated into the top-2 PCs, and
return overlap metrics plus the (subsampled) projected points. Plotting lives in
experiments.figures (compute here, plot there) so figures can be re-rendered freely.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


def _np(x):
    x = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    return x.reshape(-1, x.shape[-1]).astype(np.float64)


def pca_overlap(real, generated, n_components: int = 2, max_points: int = 4000
                ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Return (metrics, projections). projections has 'real'/'gen' (<=max_points, k)."""
    R = _np(real)
    G = _np(generated)

    mu = R.mean(axis=0)
    Rc = R - mu
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        _, _, vt = np.linalg.svd(Rc, full_matrices=False)
        comps = vt[:n_components]                   # (k, d)
        R_proj = Rc @ comps.T                       # (Nr, k)
        G_proj = (G - mu) @ comps.T                 # (Ng, k)

    real_frac = (R_proj ** 2).sum() / (Rc ** 2).sum()
    gen_frac = (G_proj ** 2).sum() / (((G - mu) ** 2).sum() + 1e-12)
    r_std = R_proj.std(0)
    metrics = {
        "pca_real_var_captured_frac": float(real_frac),
        "pca_gen_var_captured_frac": float(gen_frac),
        "pca_pc_mean_gap_norm": float(np.linalg.norm((R_proj.mean(0) - G_proj.mean(0)) / (r_std + 1e-12))),
        "pca_pc_std_ratio": float(np.linalg.norm(G_proj.std(0)) / (np.linalg.norm(r_std) + 1e-12)),
    }

    def _sub(a):
        if a.shape[0] > max_points:
            idx = np.random.default_rng(0).choice(a.shape[0], max_points, replace=False)
            a = a[idx]
        return a.astype(np.float32)

    return metrics, {"real": _sub(R_proj), "gen": _sub(G_proj)}
