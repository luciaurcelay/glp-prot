"""Fréchet Distance between two activation sets (PLAN §6.4; W1 = FD + lower bound).

FD (a.k.a. FID form) = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2 (C_1 C_2)^{1/2}).
Computed on standardized or raw activations (be consistent between the two sets).

For G1 we report:
  - FD(generated, real_held_out)                      -> the number we care about
  - FD(train_sample, val_sample) = lower bound anchor  (PLAN §6.4 anchor (i))
The SAE-reconstruction-FD anchor and matching GLP's exact one-token-per-doc sampling
protocol are W2/G2 concerns (PLAN §6.4, H3) — not needed to clear G1.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import linalg


def _mu_sigma(x: torch.Tensor | np.ndarray):
    x = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    x = x.reshape(-1, x.shape[-1]).astype(np.float64)
    mu = x.mean(axis=0)
    sigma = np.cov(x, rowvar=False)
    return mu, sigma


def frechet_distance(a, b, eps: float = 1e-6) -> float:
    """Fréchet distance between activation sets `a` and `b`, each (N, d)."""
    mu1, s1 = _mu_sigma(a)
    mu2, s2 = _mu_sigma(b)
    diff = mu1 - mu2

    # errstate silences spurious Apple-Accelerate BLAS fp warnings inside sqrtm.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if not np.isfinite(covmean).all():
        # numerical fallback: nudge the diagonals (as in the standard FID impl)
        offset = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + offset) @ (s2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fd = diff @ diff + np.trace(s1) + np.trace(s2) - 2.0 * np.trace(covmean)
    return float(fd)
