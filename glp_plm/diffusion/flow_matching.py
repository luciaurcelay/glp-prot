"""Flow-matching objective (PLAN §2.1/§6.3; verified in INTERPLM_MAP §6).

Convention (matches GLP reference up to notation): t in [0,1],
    z_t    = (1 - t) * z0 + t * eps          # t=1 pure noise, t=0 clean
    target = eps - z0   (the velocity u)
    loss   = MSE(û_theta(z_t, t), target)

The GLP reference implements the identical thing via diffusers'
FlowMatchEulerDiscreteScheduler, where sigma plays the role of t
(flow_matching.py:32-35). We keep a stdlib-only version so training/tests
have no diffusers dependency; sigma-spacing only matters for reproducing
their published FD magnitudes (INTERPLM_MAP §6, D6).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def sample_t(batch: int, device=None, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """t ~ U(0, 1), shape (batch,)."""
    return torch.rand(batch, device=device, generator=generator)


def interpolate(z0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """z_t = (1 - t) z0 + t eps. `t` broadcasts over the last dim(s)."""
    t = _broadcast_t(t, z0.ndim)
    return (1.0 - t) * z0 + t * eps


def velocity_target(z0: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """u = eps - z0."""
    return eps - z0


def _broadcast_t(t: torch.Tensor, ndim: int) -> torch.Tensor:
    # (N,) -> (N, 1, ... ) so it broadcasts against (N, ..., d)
    while t.ndim < ndim:
        t = t.unsqueeze(-1)
    return t


def flow_matching_loss(
    denoiser,
    z0: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    t: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """Sample eps, t; form z_t; MSE(û, eps - z0).

    z0 is expected already standardized (see diffusion.scaler). Returns
    (loss, aux) where aux carries tensors useful for logging/debugging.
    """
    device = z0.device
    n = z0.shape[0]
    if t is None:
        t = sample_t(n, device=device, generator=generator)
    eps = torch.randn(z0.shape, device=device, generator=generator, dtype=z0.dtype)
    z_t = interpolate(z0, eps, t)
    target = velocity_target(z0, eps)
    pred = denoiser(z_t, t)
    loss = torch.nn.functional.mse_loss(pred, target)
    return loss, {"z_t": z_t, "eps": eps, "t": t, "target": target, "pred": pred}
