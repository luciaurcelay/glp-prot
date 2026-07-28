"""Reverse-integration sampler for flow matching (PLAN §6.3; docs/INTERPLM_MAP.md).

Euler integration of dz/dt = û_theta(z, t) from t_start down to 0:
    for consecutive times t_cur -> t_next (t_next < t_cur):
        z <- z + (t_next - t_cur) * û_theta(z, t_cur)

- `sample`:            t_start=1  -> generate from noise (z_start ~ N(0,I))
- `sample_on_manifold`: t_start<1 -> SDEdit cleanup of an edited activation

For a CONSTANT velocity field û = eps - z0 this integrator is EXACT regardless of
`num_steps`, because z(t) = z0 + t*(eps - z0) is linear in t (docs/INTERPLM_MAP.md). That is
what the flow-matching round-trip test relies on.

`denoiser` is any callable (z, t) -> velocity of matching shape; t is a scalar
tensor. GLPDenoiser satisfies this signature directly.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

Velocity = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _timesteps(t_start: float, num_steps: int, device) -> torch.Tensor:
    return torch.linspace(t_start, 0.0, num_steps + 1, device=device)


@torch.no_grad()
def sample(
    denoiser: Velocity,
    z_start: torch.Tensor,
    num_steps: int = 20,
    t_start: float = 1.0,
) -> torch.Tensor:
    """Integrate from t_start to 0. With t_start=1 and z_start~N(0,I): gen-from-noise."""
    z = z_start
    ts = _timesteps(t_start, num_steps, z.device)
    for i in range(num_steps):
        t_cur, t_next = ts[i], ts[i + 1]
        v = denoiser(z, t_cur)
        z = z + (t_next - t_cur) * v
    return z


@torch.no_grad()
def sample_on_manifold(
    denoiser: Velocity,
    z_edit: torch.Tensor,
    num_steps: int = 20,
    t_start: float = 0.5,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """SDEdit: noise `z_edit` to t_start, then reverse-integrate to 0 (PLAN §6.6).

    z_edit is expected standardized. Caller de-standardizes the result.
    """
    eps = torch.randn(z_edit.shape, device=z_edit.device, dtype=z_edit.dtype, generator=generator)
    z_noisy = (1.0 - t_start) * z_edit + t_start * eps
    return sample(denoiser, z_noisy, num_steps=num_steps, t_start=t_start)
