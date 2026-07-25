"""Sinusoidal timestep embedding + MLP, matching the GLP reference.

Ref: generative_latent_prior/glp/denoiser.py:76-92 (timestep_embedding) and
:155-159 (time_embed MLP). Verified in RECON §5.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor, dim: int, max_period: int = 10000
) -> torch.Tensor:
    """Sinusoidal embedding of a (N,) tensor of timesteps -> (N, dim).

    Mirrors DiT / the GLP reference (denoiser.py:76-92): cos first, then sin.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepEmbed(nn.Module):
    """sinusoidal(t) -> Linear -> SiLU -> Linear, producing a (N, d_model) embedding."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = sinusoidal_timestep_embedding(timesteps, self.d_model)
        return self.mlp(emb)
