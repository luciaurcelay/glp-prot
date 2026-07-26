"""GLP denoiser: û_θ(z_t, t) predicting flow-matching velocity, ported to d=320.

Stack of SwiGLU blocks with residuals, no attention, single-token modeling
(PLAN §2.1; docs/INTERPLM_MAP.md). Matches the released GLP architecture
(generative_latent_prior/glp/denoiser.py:129-211):
    in_proj: d_input -> d_model ; N x SwiGLUBlock @ (d_model, d_mlp) ; LN ; out_proj -> d_input.

Accepts either (N, d_input) or (B, S, d_input); operates per-token either way.
`meta_neurons()` returns the per-block down_proj-input tensor for probing (docs/INTERPLM_MAP.md).
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from glp_plm.config import ModelConfig
from glp_plm.model.swiglu_block import SwiGLUBlock
from glp_plm.model.timestep_embed import TimestepEmbed


class GLPDenoiser(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg = cfg.resolve()
        self.cfg = cfg
        d_in, d_model, d_mlp = cfg.activation_dim, cfg.d_model, cfg.d_mlp

        self.in_proj = nn.Linear(d_in, d_model)
        self.blocks = nn.ModuleList(
            [SwiGLUBlock(d_model, d_mlp) for _ in range(cfg.n_blocks)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_in)
        self.time_embed = TimestepEmbed(d_model)

    # -- shape helpers: collapse (B,S,d) -> (N,d), remember how to restore -----
    @staticmethod
    def _flatten(z: torch.Tensor):
        if z.ndim == 3:
            b, s, d = z.shape
            return z.reshape(b * s, d), (b, s)
        assert z.ndim == 2, f"expected (N,d) or (B,S,d), got {z.shape}"
        return z, None

    @staticmethod
    def _restore(z: torch.Tensor, shape) -> torch.Tensor:
        if shape is None:
            return z
        b, s = shape
        return z.reshape(b, s, -1)

    def _prep_t(self, t: torch.Tensor, n: int) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(n)
        return t.reshape(-1).to(self.in_proj.weight.device)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict velocity û for noisy latents z_t at time t in [0,1]."""
        flat, shape = self._flatten(z_t)
        t_emb = self.time_embed(self._prep_t(t, flat.shape[0]))
        x = self.in_proj(flat)
        for block in self.blocks:
            x = block(x, t_emb)
        x = self.out_proj(self.ln(x))
        return self._restore(x, shape)

    @torch.no_grad()
    def meta_neurons(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Per-block down_proj-input tensor -> (N, n_blocks * d_mlp).

        This is the GLP "meta-neuron" bank read at probing time (docs/INTERPLM_MAP.md).
        """
        flat, _ = self._flatten(z_t)
        t_emb = self.time_embed(self._prep_t(t, flat.shape[0]))
        x = self.in_proj(flat)
        neurons: List[torch.Tensor] = []
        for block in self.blocks:
            x, mn = block(x, t_emb, return_meta_neuron=True)
            neurons.append(mn)
        return torch.cat(neurons, dim=-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
