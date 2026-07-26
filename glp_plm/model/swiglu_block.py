"""GLP SwiGLU MLP block with multiplicative timestep conditioning.

Implemented to match the RELEASED GLP reference exactly
(generative_latent_prior/glp/denoiser.py:94-127), per INTERPLM_MAP §4 (deltas D3, D4, D5):

  * Norm is **LayerNorm** (NOT RMSNorm)                         [D3]
  * Timestep conditioning is a **plain multiplicative** modulation of the gate
    pre-activation by `time_proj(t_emb)` — NO (1+gamma) scale, NO beta bias,
    and the `up` path is NOT modulated                          [D4]
  * The **meta-neuron** is the input to `down_proj`:
        SiLU(gate_proj(x) * time_proj(t_emb)) * up_proj(x)      [D5]
    (post-gating gated-MLP neuron, INCLUDING the time factor).

forward:  x -> LN -> [SwiGLU with t-modulated gate] -> down_proj -> + residual
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class SwiGLUBlock(nn.Module):
    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.ln = nn.LayerNorm(d_model)                 # D3: LayerNorm, not RMSNorm
        self.up_proj = nn.Linear(d_model, d_mlp)
        self.gate_proj = nn.Linear(d_model, d_mlp)
        self.down_proj = nn.Linear(d_mlp, d_model)
        self.time_proj = nn.Linear(d_model, d_mlp)      # D4: modulates the gate
        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,          # (N, d_model)
        t_emb: torch.Tensor,      # (N, d_model)
        return_meta_neuron: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        resid = x
        h = self.ln(x)
        up = self.up_proj(h)
        gate = self.gate_proj(h)
        merged = gate * self.time_proj(t_emb)           # D4: multiplicative on gate
        meta_neuron = self.act(merged) * up             # D5: input to down_proj
        out = self.down_proj(meta_neuron) + resid
        if return_meta_neuron:
            return out, meta_neuron
        return out
