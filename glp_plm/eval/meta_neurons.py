"""GLP meta-neuron bank for 1-D probing (PLAN §6.5).

The meta-neuron is the **input to `down_proj`** at each block —
`SiLU(gate_proj(x)·time_proj(t_emb))·up_proj(x)` — so it INCLUDES the time factor
and is therefore a function of the probing time. Bank width = n_blocks · d_mlp
(6 blocks at d=320 ⇒ 7,680). See docs/INTERPLM_MAP.md.

`u` vs `t` (pinned; this is easy to get backwards)
--------------------------------------------------
GLP's released probe parameterises probing time as **`u` = schedule fraction**
(default 0.9). In their scheduler, `sigma ≈ 1 − u`, and *our* `t` IS that sigma
(t=1 pure noise, t=0 clean). So:

        t = 1 − u          u=0.9  ->  t=0.1  (lightly noised, near-clean)

Higher `u` = cleaner input. We therefore sweep `u` and convert here, so config and
paper both speak GLP's `u` while the model keeps our `t` convention.
"""

from __future__ import annotations

import torch


def u_to_t(u: float) -> float:
    """Convert GLP's schedule-fraction `u` to our noise level `t` (t = 1 - u)."""
    return 1.0 - float(u)


@torch.no_grad()
def meta_neuron_bank(model, z0: torch.Tensor, u: float, batch_size: int = 8192,
                     device: str = "cpu", generator: torch.Generator | None = None
                     ) -> torch.Tensor:
    """Meta-neurons for standardized activations `z0` at probing time `u`.

    Noises z0 to level t = 1-u (one draw, as in the released probe), runs a single
    forward pass, and returns the concatenated per-block down_proj inputs.

    Args:
        model: a GLPDenoiser (eval mode)
        z0:    (N, d) STANDARDIZED activations
        u:     schedule fraction; higher = cleaner
    Returns:
        (N, n_blocks * d_mlp) float32 on CPU
    """
    t = u_to_t(u)
    model = model.to(device).eval()
    out = []
    for i in range(0, z0.shape[0], batch_size):
        chunk = z0[i:i + batch_size].to(device).float()
        eps = torch.randn(chunk.shape, device=device, generator=generator)
        z_t = (1.0 - t) * chunk + t * eps
        tt = torch.full((chunk.shape[0],), t, device=device)
        out.append(model.meta_neurons(z_t, tt).detach().cpu())
    return torch.cat(out, dim=0)
