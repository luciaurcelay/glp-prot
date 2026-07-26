"""Minimal GLP trainer: flow-matching MSE, cosine LR w/ warmup, grad clip.

Depends only on the model + diffusion (in scope this session). The data pipeline
(activation store / producer-consumer) is deliberately NOT built yet (PLAN §9 W1);
`fit` takes an already-standardized activation tensor or any iterable of batches,
so it is unit-testable on synthetic data and swaps to real activations later.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import torch

from glp_plm.diffusion import flow_matching


def cosine_lr_with_warmup(step: int, total_steps: int, warmup_ratio: float,
                          initial_factor: float, final_factor: float) -> float:
    """Multiplier on base lr (matches the GLP reference schedule shape, docs/INTERPLM_MAP.md)."""
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    if step < warmup_steps:
        return initial_factor + (1.0 - initial_factor) * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final_factor + (1.0 - final_factor) * cosine


def fit(
    model: torch.nn.Module,
    activations: torch.Tensor,          # (N, d), already standardized
    *,
    lr: float,
    batch_size: int,
    num_epochs: int,
    warmup_ratio: float,
    lr_initial_factor: float,
    lr_final_factor: float,
    grad_clip: float,
    device: str = "cpu",
    log_every_n_steps: int = 10,
    seed: int = 0,
    logger=print,
) -> dict:
    """Train `model` (a GLPDenoiser) on `activations`. Returns a small history dict."""
    torch.manual_seed(seed)
    model = model.to(device).train()
    logger(f"[trainer] true parameter count: {sum(p.numel() for p in model.parameters()):,}")

    n = activations.shape[0]
    steps_per_epoch = max(1, n // batch_size)
    total_steps = steps_per_epoch * num_epochs
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    history = {"loss": [], "lr": []}
    step = 0
    for epoch in range(num_epochs):
        perm = torch.randperm(n, generator=gen)
        for b in range(steps_per_epoch):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            z0 = activations[idx].to(device).float()

            loss, _ = flow_matching.flow_matching_loss(model, z0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            mult = cosine_lr_with_warmup(step, total_steps, warmup_ratio,
                                         lr_initial_factor, lr_final_factor)
            for g in opt.param_groups:
                g["lr"] = lr * mult
            opt.step()

            history["loss"].append(loss.item())
            history["lr"].append(lr * mult)
            if step % log_every_n_steps == 0:
                logger(f"[trainer] epoch {epoch} step {step}/{total_steps} "
                       f"loss {loss.item():.4f} lr {lr * mult:.2e}")
            step += 1
    return history
