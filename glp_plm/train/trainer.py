"""Minimal GLP trainer: flow-matching MSE, cosine LR w/ warmup, grad clip.

Depends only on the model + diffusion (in scope this session). The data pipeline
(activation store / producer-consumer) is deliberately NOT built yet (PLAN §9 W1);
`fit` takes an already-standardized activation tensor or any iterable of batches,
so it is unit-testable on synthetic data and swaps to real activations later.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable, Optional

import torch

from glp_plm.diffusion import flow_matching


def _save_checkpoint_atomic(state: dict, ckpt_path: Path) -> None:
    """Write a checkpoint so a kill mid-save can't corrupt the live one.

    SLURM sends SIGKILL at the wall-clock TimeLimit; if that lands inside a plain
    in-place `torch.save` of a multi-hundred-MB checkpoint, the file is left
    truncated and the *next* resume crashes on `torch.load` — losing the whole run
    (exactly the failure mode a time-capped run like uniref50_500m is exposed to).
    Writing to a sidecar then `os.replace` (atomic rename on the same filesystem)
    means the live checkpoint is only ever swapped as a complete file: a kill during
    the save leaves the previous good checkpoint intact, costing at most `ckpt_every`
    steps, never everything.
    """
    ckpt_path = Path(ckpt_path)
    tmp = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, ckpt_path)


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
    run_logger=None,
    scaler=None,
    ckpt_path=None,
    ckpt_every: int = 2000,
    resume: bool = True,
) -> dict:
    """Train `model` (a GLPDenoiser) on `activations`. Returns a small history dict.

    `activations` may be a lazy memmap-backed tensor larger than RAM — batches are
    indexed, moved to device, and cast to float32 ONE AT A TIME. Pass `scaler` to
    standardize per batch (never standardize the whole store: at 25M x 320 that is
    32 GB in float32).

    If `run_logger` (experiments.RunLogger) is given, appends {step, loss, lr} to the
    "train" series each step so figures.training_loss can plot it.

    `ckpt_path` enables periodic {model, optimizer, step, total_steps, history}
    checkpoints every `ckpt_every` steps and, with `resume=True`, auto-picks up from
    the last one (resubmit a timed-out job and it continues). Without this a killed
    run loses everything — a 19.5k-step run died at step 16k during development, and
    cluster runs are far longer. Saves are atomic (temp + os.replace) so a SIGKILL at
    the SLURM TimeLimit can't corrupt the live checkpoint; an unreadable checkpoint
    is warned-through as a fresh start rather than crashing; and a resume whose
    total_steps disagrees with the checkpoint (i.e. a changed n/epochs, which would
    desync the LR schedule) is refused with a clear error.
    """
    torch.manual_seed(seed)
    model = model.to(device).train()
    logger(f"[trainer] true parameter count: {sum(p.numel() for p in model.parameters()):,}")
    if run_logger is not None:
        run_logger.reset_series("train")

    n = activations.shape[0]
    steps_per_epoch = max(1, n // batch_size)
    total_steps = steps_per_epoch * num_epochs
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    start_step = 0
    if ckpt_path is not None and resume and Path(ckpt_path).exists():
        try:
            state = torch.load(ckpt_path, map_location=device)
        except Exception as e:
            # A pre-atomic-write checkpoint (or any other corruption) shouldn't be a
            # hard crash — warn loudly and start fresh rather than wedge every resume.
            logger(f"[trainer] WARNING: checkpoint {ckpt_path} is unreadable "
                   f"({type(e).__name__}: {e}); starting from scratch.")
            state = None
        if state is not None:
            # Guard against silently resuming into a different LR schedule: total_steps
            # depends on n (activation count) and num_epochs, so a resume under a
            # changed config would recompute cosine_lr_with_warmup on a different total
            # and desync the schedule from the step counter. Older checkpoints predate
            # this field — skip the check for them (they carry no total_steps).
            ckpt_total = state.get("total_steps")
            if ckpt_total is not None and ckpt_total != total_steps:
                raise ValueError(
                    f"checkpoint {ckpt_path} was written for total_steps={ckpt_total} but "
                    f"this run computes total_steps={total_steps} (n={n}, batch={batch_size}, "
                    f"epochs={num_epochs}). Resuming would use a mismatched LR schedule. "
                    f"Use the original config, or pass --restart to train from scratch.")
            model.load_state_dict(state["model"])
            opt.load_state_dict(state["optimizer"])
            start_step = int(state["step"])
            logger(f"[trainer] resuming from {ckpt_path} at step {start_step}/{total_steps}")
            if run_logger is not None:      # don't duplicate the series we already logged
                run_logger.reset_series("train")
                for r in state.get("history", []):
                    run_logger.log_step("train", r)

    history = {"loss": [], "lr": []}
    step = 0
    for epoch in range(num_epochs):
        perm = torch.randperm(n, generator=gen)
        for b in range(steps_per_epoch):
            if step < start_step:               # fast-forward without recomputing
                step += 1
                continue
            idx = perm[b * batch_size:(b + 1) * batch_size]
            # index -> float32 -> device, one batch at a time (store may exceed RAM)
            z0 = activations[idx].float()
            if scaler is not None:
                z0 = scaler.standardize(z0)
            z0 = z0.to(device)

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
            if run_logger is not None:
                run_logger.log_step("train", {"step": step, "loss": loss.item(), "lr": lr * mult})
            if step % log_every_n_steps == 0:
                logger(f"[trainer] epoch {epoch} step {step}/{total_steps} "
                       f"loss {loss.item():.4f} lr {lr * mult:.2e}")
            step += 1
            if ckpt_path is not None and step % ckpt_every == 0:
                _save_checkpoint_atomic(
                    {"model": model.state_dict(), "optimizer": opt.state_dict(),
                     "step": step, "total_steps": total_steps,
                     "history": (run_logger.load_series("train") if run_logger else [])},
                    ckpt_path)
    return history
