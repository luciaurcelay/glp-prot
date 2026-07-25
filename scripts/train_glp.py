#!/usr/bin/env python
"""Train GLP on a standardized activation store (PLAN §11).

Loads a (N, d) activation tensor (.pt) referenced by the config, standardizes
with a fitted Scaler, trains the GLPDenoiser via flow matching, saves ckpt + scaler.
The activation store itself is produced by scripts/extract_activations.py (W1).
"""
import argparse
from pathlib import Path

import torch

from glp_plm.config import load_config
from glp_plm.diffusion.scaler import Scaler
from glp_plm.model import GLPDenoiser
from glp_plm.train.trainer import fit
from glp_plm.utils import get_device, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--activations", help="path to a (N,d) .pt activation store")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)

    if not args.activations:
        raise SystemExit(
            "Provide --activations <store.pt>. Build it first with "
            "scripts/extract_activations.py (W1; not run this session)."
        )
    raw = torch.load(args.activations, map_location="cpu").float()
    scaler = Scaler.fit(raw); scaler.save(out / "rep_statistics.pt")
    z0 = scaler.standardize(raw)

    model = GLPDenoiser(cfg.model)
    fit(model, z0, lr=cfg.train.lr, batch_size=cfg.train.batch_size,
        num_epochs=cfg.train.num_epochs, warmup_ratio=cfg.train.warmup_ratio,
        lr_initial_factor=cfg.train.lr_initial_factor,
        lr_final_factor=cfg.train.lr_final_factor, grad_clip=cfg.train.grad_clip,
        device=device, log_every_n_steps=cfg.train.log_every_n_steps, seed=cfg.seed)
    torch.save(model.state_dict(), out / "glp_final.pt")
    print(f"[train_glp] saved to {out}")


if __name__ == "__main__":
    main()
