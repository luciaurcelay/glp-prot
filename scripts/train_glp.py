#!/usr/bin/env python
"""Train GLP on the standardized activation store (PLAN §11).

Reads the memmap store built by scripts/extract_activations.py (default
cfg.output_dir/train_store), standardizes with its fitted Scaler, trains the
GLPDenoiser via flow matching, saves ckpt + scaler. `--activations <.pt>` is also
accepted as a raw-tensor fallback.
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from glp_plm.config import load_config
from glp_plm.data.activation_store import ActivationStore
from glp_plm.diffusion.scaler import Scaler
from glp_plm.experiments import RunLogger, make_report
from glp_plm.model import GLPDenoiser
from glp_plm.train.trainer import fit
from glp_plm.utils import get_device, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--store-dir", help="ActivationStore dir (default: <output_dir>/train_store)")
    ap.add_argument("--activations", help="raw (N,d) .pt fallback instead of a store dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)

    if args.activations:
        raw = torch.load(args.activations, map_location="cpu").float()
        scaler = Scaler.fit(raw); scaler.save(out / "rep_statistics.pt")
        z0 = scaler.standardize(raw)
    else:
        store_dir = Path(args.store_dir) if args.store_dir else out / "train_store"
        store = ActivationStore(store_dir, d=cfg.model.activation_dim, dtype=cfg.data.activation_dtype)
        raw = store.load().float()
        try:
            scaler = store.load_scaler()
        except FileNotFoundError:
            scaler = store.fit_scaler()
        scaler.save(out / "rep_statistics.pt")
        z0 = scaler.standardize(raw)
    print(f"[train_glp] training on {z0.shape[0]:,} activations (d={z0.shape[1]}) on {device}")

    run_logger = RunLogger(out)
    model = GLPDenoiser(cfg.model)
    fit(model, z0, lr=cfg.train.lr, batch_size=cfg.train.batch_size,
        num_epochs=cfg.train.num_epochs, warmup_ratio=cfg.train.warmup_ratio,
        lr_initial_factor=cfg.train.lr_initial_factor,
        lr_final_factor=cfg.train.lr_final_factor, grad_clip=cfg.train.grad_clip,
        device=device, log_every_n_steps=cfg.train.log_every_n_steps, seed=cfg.seed,
        run_logger=run_logger)
    torch.save(model.state_dict(), out / "glp_final.pt")
    run_logger.log_summary(final_loss=run_logger.load_series("train")[-1]["loss"],
                           n_params=model.num_parameters(), n_blocks=cfg.model.n_blocks)
    make_report(out)
    print(f"[train_glp] saved checkpoint to {out / 'glp_final.pt'}")


if __name__ == "__main__":
    main()
