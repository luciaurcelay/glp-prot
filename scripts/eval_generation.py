#!/usr/bin/env python
"""Generation-quality eval for Gate G1 (PLAN §6.4/§9): FD + PCA overlap.

Loads the trained GLP + scaler, samples activations from noise (t_start=1), de-
standardizes, and compares to held-out real activations via:
  - Fréchet Distance FD(generated, real) and the FD(train, val) lower-bound anchor
  - top-2 PCA overlap (+ scatter PNG) — the "non-garbage" G1 check.
%LR and SAE-recon-FD anchors are W2/G2 (PLAN §6.4).
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import torch

from glp_plm.config import load_config
from glp_plm.data.activation_store import ActivationStore
from glp_plm.diffusion import sampler
from glp_plm.diffusion.scaler import Scaler
from glp_plm.eval.frechet import frechet_distance
from glp_plm.eval.pca_overlap import pca_overlap
from glp_plm.model import GLPDenoiser
from glp_plm.utils import get_device, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--store-dir", help="ActivationStore dir (default: <output_dir>/train_store)")
    ap.add_argument("--n-samples", type=int, default=10000)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    out = Path(cfg.output_dir)
    store_dir = Path(args.store_dir) if args.store_dir else out / "train_store"

    store = ActivationStore(store_dir, d=cfg.model.activation_dim, dtype=cfg.data.activation_dtype)
    real_all = store.load().float()
    scaler = Scaler.load(out / "rep_statistics.pt")

    # held-out real split for FD; a disjoint slice as the lower-bound anchor
    n = real_all.shape[0]
    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(n, generator=g)
    k = min(args.n_samples, n // 2)
    real_eval = real_all[perm[:k]]
    real_anchor_a = real_all[perm[k:2 * k]]
    real_anchor_b = real_all[perm[:k]]  # train vs val style split for the lower bound

    # sample from noise in standardized space, then de-standardize
    model = GLPDenoiser(cfg.model).to(device).eval()
    model.load_state_dict(torch.load(out / "glp_final.pt", map_location=device))
    z1 = torch.randn(k, cfg.model.activation_dim, device=device)
    gen_std = sampler.sample(lambda z, t: model(z, t), z1,
                             num_steps=cfg.sampler.num_steps, t_start=1.0)
    gen = scaler.destandardize(gen_std.cpu())

    fd_gen = frechet_distance(gen, real_eval)
    fd_lb = frechet_distance(real_anchor_a, real_anchor_b)
    pca = pca_overlap(real_eval, gen, save_path=out / "pca_overlap.png")

    print("==== Gate G1 generation report ====")
    print(f"FD(generated, real)       = {fd_gen:.4f}")
    print(f"FD(train, val) lowerbound = {fd_lb:.4f}   (ratio {fd_gen / max(fd_lb, 1e-9):.2f}x)")
    print(f"PCA real_var_captured_frac= {pca['real_var_captured_frac']:.4f}  (top-2 PCs of real; the baseline)")
    print(f"PCA gen_var_captured_frac = {pca['gen_var_captured_frac']:.4f}  (match to baseline = well-aligned)")
    print(f"PCA pc_mean_gap_norm      = {pca['pc_mean_gap_norm']:.4f}  (0 = centered like real, in PC-std units)")
    print(f"PCA pc_std_ratio          = {pca['pc_std_ratio']:.4f}  (1.0 = matched spread)")
    print(f"PCA scatter -> {pca.get('scatter_path')}")


if __name__ == "__main__":
    main()
