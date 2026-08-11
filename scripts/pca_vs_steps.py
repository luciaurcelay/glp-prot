#!/usr/bin/env python
"""Real-vs-generated PCA projections at several sampler-step counts, for the sampler-
convergence figure (glp_plm.experiments.compare.pca_vs_steps). Saves
metrics/pca_proj_vs_steps.npz with `real` and `gen_<steps>` (all on real's top-2 PCs).

Run once for a representative trained run (e.g. the deepest/best), e.g.:
  python scripts/pca_vs_steps.py --config configs/configs_lucia/esm8m_500m_4096_d24.yaml
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from glp_plm.config import load_config
from glp_plm.data.activation_store import ActivationStore
from glp_plm.diffusion import sampler
from glp_plm.diffusion.scaler import Scaler
from glp_plm.eval.pca_overlap import pca_overlap
from glp_plm.model import GLPDenoiser
from glp_plm.utils import get_device, resolve_config_data_dir, resolve_storage_dir, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", default="1,4,20,1000", help="comma list of sampler step counts")
    ap.add_argument("--n-samples", type=int, default=20000)
    ap.add_argument("--store-dir", help="override (default: config data.store_dir / per-run)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    out = Path(cfg.output_dir)
    ckpt_dir = resolve_storage_dir(out, "checkpoints", "GLP_RUNS_ROOT", symlink=False)

    # store precedence: CLI > config data.store_dir > per-run default (same as eval_generation)
    if args.store_dir:
        store_dir = Path(args.store_dir)
    elif cfg.data.store_dir:
        store_dir = resolve_config_data_dir(cfg.data.store_dir)
    else:
        store_dir = resolve_storage_dir(out, "train_store", "GLP_DATA_ROOT", symlink=False)

    store = ActivationStore(store_dir, d=cfg.model.activation_dim, dtype=cfg.data.activation_dtype)
    scaler = Scaler.load(ckpt_dir / "rep_statistics.pt")
    mm = store.memmap()
    n = mm.shape[0]
    k = min(args.n_samples, n)
    g = torch.Generator().manual_seed(cfg.seed)
    idx = torch.randperm(n, generator=g)[:k].numpy()
    real = torch.from_numpy(np.asarray(mm[np.sort(idx)], dtype=np.float32))

    model = GLPDenoiser(cfg.model).to(device).eval()
    model.load_state_dict(torch.load(ckpt_dir / "glp_final.pt", map_location=device))

    def generate(steps: int):
        z1 = torch.randn(k, cfg.model.activation_dim, device=device,
                         generator=torch.Generator(device=device).manual_seed(cfg.seed))
        z = sampler.sample(lambda a, t: model(a, t), z1, num_steps=steps, t_start=1.0)
        return scaler.destandardize(z.cpu())

    saved = {}
    for s in [int(x) for x in args.steps.split(",") if x.strip()]:
        gen = generate(s)
        _, proj = pca_overlap(real, gen)     # both projected onto real's top-2 PCs
        saved["real"] = proj["real"]         # identical each call (real's own PCs)
        saved[f"gen_{s}"] = proj["gen"]
        print(f"[pca_vs_steps] steps={s} done", flush=True)

    metrics_dir = resolve_storage_dir(out, "metrics", "GLP_RUNS_ROOT")
    path = metrics_dir / "pca_proj_vs_steps.npz"
    np.savez_compressed(path, **saved)
    print(f"[pca_vs_steps] wrote {path}")


if __name__ == "__main__":
    main()
