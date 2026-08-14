#!/usr/bin/env python
"""Gate G2 (PLAN §6.4/§9, risk register): GLP FD < SAE-reconstruction FD anchor;
PCA converged; %Loss-Recovered competitive. **If G2 fails -> drop steering, ship
probing-only paper** (docs/PLAN.md's risk register; steering experiments so far
have run ahead of this gate, not instead of it — docs/steering_history.md).

Reuses eval_generation.py's exact FD/PCA protocol (same real_eval/real_anchor split,
same sampler settings) so `fd_generated_vs_real` here is directly comparable to a prior
G1 run, then adds the two pieces PLAN.md flags as "not yet implemented, not just unrun":
  - SAE-reconstruction FD anchor: FD(SAE(real_eval), real_anchor) via the same
    frechet_distance() used for G1, encoding/decoding real_eval through the released
    InterPLM SAE (glp_plm/eval/frechet.py's own docstring deferred this to "W2/G2").
  - %Loss-Recovered: glp_plm/eval/loss_recovered.py (adapts
    interplm/train/fidelity.py:calculate_loss_recovered), on real eval sequences pulled
    from the eval-store shards' protein_data.tsv (same eval split G1's activations and
    W3 probing already use — never training sequences).

No hardcoded PCA/%LR pass-fail threshold is invented here: PLAN.md states the FD
inequality and "PCA converged"/"%LR competitive" as the gate's three checks but never
pins numeric cutoffs for the latter two, so this script reports all three plainly (like
eval_generation.py's G1 report does for PCA) rather than fabricating a threshold — the
FD inequality is the one objectively binary check.
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from glp_plm.config import load_config
from glp_plm.data.activation_store import ActivationStore
from glp_plm.diffusion import sampler
from glp_plm.diffusion.scaler import Scaler
from glp_plm.eval.frechet import frechet_distance
from glp_plm.eval.loss_recovered import compute_pct_loss_recovered
from glp_plm.eval.pca_overlap import pca_overlap
from glp_plm.experiments import RunLogger
from glp_plm.model import GLPDenoiser
from glp_plm.utils import get_device, resolve_config_data_dir, resolve_storage_dir, set_seed


def load_eval_sequences(eval_dir: Path, split: str, n: int, seed: int) -> list:
    """Real sequences from the eval-store's held-out split (protein_data.tsv per shard,
    the same shards load_split() reads activations/labels from for probing) — never
    training sequences, matching the rest of this project's eval discipline."""
    import pandas as pd

    meta = json.loads((eval_dir / split / "metadata.json").read_text())
    shard_ids = meta["shard_source"]
    seqs = []
    for i in shard_ids:
        df = pd.read_csv(eval_dir / f"shard_{i}" / "protein_data.tsv", sep="\t",
                         usecols=["Sequence"])
        seqs.extend(df["Sequence"].dropna().tolist())
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(seqs), size=min(n, len(seqs)), replace=False)
    return [seqs[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", help="where to read/write metrics (default: config output_dir). "
                                      "Use this to point at the checkpoint's OWN run dir when the "
                                      "config's output_dir has since been repurposed (e.g. renamed "
                                      "for a follow-up experiment) — gate_g2_* fields should land "
                                      "next to that run's existing gate_g1 numbers, not a different one.")
    ap.add_argument("--ckpt", help="GLP checkpoint (default <output_dir>/checkpoints/glp_final.pt)")
    ap.add_argument("--scaler", help="rep_statistics.pt (default <output_dir>/checkpoints/rep_statistics.pt)")
    ap.add_argument("--store-dir", help="ActivationStore dir (default: <output_dir>/train_store)")
    ap.add_argument("--eval-dir", help="…/swissprot/processed. Default: config data.eval_dir")
    ap.add_argument("--eval-split", default="test", choices=["valid", "test"])
    ap.add_argument("--n-samples", type=int, default=10000, help="activations for FD (matches eval_generation.py's default)")
    ap.add_argument("--n-lr-sequences", type=int, default=200,
                    help="sequences for %%LR — each is 3 full ESM forward passes, keep modest")
    ap.add_argument("--lr-batch-size", type=int, default=8)
    ap.add_argument("--num-steps", type=int, help="sampler steps (default cfg.sampler.fd_num_steps)")
    ap.add_argument("--skip-fd-pca", action="store_true",
                    help="Don't regenerate GLP samples / recompute FD(generated,real) or PCA — "
                         "reuse fd_generated_vs_real/fd_lower_bound/fd_num_steps/pca_* already sitting "
                         "in --fd-pca-source/metrics/summary.json from a prior eval_generation.py (G1) "
                         "run. Only computes the two genuinely-missing pieces (SAE-recon FD anchor, %LR).")
    ap.add_argument("--fd-pca-source", help="where to READ the cached G1 numbers from when "
                                            "--skip-fd-pca (default: same as --out-dir/config output_dir). "
                                            "Separate from --out-dir because the run holding the original "
                                            "G1 numbers may not be one you have write access to (e.g. a "
                                            "teammate's run) — this project's convention (scripts/cluster/"
                                            "submit_steering.sh) is to read a teammate's run but write your "
                                            "own gate_g2_* results into a run you own instead of fighting ACLs.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    out = Path(args.out_dir) if args.out_dir else Path(cfg.output_dir)
    ckpt_dir = resolve_storage_dir(out, "checkpoints", "GLP_RUNS_ROOT", symlink=False)

    if args.store_dir:
        store_dir = Path(args.store_dir)
    elif cfg.data.store_dir:
        store_dir = resolve_config_data_dir(cfg.data.store_dir)
    else:
        store_dir = resolve_storage_dir(out, "train_store", "GLP_DATA_ROOT", symlink=False)

    store = ActivationStore(store_dir, d=cfg.model.activation_dim, dtype=cfg.data.activation_dtype)

    mm = store.memmap()
    n = mm.shape[0]
    k = min(args.n_samples, n // 2)
    g = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(n, generator=g)[: 2 * k].numpy()
    real_eval = torch.from_numpy(np.asarray(mm[np.sort(perm[:k])], dtype=np.float32))
    real_anchor = torch.from_numpy(np.asarray(mm[np.sort(perm[k:2 * k])], dtype=np.float32))

    if args.skip_fd_pca:
        fd_pca_source = Path(args.fd_pca_source) if args.fd_pca_source else out
        prior = json.loads((fd_pca_source / "metrics" / "summary.json").read_text())
        fd_gen = prior["fd_generated_vs_real"]
        fd_lb = prior["fd_lower_bound"]
        steps = prior["fd_num_steps"]
        pca = {k2: v for k2, v in prior.items() if k2.startswith("pca_")}
        print(f"[gate_g2] --skip-fd-pca: reusing fd_generated_vs_real={fd_gen}, "
             f"fd_lower_bound={fd_lb}, fd_num_steps={steps} from {fd_pca_source}/metrics/summary.json "
             "(not recomputed — these already exist from a prior G1 run)")
    else:
        scaler = Scaler.load(args.scaler or ckpt_dir / "rep_statistics.pt")
        model = GLPDenoiser(cfg.model).to(device).eval()
        model.load_state_dict(torch.load(args.ckpt or ckpt_dir / "glp_final.pt", map_location=device))

        steps = args.num_steps or cfg.sampler.fd_num_steps
        z1 = torch.randn(k, cfg.model.activation_dim, device=device,
                         generator=torch.Generator(device=device).manual_seed(cfg.seed))
        z = sampler.sample(lambda a, t: model(a, t), z1, num_steps=steps, t_start=1.0)
        gen = scaler.destandardize(z.cpu())

        fd_gen = frechet_distance(gen, real_eval)
        fd_lb = frechet_distance(real_anchor, real_eval)
        pca, proj = pca_overlap(real_eval, gen)

    # --- SAE-reconstruction FD anchor (the piece frechet.py's docstring deferred) ---
    from interplm.sae.inference import load_sae_from_hf
    plm = "esm2-8m" if "8M" in cfg.data.model_name else "esm2-650m"
    sae = load_sae_from_hf(plm, cfg.data.layer).to(device)
    with torch.no_grad():
        x_norm, orig_norms = sae._normalize_input_and_get_norms(real_eval.to(device))
        sae_recon = sae._unnormalize_output(sae.decode(sae.encode(x_norm)), orig_norms).cpu()
    fd_sae = frechet_distance(sae_recon, real_anchor)

    # --- %Loss-Recovered (the other piece) ---
    eval_dir = Path(args.eval_dir) if args.eval_dir else resolve_config_data_dir(cfg.data.eval_dir)
    seqs = load_eval_sequences(eval_dir, args.eval_split, args.n_lr_sequences, cfg.seed)
    lr = compute_pct_loss_recovered(sae, seqs, cfg.data.model_name, cfg.data.layer, device,
                                    batch_size=args.lr_batch_size)

    run = RunLogger(out)
    run.log_config(cfg)
    run.log_summary(
        gate_g2_fd_generated_vs_real=fd_gen, gate_g2_fd_sae_recon=fd_sae,
        gate_g2_fd_lower_bound=fd_lb, gate_g2_fd_gen_vs_sae_ratio=fd_gen / max(fd_sae, 1e-9),
        gate_g2_fd_num_steps=int(steps), gate_g2_fd_criterion_pass=bool(fd_gen < fd_sae),
        **{f"gate_g2_{k}": v for k, v in pca.items()},
        **{f"gate_g2_{k}": v for k, v in lr.items()},
    )
    if not args.skip_fd_pca:
        run.save_arrays("gate_g2_pca_proj", real=proj["real"], gen=proj["gen"])

    print("==== Gate G2 report ====")
    print(f"FD(generated, real)       = {fd_gen:.4f}")
    print(f"FD(SAE-recon, real)       = {fd_sae:.4f}   <- anchor: GLP should beat this")
    print(f"FD(train, val) lowerbound = {fd_lb:.4f}")
    print(f"  criterion 1 (GLP FD < SAE-recon FD): {'PASS' if fd_gen < fd_sae else 'FAIL'} "
         f"(ratio {fd_gen / max(fd_sae, 1e-9):.3f}x)")
    print(f"PCA real_var_captured_frac= {pca['pca_real_var_captured_frac']:.4f}")
    print(f"PCA gen_var_captured_frac = {pca['pca_gen_var_captured_frac']:.4f}  "
         "(criterion 2, 'PCA converged' -- no fixed cutoff in PLAN.md; judge vs. G1's own numbers)")
    print(f"PCA pc_mean_gap_norm      = {pca['pca_pc_mean_gap_norm']:.4f}")
    print(f"PCA pc_std_ratio          = {pca['pca_pc_std_ratio']:.4f}")
    print(f"%Loss-Recovered (SAE)     = {lr['pct_loss_recovered']:.2f}%  "
         f"(CE identity={lr['ce_identity']:.4f}, zero-abl={lr['ce_zero_ablation']:.4f}, "
         f"sae-recon={lr['ce_sae_recon']:.4f}, n={lr['n_sequences']} seqs)  "
         "(criterion 3, 'competitive with the SAE' -- this IS the SAE's own %LR, i.e. a "
         "substrate-quality check, not a GLP-vs-SAE comparison)")
    print("========================")
    print("GO/NO-GO is a judgment call on all three together per PLAN.md — this script "
         "computes the numbers, it does not auto-decide.")


if __name__ == "__main__":
    main()
