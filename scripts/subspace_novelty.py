#!/usr/bin/env python
"""Subspace novelty follow-up to scripts/subspace_analysis.py: that script showed
G_perp (GLP's meta-neuron activation minus its SAE-dictionary-explained part) is
not concept-empty (median AUC 0.84 across the 94 falsification-check concepts) but
does not single out the 15 not-in-dictionary concepts from the other 79. Three
follow-ups, all pre-registered before looking at results (so the PCA step below is
not a fishing expedition over post-hoc-chosen directions):

  1. UNSUPERVISED probing of G_perp: PCA on the residual itself (not chosen by
     looking at any label), then check what its top components correlate with ---
     one of the 94 discrete concepts (AUC) or one of the 10 continuous properties
     (Spearman rho). A component correlating with neither at concept and property
     was not selected for that outcome, so it is a legitimate "uncatalogued
     structure" candidate rather than something found by scanning results.
  2. "GLP minus SAE" intersected with "GLP minus the raw residual stream": the same
     ridge-projection regression as subspace_analysis.py, refit with the SAE
     dictionary S, the raw (un-standardized) ESM activation R, and their
     concatenation [S, R] as three separate design matrices. The concatenation's
     residual is orthogonal to span(S) + span(R) = S^perp \\cap R^perp (standard
     linear algebra for a joint linear fit) --- the part of GLP that is explained
     by neither the SAE dictionary nor the layer it was trained to reconstruct, so
     not simply inherited from ESM's own representation.
  3. Where in GLP does the joint-orthogonal variance concentrate? Per-unit
     variance-fraction (var(G_perp)/var(G)) reshaped by GLP block (n_blocks x
     d_mlp) and averaged per block, checked for a monotonic trend with block depth
     (Spearman, block index vs. mean fraction) --- reuses exactly the ridge fit
     from (2), no new heavy computation.

Single noise level u=0.9 throughout (the level Part 3 of subspace_analysis.py
already used for its descriptive CKA/intrinsic-dimension check, and the level 13/15
of the not-in-dictionary winners use) rather than sweeping both u in {0.5, 0.9} ---
keeps three regressions' cost bounded and the PCA/property-correlation steps
single-representation.

Run AFTER subspace_analysis.py (reuses its probing.json/topk-verdict concept list).
"""
import argparse
import gc
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import torch

from glp_plm.config import load_config
from glp_plm.data import swissprot_labels as sl
from glp_plm.eval import baselines as bl
from glp_plm.eval import continuous_properties as cp
from glp_plm.eval import subspace as sub
from glp_plm.eval.meta_neurons import meta_neuron_bank
from glp_plm.eval.probing_1d import fast_auc_matrix
from glp_plm.experiments import RunLogger
from glp_plm.model import GLPDenoiser
from glp_plm.utils import get_device, resolve_config_data_dir, resolve_storage_dir, set_seed

# Pre-registered "uncatalogued structure" thresholds (chosen before running, not
# tuned to the result): a PC counts as unexplained if its best concept AUC is
# within 0.05 of chance AND its best property |Spearman rho| is below 0.10.
AUC_STRENGTH_THRESHOLD = 0.10   # on |auc - 0.5| * 2, i.e. AUC in [0.45, 0.55]
RHO_THRESHOLD = 0.10


def _pool_gperp_joint(A_full, spans, model, scaler, sae, ridge_joint, u, device,
                      batch_proteins=16):
    """Mean-pool G_perp_joint (GLP minus the [SAE, raw] joint projection) per
    protein, `batch_proteins` proteins' residues at a time (bounds peak memory at
    GLP's 30,720-dim width, same batching `probe_continuous.py`'s `_pooled_bank`
    uses). Returns (pooled (n_protein, width), accessions, sequences) in the same
    order as `spans`."""
    pooled, accs, seqs = [], [], []
    n = len(spans)
    for i in range(0, n, batch_proteins):
        chunk = spans.iloc[i:i + batch_proteins]
        pieces = [A_full[int(r.start):int(r.end)] for r in chunk.itertuples()]
        lens = [p.shape[0] for p in pieces]
        cat = torch.cat(pieces, dim=0).float()
        z = scaler.standardize(cat)
        G = meta_neuron_bank(model, z, u, device=device).numpy()
        S = bl.sae_bank(sae, cat).numpy()
        R = cat.numpy()
        G_perp = G - ridge_joint.predict(np.concatenate([S, R], axis=1))
        off = 0
        for L, row in zip(lens, chunk.itertuples()):
            pooled.append(G_perp[off:off + L].mean(axis=0))
            off += L
            accs.append(row.accession)
            seqs.append(row.sequence)
        del cat, z, G, S, R, G_perp
        gc.collect()
    return np.stack(pooled), accs, seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval-dir")
    ap.add_argument("--ckpt")
    ap.add_argument("--scaler")
    ap.add_argument("--probing-json")
    ap.add_argument("--topk-json")
    ap.add_argument("--u", type=float, default=0.9)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=20000)
    ap.add_argument("--n-pca-components", type=int, default=10)
    ap.add_argument("--n-proteins-pool", type=int, default=500,
                    help="protein-complete subsample for the continuous-property "
                         "correlation (same size as the family-hierarchy/"
                         "selectivity checks in discovery_hierarchy.py)")
    ap.add_argument("--protein-batch-size", type=int, default=16)
    ap.add_argument("--n-shards-per-split", type=int)
    ap.add_argument("--out-dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    out = Path(cfg.output_dir)
    out_dir = Path(args.out_dir) if args.out_dir else out
    run = RunLogger(out_dir)
    ckpt_dir = resolve_storage_dir(out, "checkpoints", "GLP_RUNS_ROOT", symlink=False)

    if args.eval_dir:
        processed = Path(args.eval_dir)
    elif cfg.data.eval_dir:
        processed = resolve_config_data_dir(cfg.data.eval_dir)
    else:
        processed = resolve_storage_dir(out, "swissprot", "GLP_DATA_ROOT", symlink=False) / "processed"

    probing_path = Path(args.probing_json) if args.probing_json else run.metrics_dir / "probing.json"
    topk_path = Path(args.topk_json) if args.topk_json else run.metrics_dir / "probing_sae_topk.json"
    probing = json.loads(probing_path.read_text())
    glp_rows = {r["concept"]: r for r in probing["results"] if r["method"] == "glp"}
    topk = json.loads(topk_path.read_text())
    verdict_of = {v["concept"]: v["verdict"] for v in topk["verdicts"]}
    concepts = sorted(verdict_of)
    print(f"[novelty] {len(concepts)} concepts, u={args.u}, "
         f"{args.n_pca_components} PCA components on the joint-orthogonal residual",
         flush=True)

    ns = args.n_shards_per_split
    A_tr_full, Y_tr, _ = sl.load_split(processed, "valid", cfg.data.layer, n_shards=ns, binarize=True)
    A_te_full, Y_te, _ = sl.load_split(processed, "test", cfg.data.layer, n_shards=ns, binarize=True)
    spans_te = sl.protein_spans(processed, "test", ns)

    from glp_plm.diffusion.scaler import Scaler
    scaler = Scaler.load(args.scaler or ckpt_dir / "rep_statistics.pt")
    model = GLPDenoiser(cfg.model).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt or ckpt_dir / "glp_final.pt", map_location=device))
    from interplm.sae.inference import load_sae_from_hf
    plm = "esm2-8m" if "8M" in cfg.data.model_name else "esm2-650m"
    sae = load_sae_from_hf(plm, cfg.data.layer)
    n_blocks, d_mlp = cfg.model.n_blocks, cfg.model.d_mlp
    width = n_blocks * d_mlp

    # ---- same residue sample as subspace_analysis.py/test_nonlinearity.py (same
    # cfg.seed + rng.choice call order), so the three R^2 summaries below are
    # comparable to Table subspace_projection.json's u=0.9 row ----
    rng = np.random.default_rng(cfg.seed)
    i_tr = rng.choice(A_tr_full.shape[0], min(args.n_train, A_tr_full.shape[0]), replace=False)
    i_te = rng.choice(A_te_full.shape[0], min(args.n_test, A_te_full.shape[0]), replace=False)
    a_tr, a_te = A_tr_full[i_tr].float(), A_te_full[i_te].float()
    z_tr, z_te = scaler.standardize(a_tr), scaler.standardize(a_te)
    G_tr = meta_neuron_bank(model, z_tr, args.u, device=device).numpy()
    G_te = meta_neuron_bank(model, z_te, args.u, device=device).numpy()
    S_tr, S_te = bl.sae_bank(sae, a_tr).numpy(), bl.sae_bank(sae, a_te).numpy()
    R_tr, R_te = a_tr.numpy(), a_te.numpy()
    X_joint_tr = np.concatenate([S_tr, R_tr], axis=1)
    X_joint_te = np.concatenate([S_te, R_te], axis=1)
    del a_tr, a_te, z_tr, z_te

    # ---- Part 2: three ridge decompositions (SAE-only, raw-only, joint) ----
    print("[novelty] fitting ridge(G ~ SAE) ...", flush=True)
    ridge_sae = sub.fit_projection(S_tr, G_tr)
    print("[novelty] fitting ridge(G ~ raw layer) ...", flush=True)
    ridge_raw = sub.fit_projection(R_tr, G_tr)
    print("[novelty] fitting ridge(G ~ [SAE, raw layer]) ...", flush=True)
    ridge_joint = sub.fit_projection(X_joint_tr, G_tr)
    del X_joint_tr

    Gpar_sae_te, Gperp_sae_te = sub.decompose(ridge_sae, S_te, G_te)
    Gpar_raw_te, Gperp_raw_te = sub.decompose(ridge_raw, R_te, G_te)
    Gpar_joint_te, Gperp_joint_te = sub.decompose(ridge_joint, X_joint_te, G_te)

    r2_sae = sub.per_unit_r2(G_te, Gpar_sae_te)
    r2_raw = sub.per_unit_r2(G_te, Gpar_raw_te)
    r2_joint = sub.per_unit_r2(G_te, Gpar_joint_te)
    decomposition_summary = {
        "u": args.u, "n_train": int(len(i_tr)), "n_test": int(len(i_te)),
        "r2_median_all_units": {
            "sae": float(np.median(r2_sae)), "raw_layer": float(np.median(r2_raw)),
            "joint": float(np.median(r2_joint)),
        },
        "note": "joint's marginal gain over sae alone is the raw layer's contribution "
               "beyond what the SAE dictionary already spans",
    }
    print(f"[novelty] median R^2 (all {width} units): "
         f"sae={decomposition_summary['r2_median_all_units']['sae']:.3f} "
         f"raw_layer={decomposition_summary['r2_median_all_units']['raw_layer']:.3f} "
         f"joint={decomposition_summary['r2_median_all_units']['joint']:.3f}", flush=True)

    # ---- Part 3: per-block variance-fraction (source of the joint-orthogonal part) ----
    frac_sae = sub.variance_fraction_orthogonal(G_te, Gperp_sae_te).reshape(n_blocks, d_mlp).mean(axis=1)
    frac_joint = sub.variance_fraction_orthogonal(G_te, Gperp_joint_te).reshape(n_blocks, d_mlp).mean(axis=1)
    from scipy.stats import spearmanr
    block_idx = np.arange(n_blocks)
    corr_sae = spearmanr(block_idx, frac_sae)
    corr_joint = spearmanr(block_idx, frac_joint)
    blocks_df = pd.DataFrame({"block": block_idx, "sae_orthogonal_frac": frac_sae,
                              "joint_orthogonal_frac": frac_joint})
    blocks_df.to_csv(run.metrics_dir / "subspace_novelty_blocks.csv", index=False)
    print(f"[novelty] block depth vs. orthogonal-variance fraction (Spearman): "
         f"sae rho={corr_sae.statistic:.3f} p={corr_sae.pvalue:.4f}; "
         f"joint rho={corr_joint.statistic:.3f} p={corr_joint.pvalue:.4f}", flush=True)

    # ---- Part 1: PCA on the joint-orthogonal residual, correlate with concepts
    # and continuous properties ----
    pca, pc_scores_te = sub.fit_pca(Gperp_joint_te, n_components=args.n_pca_components, seed=cfg.seed)
    evr = pca.explained_variance_ratio_

    Y_te_arr = Y_te[concepts].to_numpy()[i_te]                      # (n_test, 94)
    auc_matrix = fast_auc_matrix(pc_scores_te, Y_te_arr)             # (94, n_pca)
    auc_strength = np.abs(auc_matrix - 0.5) * 2                     # (94, n_pca)
    # a concept with 0 or all-N positives in this residue sample makes
    # fast_auc_matrix degenerate to exactly 0.0 (an artifact of no label
    # variance, not a real association) -- exclude it from the "best concept"
    # search so it cannot spuriously win every PC's argmax.
    n_pos_concept = Y_te_arr.sum(axis=0)
    degenerate = (n_pos_concept == 0) | (n_pos_concept == Y_te_arr.shape[0])
    if degenerate.any():
        print(f"[novelty] excluding {int(degenerate.sum())} concept(s) with no label "
             f"variance in this {len(i_te)}-residue sample from the PC correlation: "
             f"{[c for c, d in zip(concepts, degenerate) if d]}", flush=True)
    auc_strength[degenerate, :] = -1.0
    best_concept_idx = auc_strength.argmax(axis=0)
    best_concept_strength = auc_strength.max(axis=0)
    best_concept_auc = np.take_along_axis(auc_matrix, best_concept_idx[None, :], axis=0)[0]

    # protein-complete pooled sample for the continuous-property correlation
    n_pool = min(args.n_proteins_pool, len(spans_te))
    pool_idx = np.sort(np.random.default_rng(cfg.seed + 3).choice(len(spans_te), n_pool, replace=False))
    spans_sample = spans_te.iloc[pool_idx].reset_index(drop=True)
    print(f"[novelty] pooling G_perp_joint for {n_pool} test proteins "
         f"(continuous-property correlation) ...", flush=True)
    pooled, accs, seqs = _pool_gperp_joint(A_te_full, spans_sample, model, scaler, sae, ridge_joint,
                                           args.u, device, args.protein_batch_size)
    props_df = cp.properties_for_proteins(accs, seqs)
    pooled_df = pd.DataFrame(pooled, index=accs)
    common = props_df.index.intersection(pooled_df.index)
    pooled_common = pooled_df.loc[common].to_numpy()
    props_common = props_df.loc[common]
    pc_scores_pooled = pca.transform(pooled_common)                  # (n_common, n_pca)

    rho_matrix = np.zeros((len(cp.PROPERTY_NAMES), args.n_pca_components))
    for j, prop in enumerate(cp.PROPERTY_NAMES):
        for k in range(args.n_pca_components):
            rho_matrix[j, k] = spearmanr(pc_scores_pooled[:, k], props_common[prop].to_numpy()).statistic
    rho_matrix = np.nan_to_num(rho_matrix)
    best_prop_idx = np.abs(rho_matrix).argmax(axis=0)
    best_prop_rho = np.take_along_axis(rho_matrix, best_prop_idx[None, :], axis=0)[0]

    pc_rows = []
    for k in range(args.n_pca_components):
        concept_strength = float(best_concept_strength[k])
        prop_strength = float(abs(best_prop_rho[k]))
        uncatalogued = (concept_strength < AUC_STRENGTH_THRESHOLD) and (prop_strength < RHO_THRESHOLD)
        row = {
            "pc": k, "explained_variance_ratio": float(evr[k]),
            "best_concept": concepts[int(best_concept_idx[k])],
            "best_concept_auc": float(best_concept_auc[k]),
            "best_property": cp.PROPERTY_NAMES[int(best_prop_idx[k])],
            "best_property_rho": float(best_prop_rho[k]),
            "uncatalogued_candidate": bool(uncatalogued),
        }
        pc_rows.append(row)
        print(f"[novelty] PC{k}: evr={row['explained_variance_ratio']:.3f} "
             f"best_concept={row['best_concept'][:40]:<40} auc={row['best_concept_auc']:.3f} "
             f"best_property={row['best_property']:<18} rho={row['best_property_rho']:+.3f} "
             f"{'<-- UNCATALOGUED CANDIDATE' if uncatalogued else ''}", flush=True)

    (run.metrics_dir / "subspace_novelty_pca.json").write_text(json.dumps({
        "u": args.u, "n_pca_components": args.n_pca_components,
        "n_residues_for_pca": int(len(i_te)), "n_proteins_for_property_corr": int(len(common)),
        "auc_strength_threshold": AUC_STRENGTH_THRESHOLD, "rho_threshold": RHO_THRESHOLD,
        "pcs": pc_rows,
    }, indent=2))
    (run.metrics_dir / "subspace_novelty_summary.json").write_text(json.dumps({
        "decomposition": decomposition_summary,
        "block_depth_correlation": {
            "sae": {"rho": float(corr_sae.statistic), "p": float(corr_sae.pvalue)},
            "joint": {"rho": float(corr_joint.statistic), "p": float(corr_joint.pvalue)},
        },
        "n_uncatalogued_candidates": int(sum(r["uncatalogued_candidate"] for r in pc_rows)),
    }, indent=2))
    print(f"[novelty] wrote {run.metrics_dir / 'subspace_novelty_pca.json'}, "
         f"subspace_novelty_blocks.csv, subspace_novelty_summary.json", flush=True)


if __name__ == "__main__":
    main()
