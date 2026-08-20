#!/usr/bin/env python
"""Subspace analysis — answers the open question left by the Discovery section:
what distinguishes the 15 concepts the joint SAE top-k probe cannot recover
(scripts/probe_sae_topk.py) from the 79 it does? Polysemanticity was ruled out
(glp_plm/eval/discovery_selectivity.py; Mann-Whitney p=0.70/0.30). This tests a
geometric hypothesis: decompose GLP's meta-neuron activations G against the
linear span of the SAE dictionary S (ridge projection, same regularization as
glp_plm.eval.falsification's R^2 check), then probe the orthogonal component
G_perp for all 94 concepts. If the 15 are recoverable from G_perp and the 79 are
not, the SAE-missing information lives in an identifiable subspace.

Not the residual probe already run (x - sae.decode(sae.encode(x)), the part of
the ESM ACTIVATION the SAE fails to reconstruct). This decomposes GLP's OWN
meta-neuron activations against the SAE's span — a different object.

Part 1 fits the projection (RidgeCV, alpha_per_target) on a random residue
sample using the EXACT same sampling as test_nonlinearity.py (same cfg.seed,
same rng.choice call order) so the per-unit R^2 sanity check is comparable to
falsification.json's numbers, not just similarly-distributed.
Part 2 probes G_perp/G_parallel per concept (same balanced_indices protocol as
run_probing.py) and reports the Mann-Whitney effect size between the two groups.
Part 3 is descriptive only (intrinsic dimension, linear CKA) — supporting, not
evidence for the main hypothesis.

Run AFTER run_probing.py, test_nonlinearity.py, and probe_sae_topk.py.
"""
import argparse
import gc
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from glp_plm.config import load_config
from glp_plm.data import swissprot_labels as sl
from glp_plm.eval import baselines as bl
from glp_plm.eval import probing_1d as pr
from glp_plm.eval import subspace as sub
from glp_plm.eval.meta_neurons import meta_neuron_bank
from glp_plm.experiments import RunLogger
from glp_plm.model import GLPDenoiser
from glp_plm.utils import get_device, resolve_config_data_dir, resolve_storage_dir, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval-dir")
    ap.add_argument("--ckpt")
    ap.add_argument("--scaler")
    ap.add_argument("--probing-json")
    ap.add_argument("--falsification-json")
    ap.add_argument("--topk-json", help="probe_sae_topk.py's probing_sae_topk.json "
                                        "(defines the 15/79 grouping)")
    ap.add_argument("--u-values", default="0.5,0.9",
                    help="noise levels to decompose+probe (default: the two that "
                         "actually won any of the 94 concepts)")
    ap.add_argument("--n-train", type=int, default=20000,
                    help="residues to fit the ridge projection on (matches "
                         "test_nonlinearity.py's default so the sanity check is comparable)")
    ap.add_argument("--n-test", type=int, default=20000)
    ap.add_argument("--max-pos", type=int, default=3000, help="match run_probing.py")
    ap.add_argument("--neg-per-pos", type=float, default=1.0)
    ap.add_argument("--id-cka-sample-size", type=int, default=5000)
    ap.add_argument("--n-shards-per-split", type=int)
    ap.add_argument("--out-dir")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--skip-descriptive", action="store_true",
                    help="skip Part 3 (intrinsic dimension / CKA)")
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
    fals_path = Path(args.falsification_json) if args.falsification_json else run.metrics_dir / "falsification.json"

    probing = json.loads(probing_path.read_text())
    glp_rows = {r["concept"]: r for r in probing["results"] if r["method"] == "glp"}
    topk = json.loads(topk_path.read_text())
    verdict_of = {v["concept"]: v["verdict"] for v in topk["verdicts"]}
    fals_r2 = {r["concept"]: r["r2"] for r in json.loads(fals_path.read_text())["results"]}
    concepts = sorted(verdict_of)
    u_values = sorted(float(x) for x in args.u_values.split(","))
    print(f"[subspace] {len(concepts)} concepts "
         f"({sum(1 for v in verdict_of.values() if v == 'not_in_dictionary')} not_in_dictionary, "
         f"{sum(1 for v in verdict_of.values() if v == 'distributed_in_dictionary')} distributed), "
         f"u in {u_values}", flush=True)
    actual_us = sorted({glp_rows[c]["u"] for c in concepts})
    if set(actual_us) - set(u_values):
        print(f"[subspace] WARNING: winners also use u={sorted(set(actual_us) - set(u_values))}, "
             f"not in --u-values — those concepts will be mis-decoded downstream", flush=True)

    ns = args.n_shards_per_split
    A_tr_full, Y_tr, names = sl.load_split(processed, "valid", cfg.data.layer, n_shards=ns, binarize=True)
    A_te_full, Y_te, _ = sl.load_split(processed, "test", cfg.data.layer, n_shards=ns, binarize=True)
    protein_id_tr = sl.residue_protein_ids(processed, "valid", ns)
    protein_id_te = sl.residue_protein_ids(processed, "test", ns)

    from glp_plm.diffusion.scaler import Scaler
    scaler = Scaler.load(args.scaler or ckpt_dir / "rep_statistics.pt")
    model = GLPDenoiser(cfg.model).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt or ckpt_dir / "glp_final.pt", map_location=device))
    from interplm.sae.inference import load_sae_from_hf
    plm = "esm2-8m" if "8M" in cfg.data.model_name else "esm2-650m"
    sae = load_sae_from_hf(plm, cfg.data.layer)
    width = cfg.model.n_blocks * cfg.model.d_mlp

    # ---- Part 1: fit G ~ S ridge projection per u, on the SAME random residue
    # sample test_nonlinearity.py uses (identical cfg.seed + rng.choice call order) ----
    rng = np.random.default_rng(cfg.seed)
    i_tr = rng.choice(A_tr_full.shape[0], min(args.n_train, A_tr_full.shape[0]), replace=False)
    i_te = rng.choice(A_te_full.shape[0], min(args.n_test, A_te_full.shape[0]), replace=False)
    a_tr, a_te = A_tr_full[i_tr].float(), A_te_full[i_te].float()
    S_tr = bl.sae_bank(sae, a_tr).numpy()
    S_te = bl.sae_bank(sae, a_te).numpy()

    ridges, proj_summary = {}, {}
    for u in u_values:
        print(f"[subspace] fitting projection at u={u} ({len(i_tr)} train / {len(i_te)} test residues)...",
             flush=True)
        z_tr, z_te = scaler.standardize(a_tr), scaler.standardize(a_te)
        G_tr = meta_neuron_bank(model, z_tr, u, device=device).numpy()
        G_te = meta_neuron_bank(model, z_te, u, device=device).numpy()
        ridge = sub.fit_projection(S_tr, G_tr)
        ridges[u] = ridge
        G_par_te, _ = sub.decompose(ridge, S_te, G_te)
        r2_all = sub.per_unit_r2(G_te, G_par_te)

        checks = [{"concept": c, "r2_here": float(r2_all[glp_rows[c]["feature_index"]]),
                  "r2_falsification": fals_r2[c]}
                 for c in concepts if glp_rows[c]["u"] == u and c in fals_r2]
        proj_summary[str(u)] = {
            "n_train": int(len(i_tr)), "n_test": int(len(i_te)),
            "r2_median_all_units": float(np.median(r2_all)), "r2_mean_all_units": float(np.mean(r2_all)),
            "sanity_check_n": len(checks),
            "sanity_check_r2_median_here": float(np.median([c["r2_here"] for c in checks])) if checks else None,
            "sanity_check_r2_median_falsification": float(np.median([c["r2_falsification"] for c in checks])) if checks else None,
            "sanity_check_rows": checks,
        }
        print(f"[subspace] u={u}: median R^2 all {width} units = {np.median(r2_all):.3f}; "
             f"sanity check ({len(checks)} winners) here={proj_summary[str(u)]['sanity_check_r2_median_here']} "
             f"vs falsification.json={proj_summary[str(u)]['sanity_check_r2_median_falsification']}", flush=True)
        del G_tr, G_te, G_par_te; gc.collect()

    (run.metrics_dir / "subspace_projection.json").write_text(json.dumps({
        "alphas": list(np.logspace(-2, 4, 13)), "u_values": u_values, "by_u": proj_summary,
    }, indent=2))

    # ---- Part 2: probe G_perp / G_parallel per concept ----
    partial = run.metrics_dir / "subspace_probing_partial.jsonl"
    results = []
    if partial.exists() and not args.restart:
        results = [json.loads(l) for l in partial.read_text().splitlines() if l.strip()]
        print(f"[subspace] resuming: {len(results)} concepts already done")
    done = {r["concept"] for r in results}

    for ci, concept in enumerate(concepts, 1):
        if concept in done:
            continue
        y_tr_full = Y_tr[concept].to_numpy().astype(int)
        y_te_full = Y_te[concept].to_numpy().astype(int)
        i_tr_c = pr.balanced_indices(y_tr_full, args.max_pos, args.neg_per_pos, seed=cfg.seed,
                                     protein_id=protein_id_tr)
        i_te_c = pr.balanced_indices(y_te_full, args.max_pos, args.neg_per_pos, seed=cfg.seed + 1,
                                     protein_id=protein_id_te)
        y_tr_c, y_te_c = y_tr_full[i_tr_c], y_te_full[i_te_c]
        a_tr_c, a_te_c = A_tr_full[i_tr_c].float(), A_te_full[i_te_c].float()
        z_tr_c, z_te_c = scaler.standardize(a_tr_c), scaler.standardize(a_te_c)
        S_tr_c = bl.sae_bank(sae, a_tr_c).numpy()
        S_te_c = bl.sae_bank(sae, a_te_c).numpy()

        perp_tr, perp_te, par_tr, par_te = [], [], [], []
        for u in u_values:
            G_tr_u = meta_neuron_bank(model, z_tr_c, u, device=device).numpy()
            G_te_u = meta_neuron_bank(model, z_te_c, u, device=device).numpy()
            gp_tr, gr_tr = sub.decompose(ridges[u], S_tr_c, G_tr_u)
            gp_te, gr_te = sub.decompose(ridges[u], S_te_c, G_te_u)
            par_tr.append(gp_tr); perp_tr.append(gr_tr)
            par_te.append(gp_te); perp_te.append(gr_te)
        Gperp_tr, Gperp_te = np.concatenate(perp_tr, 1), np.concatenate(perp_te, 1)
        Gpar_tr, Gpar_te = np.concatenate(par_tr, 1), np.concatenate(par_te, 1)

        res_perp = pr.probe_concept(Gperp_tr, y_tr_c, Gperp_te, y_te_c, concept, "glp_perp",
                                    topk=cfg.probing.topk, seed=cfg.probing.seed)
        res_par = pr.probe_concept(Gpar_tr, y_tr_c, Gpar_te, y_te_c, concept, "glp_parallel",
                                   topk=cfg.probing.topk, seed=cfg.probing.seed)
        sel_u = u_values[res_perp.feature_index // width]
        sel_local = res_perp.feature_index % width

        row = sub.SubspaceProbeRow(
            concept=concept, group=verdict_of[concept],
            auc_glp_full=glp_rows[concept]["test_auc"],
            auc_glp_parallel=res_par.test_auc, auc_glp_perp=res_perp.test_auc,
            selected_unit_perp=sel_local, noise_level_u=sel_u,
        )
        results.append(row.to_dict())
        with open(partial, "a") as f:
            f.write(json.dumps(row.to_dict()) + "\n")
        print(f"[subspace] ({ci}/{len(concepts)}) {concept[:40]:<40} group={verdict_of[concept]:<22} "
             f"full={row.auc_glp_full:.3f} perp={row.auc_glp_perp:.3f} par={row.auc_glp_parallel:.3f}",
             flush=True)
        del Gperp_tr, Gperp_te, Gpar_tr, Gpar_te; gc.collect()

    import csv
    with open(run.metrics_dir / "subspace_probing.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["concept", "group", "auc_glp_full", "auc_glp_parallel",
                                          "auc_glp_perp", "selected_unit_perp", "noise_level_u"])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"[subspace] wrote {run.metrics_dir / 'subspace_probing.csv'} ({len(results)} concepts)")

    not_in_perp = [r["auc_glp_perp"] for r in results if r["group"] == "not_in_dictionary"]
    dist_perp = [r["auc_glp_perp"] for r in results if r["group"] == "distributed_in_dictionary"]
    not_in_par = [r["auc_glp_parallel"] for r in results if r["group"] == "not_in_dictionary"]
    dist_par = [r["auc_glp_parallel"] for r in results if r["group"] == "distributed_in_dictionary"]
    stats = {
        "auc_glp_perp": sub.mann_whitney_effect(not_in_perp, dist_perp),
        "auc_glp_parallel": sub.mann_whitney_effect(not_in_par, dist_par),
    }
    (run.metrics_dir / "subspace_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[subspace] G_perp:     not_in median={stats['auc_glp_perp']['median_not_in_dictionary']:.3f} "
         f"distributed median={stats['auc_glp_perp']['median_distributed']:.3f} "
         f"p={stats['auc_glp_perp']['p_value']:.4f} effect={stats['auc_glp_perp']['rank_biserial_effect_size']:.3f}")
    print(f"[subspace] G_parallel: not_in median={stats['auc_glp_parallel']['median_not_in_dictionary']:.3f} "
         f"distributed median={stats['auc_glp_parallel']['median_distributed']:.3f} "
         f"p={stats['auc_glp_parallel']['p_value']:.4f} effect={stats['auc_glp_parallel']['rank_biserial_effect_size']:.3f}")

    if args.skip_descriptive:
        return

    # ---- Part 3: descriptive intrinsic dimension + linear CKA ----
    n_desc = min(args.id_cka_sample_size, A_te_full.shape[0])
    idx_desc = np.random.default_rng(cfg.seed + 2).choice(A_te_full.shape[0], n_desc, replace=False)
    a_desc = A_te_full[idx_desc].float()
    z_desc = scaler.standardize(a_desc)
    u_desc = max(u_values)
    G_desc = meta_neuron_bank(model, z_desc, u_desc, device=device).numpy()
    S_desc = bl.sae_bank(sae, a_desc).numpy()
    R_desc = a_desc.numpy()
    tag = f"s{ns}" if ns else "all"
    mlp_bank = bl.raw_mlp_cached_bank(processed, sl.split_shards(processed, "test", ns), cfg.data.layer,
                                      cfg.data.model_name, device, processed / "cache" / f"raw_mlp_test_{tag}.npy")
    M_desc = np.asarray(mlp_bank[idx_desc], dtype=np.float32)

    reps = {"glp": G_desc, "sae": S_desc, "raw_mlp": M_desc, "raw_layer": R_desc}
    ids = {name: sub.twonn_intrinsic_dimension(X, seed=cfg.seed) for name, X in reps.items()}
    cka = {
        "glp_sae": sub.linear_cka(G_desc, S_desc),
        "glp_raw": sub.linear_cka(G_desc, R_desc),
        "sae_raw": sub.linear_cka(S_desc, R_desc),
    }
    print(f"[subspace] intrinsic dimension: {ids}")
    print(f"[subspace] linear CKA: {cka}")
    (run.metrics_dir / "representation_similarity.json").write_text(json.dumps({
        "sample_size": n_desc, "u_used_for_glp": u_desc,
        "preprocessing": "raw (unstandardized) per-residue activations, no additional centering "
                         "beyond CKA's own Gram-matrix centering",
        "intrinsic_dimension": ids, "cka": cka,
    }, indent=2))
    print(f"[subspace] wrote {run.metrics_dir / 'representation_similarity.json'}")


if __name__ == "__main__":
    main()
