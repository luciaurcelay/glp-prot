#!/usr/bin/env python
"""Discovery §3 — selectivity (polysemanticity). Builds the full winning-unit x
concept AUC matrix `probing_1d.probe_concept` never keeps (only each concept's
winner survives run_probing.py's output), then computes each winning unit's
selectivity ratio (its own concept's AUC / its second-best AUC elsewhere) and
concept count (# concepts where it exceeds AUC 0.8) — for GLP, SAE, raw_mlp (FFN),
and raw_layer alike, per Discovery spec Part 3 ("the comparison is the result").

Per concept, builds one bank per method (same balanced_indices subsample every
other probe in this codebase uses) and evaluates every OTHER concept's winning
unit against it — mirrors `run_probing.py`'s `bank()` closure, generalized from
"the topk-filtered candidates" to "the fixed list of units that won something,
somewhere." Reuses `probing_1d._fit_one` (the same single-feature
StandardScaler+LogisticRegressionCV fit every probe in this repo uses) under
`joblib.Parallel`, the same parallelization `probe_concept` already relies on for
its 512-candidate sweep — here the sweep is the ~300 winning units instead.

Run AFTER run_probing.py (--metric auc) — needs its probing.json for the winner list.
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
from glp_plm.eval import discovery_selectivity as ds
from glp_plm.eval import probing_1d as pr
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
    ap.add_argument("--max-pos", type=int, default=3000)
    ap.add_argument("--neg-per-pos", type=float, default=1.0)
    ap.add_argument("--n-shards-per-split", type=int)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out-dir")
    ap.add_argument("--restart", action="store_true")
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
    probing = json.loads(probing_path.read_text())

    # ---- winner list per method: concept -> (u_or_None, local_col) ----
    winners = {m: {} for m in ("glp", "sae", "raw_layer", "raw_mlp")}
    for r in probing["results"]:
        if r["method"] in winners:
            winners[r["method"]][r["concept"]] = (r.get("u"), r["feature_index"])
    # unique (method, unit) pairs actually needed, keyed by an id string for output
    unit_list = {m: sorted({col for _, col in winners[m].values()}) for m in winners}
    # GLP units carry a `u` too — a (u, local_col) pair IS the unit identity
    glp_units = sorted({(u, col) for u, col in winners["glp"].values()})
    for m in winners:
        n = len(glp_units) if m == "glp" else len(unit_list[m])
        print(f"[discovery_selectivity] {m}: {n} unique winning unit(s)", flush=True)

    ns = args.n_shards_per_split
    binz = True
    A_tr, Y_tr, names = sl.load_split(processed, "valid", cfg.data.layer, n_shards=ns, binarize=binz)
    A_te, Y_te, _ = sl.load_split(processed, "test", cfg.data.layer, n_shards=ns, binarize=binz)
    A_tr, A_te = A_tr.half(), A_te.half()
    gc.collect()
    protein_id_tr = sl.residue_protein_ids(processed, "valid", ns)
    protein_id_te = sl.residue_protein_ids(processed, "test", ns)

    from glp_plm.diffusion.scaler import Scaler
    scaler = Scaler.load(args.scaler or ckpt_dir / "rep_statistics.pt")
    model = GLPDenoiser(cfg.model).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt or ckpt_dir / "glp_final.pt", map_location=device))
    from interplm.sae.inference import load_sae_from_hf
    plm = "esm2-8m" if "8M" in cfg.data.model_name else "esm2-650m"
    sae = load_sae_from_hf(plm, cfg.data.layer)
    tag = f"s{ns}" if ns else "all"
    mlp_tr = mlp_te = None
    for split, holder in (("valid", "tr"), ("test", "te")):
        shards = sl.split_shards(processed, split, ns)
        bank = bl.raw_mlp_cached_bank(processed, shards, cfg.data.layer, cfg.data.model_name,
                                      device, processed / "cache" / f"raw_mlp_{split}_{tag}.npy")
        if holder == "tr":
            mlp_tr = bank
        else:
            mlp_te = bank

    all_concepts = sorted(set().union(*[winners[m].keys() for m in winners]))
    print(f"[discovery_selectivity] {len(all_concepts)} concepts x "
         f"{sum(len(glp_units) if m == 'glp' else len(unit_list[m]) for m in winners)} unit(s)",
         flush=True)

    out_csv = run.metrics_dir / "discovery_selectivity.csv"
    partial = run.metrics_dir / "discovery_selectivity_partial.jsonl"
    matrix = {}   # (method, unit_key) -> {concept: test_auc}
    if partial.exists() and not args.restart:
        for line in partial.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            matrix.setdefault((d["method"], d["unit_key"]), {})[d["concept"]] = d["test_auc"]
        n_done = sum(len(v) for v in matrix.values())
        print(f"[discovery_selectivity] resuming: {n_done} (unit, concept) probes already done")

    for ci, concept in enumerate(all_concepts, 1):
        y_tr_full = Y_tr[concept].to_numpy().astype(int)
        y_te_full = Y_te[concept].to_numpy().astype(int)
        i_tr = pr.balanced_indices(y_tr_full, args.max_pos, args.neg_per_pos, seed=cfg.seed,
                                   protein_id=protein_id_tr)
        i_te = pr.balanced_indices(y_te_full, args.max_pos, args.neg_per_pos, seed=cfg.seed + 1,
                                   protein_id=protein_id_te)
        y_tr, y_te = y_tr_full[i_tr], y_te_full[i_te]
        a_tr, a_te = A_tr[i_tr].float(), A_te[i_te].float()

        jobs = []   # (method, unit_key, x_tr, x_te)
        # GLP: one meta-neuron forward pass per distinct u actually used by a winner
        needed_us = sorted({u for u, _ in glp_units})
        z_tr, z_te = scaler.standardize(a_tr), scaler.standardize(a_te)
        for u in needed_us:
            btr = meta_neuron_bank(model, z_tr, u, device=device).numpy()
            bte = meta_neuron_bank(model, z_te, u, device=device).numpy()
            for uu, col in glp_units:
                if uu != u:
                    continue
                key = f"u{uu}:{col}"
                if concept in matrix.get(("glp", key), {}):
                    continue
                jobs.append(("glp", key, btr[:, col], bte[:, col]))
        del z_tr, z_te

        if unit_list["sae"]:
            Ftr = bl.sae_bank(sae, a_tr).numpy()
            Fte = bl.sae_bank(sae, a_te).numpy()
            for col in unit_list["sae"]:
                key = str(col)
                if concept in matrix.get(("sae", key), {}):
                    continue
                jobs.append(("sae", key, Ftr[:, col], Fte[:, col]))
            del Ftr, Fte

        if unit_list["raw_layer"]:
            for col in unit_list["raw_layer"]:
                key = str(col)
                if concept in matrix.get(("raw_layer", key), {}):
                    continue
                jobs.append(("raw_layer", key, a_tr[:, col].numpy(), a_te[:, col].numpy()))

        if unit_list["raw_mlp"]:
            Mtr = np.asarray(mlp_tr[i_tr], dtype=np.float32)
            Mte = np.asarray(mlp_te[i_te], dtype=np.float32)
            for col in unit_list["raw_mlp"]:
                key = str(col)
                if concept in matrix.get(("raw_mlp", key), {}):
                    continue
                jobs.append(("raw_mlp", key, Mtr[:, col], Mte[:, col]))
            del Mtr, Mte

        if jobs:
            from joblib import Parallel, delayed
            scored = Parallel(n_jobs=args.n_jobs, max_nbytes=None)(
                delayed(pr._fit_one)(np.ascontiguousarray(x_tr), y_tr,
                                     np.ascontiguousarray(x_te), y_te, cfg.probing.seed)
                for _, _, x_tr, x_te in jobs
            )
            with open(partial, "a") as f:
                for (method, key, _, _), (val_auc, test_auc) in zip(jobs, scored):
                    matrix.setdefault((method, key), {})[concept] = test_auc
                    f.write(json.dumps({"method": method, "unit_key": key, "concept": concept,
                                        "val_auc": val_auc, "test_auc": test_auc}) + "\n")
        gc.collect()
        print(f"[discovery_selectivity] ({ci}/{len(all_concepts)}) {concept[:44]:<44} "
             f"{len(jobs)} new fit(s)", flush=True)

    # ---- selectivity metrics: only for units evaluated against ALL concepts they need ----
    rows = []
    for method in ("glp", "sae", "raw_layer", "raw_mlp"):
        for concept, (u, col) in winners[method].items():
            key = f"u{u}:{col}" if method == "glp" else str(col)
            auc_by_concept = matrix.get((method, key), {})
            if concept not in auc_by_concept or len(auc_by_concept) < len(all_concepts):
                continue
            unit_id = col
            row = ds.selectivity_rows(method, concept, unit_id, auc_by_concept)
            rows.append(row)

    import csv
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["representation", "unit", "winning_concept", "winning_auc",
                                          "second_best_auc", "second_best_concept",
                                          "selectivity_ratio", "n_concepts_above_0_8"])
        w.writeheader()
        for r in rows:
            w.writerow(r.to_dict())
    print(f"[discovery_selectivity] wrote {out_csv} ({len(rows)} units)")

    summary = ds.summarize(rows)
    summary_path = run.metrics_dir / "discovery_selectivity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    for rep, s in summary.items():
        print(f"[discovery_selectivity] {rep}: median ratio={s['selectivity_ratio_median']:.2f} "
             f"median n_concepts>0.8={s['n_concepts_above_0.8_median']:.1f} "
             f"frac polysemantic={s['frac_polysemantic_gt1']:.2f}")
    print(f"[discovery_selectivity] wrote {summary_path}")


if __name__ == "__main__":
    main()
