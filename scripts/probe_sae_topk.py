#!/usr/bin/env python
"""SAE top-k probing — the R^2-insufficiency follow-up (probing spec, after
test_nonlinearity.py). `falsification.py`'s R^2 asks whether a WINNING meta-neuron's
activation is reconstructible from the full SAE dictionary — a unit-level question.
Low R^2 doesn't settle whether the CONCEPT itself is in the dictionary: the slice of
the meta-neuron that correlates with the concept could still lie in the SAE's span
while the R^2 is dragged down by the rest of the unit's variance.

For each concept with R^2 < --r2-threshold (default 0.55, ~6 concepts expected), this
runs the SAME 1-D probing protocol (`probing_1d.balanced_indices` subsampling,
identical seeds/splits to run_probing.py) but fits a JOINT logistic regression over the
SAE's own top-k candidate features (k = 2, 5, 10, 20) instead of the single best 1-D
candidate:
  - if SAE-topk test AUC reaches GLP's test AUC at k<=10 -> the concept IS in the
    dictionary, spread across features ("GLP concentrates what the SAE distributes").
  - if it doesn't reach even at k=20 -> the dictionary doesn't capture the concept,
    jointly or separately — a genuine discovery result, on named/annotated concepts.

Run AFTER run_probing.py (`--metric auc`) and test_nonlinearity.py (needs both
probing.json for the GLP/SAE-k1 baselines and falsification.json for the R^2 filter).
"""
import argparse
import gc
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from glp_plm.config import load_config
from glp_plm.data import swissprot_labels as sl
from glp_plm.eval import baselines as bl
from glp_plm.eval import probing_1d as pr
from glp_plm.experiments import RunLogger
from glp_plm.utils import resolve_config_data_dir, resolve_storage_dir, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval-dir", help="…/swissprot/processed. Default: config data.eval_dir, "
                                        "else <output_dir>/swissprot/processed")
    ap.add_argument("--probing-json", help="run_probing.py's output (default "
                                           "<output_dir>_probing/metrics/probing.json — its "
                                           "sibling-dir convention, see run_probing.py --out-dir)")
    ap.add_argument("--falsification-json", help="test_nonlinearity.py's output (default "
                                                  "<output_dir>/metrics/falsification.json)")
    ap.add_argument("--r2-threshold", type=float, default=0.55,
                    help="probe every falsification-json concept with R^2 below this")
    ap.add_argument("--concepts", help="comma-separated concept names; overrides the "
                                       "R^2 filter entirely")
    ap.add_argument("--ks", default="2,5,10,20")
    ap.add_argument("--max-pos", type=int, default=3000, help="match run_probing.py's default")
    ap.add_argument("--neg-per-pos", type=float, default=1.0, help="match run_probing.py's default")
    ap.add_argument("--n-shards-per-split", type=int)
    ap.add_argument("--out-dir", help="default <output_dir>_probing — the same sibling dir "
                                      "run_probing.py writes into")
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    out = Path(cfg.output_dir)
    if args.eval_dir:
        processed = Path(args.eval_dir)
    elif cfg.data.eval_dir:
        processed = resolve_config_data_dir(cfg.data.eval_dir)
    else:
        processed = resolve_storage_dir(out, "swissprot", "GLP_DATA_ROOT", symlink=False) / "processed"
    ns = args.n_shards_per_split
    ks = [int(k) for k in args.ks.split(",") if k.strip()]

    # ---- baselines: GLP + SAE(k=1) test AUC per concept, from run_probing.py ----
    probing_path = Path(args.probing_json) if args.probing_json else out.parent / f"{out.name}_probing" / "metrics" / "probing.json"
    probing = json.loads(probing_path.read_text())
    glp_auc = {r["concept"]: r["test_auc"] for r in probing["results"] if r["method"] == "glp"}
    sae_k1_auc = {r["concept"]: r["test_auc"] for r in probing["results"] if r["method"] == "sae"}

    # ---- concept list: explicit override, else every low-R^2 concept from test_nonlinearity.py ----
    if args.concepts:
        concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]
    else:
        fals_path = Path(args.falsification_json) if args.falsification_json else out / "metrics" / "falsification.json"
        fals = json.loads(fals_path.read_text())
        concepts = [r["concept"] for r in fals["results"] if r["r2"] < args.r2_threshold]
    concepts = [c for c in concepts if c in glp_auc and c in sae_k1_auc]
    print(f"[probe_sae_topk] {len(concepts)} concept(s) x k in {ks}: "
         f"{concepts}", flush=True)
    if not concepts:
        print("[probe_sae_topk] nothing below the R^2 threshold — nothing to probe")
        return

    # ---- data (same protocol as run_probing.py: valid=train, test=held out) ----
    A_tr, Y_tr, _ = sl.load_split(processed, "valid", cfg.data.layer, n_shards=ns, binarize=True)
    A_te, Y_te, _ = sl.load_split(processed, "test", cfg.data.layer, n_shards=ns, binarize=True)
    A_tr, A_te = A_tr.half(), A_te.half()
    gc.collect()
    protein_id_tr = sl.residue_protein_ids(processed, "valid", ns)
    protein_id_te = sl.residue_protein_ids(processed, "test", ns)

    from interplm.sae.inference import load_sae_from_hf
    plm = "esm2-8m" if "8M" in cfg.data.model_name else "esm2-650m"
    sae = load_sae_from_hf(plm, cfg.data.layer)

    out_dir = Path(args.out_dir) if args.out_dir else out.parent / f"{out.name}_probing"
    run = RunLogger(out_dir)
    partial = run.metrics_dir / "probing_sae_topk_partial.jsonl"
    results = []
    if partial.exists() and not args.restart:
        results = [json.loads(l) for l in partial.read_text().splitlines() if l.strip()]
        print(f"[probe_sae_topk] resuming: {len(results)} (concept, k) probes already done")
    done = {(r["concept"], r["k"]) for r in results}

    for ci, concept in enumerate(concepts, 1):
        todo = [k for k in ks if (concept, k) not in done]
        if not todo:
            continue
        y_tr_full = Y_tr[concept].to_numpy().astype(int)
        y_te_full = Y_te[concept].to_numpy().astype(int)
        i_tr = pr.balanced_indices(y_tr_full, args.max_pos, args.neg_per_pos, seed=cfg.seed,
                                   protein_id=protein_id_tr)
        i_te = pr.balanced_indices(y_te_full, args.max_pos, args.neg_per_pos, seed=cfg.seed + 1,
                                   protein_id=protein_id_te)
        y_tr, y_te = y_tr_full[i_tr], y_te_full[i_te]

        Ftr = bl.sae_bank(sae, A_tr[i_tr].float()).numpy()
        Fte = bl.sae_bank(sae, A_te[i_te].float()).numpy()
        for k in todo:
            res = pr.probe_concept_topk(Ftr, y_tr, Fte, y_te, concept, "sae", k, seed=cfg.probing.seed)
            res_d = res.to_dict()
            results.append(res_d)
            with open(partial, "a") as f:
                f.write(json.dumps(res_d) + "\n")
        del Ftr, Fte
        gc.collect()
        row = {r["k"]: round(r["test_auc"], 3) for r in results if r["concept"] == concept}
        print(f"[probe_sae_topk] ({ci}/{len(concepts)}) {concept[:44]:<44} "
             f"glp={glp_auc[concept]:.3f} sae_k1={sae_k1_auc[concept]:.3f} sae_topk={row}", flush=True)

    # ---- verdict per concept: does SAE-topk reach GLP, and at what k? ----
    verdicts = []
    for concept in concepts:
        rows = sorted((r for r in results if r["concept"] == concept), key=lambda r: r["k"])
        reaches = [r["k"] for r in rows if r["test_auc"] >= glp_auc[concept]]
        verdicts.append({
            "concept": concept,
            "glp_test_auc": glp_auc[concept],
            "sae_k1_test_auc": sae_k1_auc[concept],
            "sae_topk_test_auc": {r["k"]: r["test_auc"] for r in rows},
            "min_k_reaching_glp": min(reaches) if reaches else None,
            "verdict": "distributed_in_dictionary" if reaches else "not_in_dictionary",
        })
        v = verdicts[-1]
        tag = f"reaches at k={v['min_k_reaching_glp']}" if reaches else f"never reaches (k<=" + f"{max(ks)})"
        print(f"[probe_sae_topk] VERDICT {concept[:44]:<44} {v['verdict']:<24} {tag}")

    (run.metrics_dir / "probing_sae_topk.json").write_text(json.dumps(
        {"results": results, "verdicts": verdicts, "ks": ks, "r2_threshold": args.r2_threshold},
        indent=2))

    import csv
    csv_path = run.metrics_dir / "probing_sae_topk.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["concept", "glp_test_auc", "sae_k1_test_auc", "k", "sae_topk_test_auc",
                   "min_k_reaching_glp", "verdict"])
        for v in verdicts:
            for k in ks:
                w.writerow([v["concept"], f"{v['glp_test_auc']:.6f}", f"{v['sae_k1_test_auc']:.6f}",
                           k, f"{v['sae_topk_test_auc'].get(k, float('nan')):.6f}",
                           v["min_k_reaching_glp"], v["verdict"]])
    print(f"[probe_sae_topk] wrote {csv_path}")


if __name__ == "__main__":
    main()
