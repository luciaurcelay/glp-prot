#!/usr/bin/env python
"""Discovery §1 — k-feature curve + residual probing (Discovery section spec, Part 1).

Assembles the k-curve deliverable from data that already exists (no new compute
for k in {1,2,5,10,20}):
  - k=1  : `probing.json`'s existing `method="sae"` row IS the single-best-feature
           probing result this whole codebase already computes.
  - k=2,5,10,20: `probing_sae_topk.json` (scripts/probe_sae_topk.py's joint SAE
           top-k probe, already run for every falsification-check concept).
  - falsification_r2: `falsification.json`.

Then, for the R^2 < --r2-threshold candidates only (the concepts the aggregate R^2
check flags as NOT already spanned by the SAE dictionary — Discovery spec's
"strongest candidates"), runs the "optional but stronger" residual probe: encode
then decode each candidate through the SAE and probe what's LEFT OVER
(`residual = x - sae.decode(sae.encode(x))`, same normalize/unnormalize convention
already used by `glp_plm/eval/loss_recovered.py` and `scripts/run_gate_g2.py` —
`sae._normalize_input_and_get_norms` / `sae._unnormalize_output`) with the exact
same 1-D probing protocol (`probing_1d.probe_concept`) raw_layer/raw_mlp already
use. A concept probeable from the residual is direct positive evidence the SAE's
own reconstruction threw away exactly what encodes that concept.

Run AFTER run_probing.py (--metric auc) and scripts/probe_sae_topk.py.
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
from glp_plm.eval import probing_1d as pr
from glp_plm.experiments import RunLogger
from glp_plm.utils import resolve_config_data_dir, resolve_storage_dir, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval-dir", help="…/swissprot/processed. Default: config data.eval_dir, "
                                        "else <output_dir>/swissprot/processed")
    ap.add_argument("--probing-json", help="default <out-dir>/metrics/probing.json")
    ap.add_argument("--falsification-json", help="default <out-dir>/metrics/falsification.json")
    ap.add_argument("--topk-json", help="probe_sae_topk.py's output; default "
                                        "<out-dir>/metrics/probing_sae_topk.json")
    ap.add_argument("--r2-threshold", type=float, default=0.6,
                    help="candidates for residual probing = falsification R^2 below this "
                         "(Discovery spec default; fix in advance, don't tune post-hoc)")
    ap.add_argument("--max-pos", type=int, default=3000)
    ap.add_argument("--neg-per-pos", type=float, default=1.0)
    ap.add_argument("--n-shards-per-split", type=int)
    ap.add_argument("--out-dir", help="default <output_dir> itself for this run "
                                      "(uniref50_500m_4096_d24_steering keeps probing/"
                                      "falsification/topk metrics directly under its own "
                                      "metrics/, not a sibling _probing dir)")
    ap.add_argument("--skip-residual", action="store_true",
                    help="only assemble discovery_kfeature.csv, skip the residual probe")
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    out = Path(cfg.output_dir)
    out_dir = Path(args.out_dir) if args.out_dir else out
    run = RunLogger(out_dir)

    probing_path = Path(args.probing_json) if args.probing_json else run.metrics_dir / "probing.json"
    fals_path = Path(args.falsification_json) if args.falsification_json else run.metrics_dir / "falsification.json"
    topk_path = Path(args.topk_json) if args.topk_json else run.metrics_dir / "probing_sae_topk.json"

    probing = json.loads(probing_path.read_text())
    fals = json.loads(fals_path.read_text())
    topk = json.loads(topk_path.read_text())

    glp_auc = {r["concept"]: r["test_auc"] for r in probing["results"] if r["method"] == "glp"}
    sae_k1 = {r["concept"]: r for r in probing["results"] if r["method"] == "sae"}
    r2 = {r["concept"]: r["r2"] for r in fals["results"]}
    candidates = {c for c, v in r2.items() if v < args.r2_threshold}
    print(f"[discovery_kfeature] {len(candidates)} candidate concept(s) at "
         f"R^2 < {args.r2_threshold}: {sorted(candidates, key=lambda c: r2[c])}", flush=True)

    # ---- assemble discovery_kfeature.csv: k=1 (from probing.json) + k=2,5,10,20 (from topk.json) ----
    rows = []
    for concept, r2v in r2.items():
        if concept not in glp_auc:
            continue
        is_cand = concept in candidates
        if concept in sae_k1:
            s = sae_k1[concept]
            rows.append({"concept": concept, "k": 1, "val_auc": s["val_auc"], "test_auc": s["test_auc"],
                        "glp_metaneuron_auc": glp_auc[concept], "falsification_r2": r2v,
                        "is_candidate": is_cand})
    for r in topk["results"]:
        concept = r["concept"]
        if concept not in r2:
            continue
        rows.append({"concept": concept, "k": r["k"], "val_auc": r["val_auc"], "test_auc": r["test_auc"],
                    "glp_metaneuron_auc": glp_auc[concept], "falsification_r2": r2[concept],
                    "is_candidate": concept in candidates})
    rows.sort(key=lambda r: (r["falsification_r2"], r["concept"], r["k"]))

    import csv
    kfeature_csv = run.metrics_dir / "discovery_kfeature.csv"
    with open(kfeature_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["concept", "k", "val_auc", "test_auc",
                                          "glp_metaneuron_auc", "falsification_r2", "is_candidate"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[discovery_kfeature] wrote {kfeature_csv} ({len(rows)} rows, "
         f"{len(set(r['concept'] for r in rows))} concepts x up to 5 k's)")

    if args.skip_residual or not candidates:
        return

    # ---- residual probing: candidates only ----
    ns = args.n_shards_per_split
    if args.eval_dir:
        processed = Path(args.eval_dir)
    elif cfg.data.eval_dir:
        processed = resolve_config_data_dir(cfg.data.eval_dir)
    else:
        processed = resolve_storage_dir(out, "swissprot", "GLP_DATA_ROOT", symlink=False) / "processed"

    A_tr, Y_tr, _ = sl.load_split(processed, "valid", cfg.data.layer, n_shards=ns, binarize=True)
    A_te, Y_te, _ = sl.load_split(processed, "test", cfg.data.layer, n_shards=ns, binarize=True)
    A_tr, A_te = A_tr.half(), A_te.half()
    gc.collect()
    protein_id_tr = sl.residue_protein_ids(processed, "valid", ns)
    protein_id_te = sl.residue_protein_ids(processed, "test", ns)

    from interplm.sae.inference import load_sae_from_hf
    plm = "esm2-8m" if "8M" in cfg.data.model_name else "esm2-650m"
    sae = load_sae_from_hf(plm, cfg.data.layer)

    @torch.no_grad()
    def residual_bank(x: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
        """x - sae.decode(sae.encode(x)) — the SAE's own reconstruction convention
        (glp_plm/eval/loss_recovered.py:127-128, scripts/run_gate_g2.py:147-148)."""
        dev = next(sae.parameters()).device
        out = []
        for i in range(0, x.shape[0], batch_size):
            chunk = x[i:i + batch_size].float().to(dev)
            x_norm, orig_norms = sae._normalize_input_and_get_norms(chunk)
            recon = sae._unnormalize_output(sae.decode(sae.encode(x_norm)), orig_norms)
            out.append((chunk - recon).detach().cpu())
        return torch.cat(out, dim=0)

    residual_path = run.metrics_dir / "discovery_residual.csv"
    partial = run.metrics_dir / "discovery_residual_partial.jsonl"
    results = []
    if partial.exists() and not args.restart:
        results = [json.loads(l) for l in partial.read_text().splitlines() if l.strip()]
        print(f"[discovery_kfeature] resuming: {len(results)} residual probe(s) already done")
    done = {r["concept"] for r in results}

    for ci, concept in enumerate(sorted(candidates, key=lambda c: r2[c]), 1):
        if concept in done:
            continue
        y_tr_full = Y_tr[concept].to_numpy().astype(int)
        y_te_full = Y_te[concept].to_numpy().astype(int)
        i_tr = pr.balanced_indices(y_tr_full, args.max_pos, args.neg_per_pos, seed=cfg.seed,
                                   protein_id=protein_id_tr)
        i_te = pr.balanced_indices(y_te_full, args.max_pos, args.neg_per_pos, seed=cfg.seed + 1,
                                   protein_id=protein_id_te)
        y_tr, y_te = y_tr_full[i_tr], y_te_full[i_te]

        Rtr = residual_bank(A_tr[i_tr]).numpy()
        Rte = residual_bank(A_te[i_te]).numpy()
        res = pr.probe_concept(Rtr, y_tr, Rte, y_te, concept, "residual", seed=cfg.probing.seed)
        row = {"concept": concept, "selected_residual_dim": res.feature_index,
              "val_auc": res.val_auc, "test_auc": res.test_auc}
        results.append(row)
        with open(partial, "a") as f:
            f.write(json.dumps(row) + "\n")
        del Rtr, Rte
        gc.collect()
        print(f"[discovery_kfeature] residual ({ci}/{len(candidates)}) {concept[:44]:<44} "
             f"glp={glp_auc[concept]:.3f} residual_test_auc={res.test_auc:.3f} "
             f"(r2={r2[concept]:.3f})", flush=True)

    with open(residual_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["concept", "selected_residual_dim", "val_auc", "test_auc"])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"[discovery_kfeature] wrote {residual_path} ({len(results)} concepts)")


if __name__ == "__main__":
    main()
