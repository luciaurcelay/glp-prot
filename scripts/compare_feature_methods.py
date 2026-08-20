#!/usr/bin/env python
"""Three feature-selection methods x two representations, for a concept group.

Rebuild of the ad-hoc `compare_zinc_finger_methods.py` / `compare_ss_*` scripts that
produced `docs/steering_history.md` §15-18's numbers — those were never saved anywhere
(not committed, not on disk after the session ended), so none of that was reproducible.
This is the tracked, generalized replacement: same three methods, any concept group,
run identically on both the SAE and GLP representations (interplm_exact's operating-
point grid adapts per representation — see below — but the method itself, "exhaustive
search, no shortlist," is the same on both; nothing is SAE-only by principle).

Methods (SAE, 10240 features):
  own            AUC shortlist (`probing_1d.fast_auc_matrix`, top `--shortlist-k`) ->
                 domain-adjusted F1 selection (`probing_f1.select_topn_by_f1`), the
                 rate-matched two-sided operating-point grid (`probing_f1.RATES`).
                 Same protocol as `run_steering.py`'s `select_zinc_finger_features`,
                 generalized past zinc-finger (that function's own AUC shortlist step
                 still uses `topk_filter`'s mean-diff proxy per §16.2 — unfixed there;
                 this script's shortlist uses the real AUC, resolving §16.2 here).
  interplm_exact Literal reproduction of InterPLM's own criterion: exhaustive grid over
                 ALL 10240 features x their default threshold_percents
                 [0, 0.15, 0.5, 0.6, 0.8] (`interplm/analysis/concepts/compare_activations
                 .py:325`), one-sided (their non-negative-ReLU assumption). Same
                 recall/recall_per_domain/F1 formulas as "own" (`calculate_f1.py:64-86`,
                 already faithfully ported in `probing_f1.py`) — only the operating-point
                 grid and candidate breadth differ.
  villegas_garcia Villegas-Garcia & Ansuini eq. 5-6: precision OR recall > 0.80 (plain
                 activation>0, no threshold sweep) over all 10240 features, ranked by
                 recall. Not F1-optimized — a description of associated latents, not a
                 selection criterion (docs/steering_history.md §15.1).

GLP (meta-neurons, n_blocks*d_mlp dims, dense signed activations — for
`configs/esm8m_h100.yaml`, 12*1280 = 15,360, actually WIDER than the SAE's 10240):
all three methods run. own's shortlist/F1 grid and villegas_garcia's sign rule are
unchanged (AUC ranking is rank-based and scale-free; sign is scale-free). interplm_exact
keeps its defining property — exhaustive, every candidate, no shortlist — but can't
reuse InterPLM's literal fixed threshold VALUES (0.15, 0.5, 0.6, 0.8): those are
calibrated to the SAE's own normalized activation scale and are meaningless on GLP's
differently-scaled dense signed meta-neurons (the exact reason `probing_f1.py`'s
rate-based grid replaced that literal grid project-wide for GLP, see that module's
docstring). So GLP's interplm_exact scans every meta-neuron with the RATES grid
instead, two-sided (no non-negative-ReLU assumption to exploit here) — same
"exhaustive, no shortlist" search, representation-appropriate operating points. This is
substantially more expensive than the SAE version: ~2x the candidates, two-sided vs.
one-sided (~4x the columns scanned), so budget accordingly when launching.

Concept groups (`--concept-group`):
  ss    Helix, Beta strand, Coiled coil, Turn
  zinc  Zinc finger_C2H2-type (base) + numbered 1-5 + a synthetic UNION(1-5) built here
        (offset each numbered column's domain-instance ids so they stay globally unique
        when merged; exclusivity across the five numbered columns is asserted, not
        assumed — see `docs/steering_history.md` §18.4 for why the union matters: it
        found feature 1435 at InterPLM-exact test F1 0.907, far above the base label).

Output: `<out-dir>/metrics/compare_feature_methods_<group>.json` (full results) +
a markdown summary with a final steering-candidate shortlist per concept group and
representation, printed to stdout and written alongside the JSON.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import _bootstrap  # noqa: F401
import numpy as np
import torch

from glp_plm.config import load_config
from glp_plm.data import swissprot_labels as sl
from glp_plm.diffusion.scaler import Scaler
from glp_plm.eval import baselines as bl
from glp_plm.eval import probing_1d as pr
from glp_plm.eval import probing_f1 as pf
from glp_plm.eval.meta_neurons import meta_neuron_bank
from glp_plm.experiments import RunLogger
from glp_plm.model import GLPDenoiser
from glp_plm.utils import get_device, resolve_config_data_dir, resolve_storage_dir, set_seed

SS_CONCEPTS = ["Helix", "Beta strand", "Coiled coil", "Turn"]
ZINC_BASE = "Zinc finger_C2H2-type"
ZINC_NUMBERED = [f"{ZINC_BASE} {i}" for i in range(1, 6)]
ZINC_UNION_NAME = f"{ZINC_BASE} UNION(1-5)"
UNION_OFFSET = 10_000_000


def build_union_labels(labels_df, numbered: List[str]) -> Tuple[np.ndarray, int]:
    """Merge numbered domain-instance columns into one, offsetting each column's ids
    so instances from different numbered concepts never collide once merged. Returns
    (union_labels, n_residues_positive_in_more_than_one_numbered_column)."""
    cols = [labels_df[c].to_numpy().astype(np.int64) for c in numbered]
    union = np.zeros_like(cols[0])
    overlap = np.zeros_like(cols[0], dtype=bool)
    for i, col in enumerate(cols):
        nz = col > 0
        overlap |= nz & (union > 0)
        union = np.where(nz, (i + 1) * UNION_OFFSET + col, union)
    return union, int(overlap.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval-dir")
    ap.add_argument("--ckpt")
    ap.add_argument("--scaler")
    ap.add_argument("--concept-group", choices=["ss", "zinc"], required=True)
    ap.add_argument("--representations", default="sae,glp")
    ap.add_argument("--top-n", type=int, default=2)
    ap.add_argument("--shortlist-k", type=int, default=64,
                    help="own method's AUC shortlist size before F1 selection")
    ap.add_argument("--max-pos", type=int, default=3000)
    ap.add_argument("--neg-per-pos", type=float, default=1.0)
    ap.add_argument("--villegas-precision-thr", type=float, default=0.8)
    ap.add_argument("--villegas-recall-thr", type=float, default=0.8)
    ap.add_argument("--interplm-thresholds", default="0,0.15,0.5,0.6,0.8")
    ap.add_argument("--n-shards-per-split", type=int)
    ap.add_argument("--out-dir", help="default <output_dir>_probing (sibling of the training "
                                      "run — same convention as run_probing.py)")
    ap.add_argument("--train-manifest")
    ap.add_argument("--skip-leakage-check", action="store_true")
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    out = Path(cfg.output_dir)
    ckpt_dir = resolve_storage_dir(out, "checkpoints", "GLP_RUNS_ROOT", symlink=False)
    if args.eval_dir:
        processed = Path(args.eval_dir)
    elif cfg.data.eval_dir:
        processed = resolve_config_data_dir(cfg.data.eval_dir)
    else:
        processed = resolve_storage_dir(out, "swissprot", "GLP_DATA_ROOT", symlink=False) / "processed"

    sl.leakage_gate(cfg, out, processed, args.train_manifest, args.skip_leakage_check)

    reps = [r.strip() for r in args.representations.split(",") if r.strip()]
    interplm_thresholds = [float(t) for t in args.interplm_thresholds.split(",") if t.strip()]
    ns = args.n_shards_per_split

    # ---- data: binarize=False -- F1 needs raw domain-instance indices (§16.1) ----
    A_val, labels_val, names = sl.load_split(processed, "valid", cfg.data.layer, n_shards=ns,
                                             binarize=False)
    A_test, labels_test, _ = sl.load_split(processed, "test", cfg.data.layer, n_shards=ns,
                                           binarize=False)
    A_val, A_test = A_val.half(), A_test.half()
    gc.collect()
    print(f"[compare] valid {tuple(A_val.shape)}  test {tuple(A_test.shape)}", flush=True)
    protein_id_val = sl.residue_protein_ids(processed, "valid", ns)
    protein_id_test = sl.residue_protein_ids(processed, "test", ns)

    # ---- concept list ----
    concepts: List[Tuple[str, np.ndarray, np.ndarray]] = []
    if args.concept_group == "ss":
        for c in SS_CONCEPTS:
            assert c in names, f"{c!r} not in eval store concepts"
            concepts.append((c, labels_val[c].to_numpy().astype(np.int64),
                            labels_test[c].to_numpy().astype(np.int64)))
        # Beta strand's DSSP-based fold check (ss_hit >= 70% strand content of the
        # WHOLE chain) never fires -- real strand segments are short runs bounded by
        # turns, not chain-spanning. Fusing "Beta strand" with "Turn" (a beta-hairpin's
        # two components) via the same union machinery as zinc's UNION(1-5) tests
        # whether a compound SAE detector finds a stronger, more attainable signal
        # than either standalone concept -- reuses build_union_labels exactly (it
        # only needs a list of concept-column names, nothing zinc-specific).
        bt_val, ov_val = build_union_labels(labels_val, ["Beta strand", "Turn"])
        bt_test, ov_test = build_union_labels(labels_test, ["Beta strand", "Turn"])
        if ov_val or ov_test:
            print(f"[compare] WARNING: Beta strand + Turn union exclusivity violated -- "
                 f"{ov_val} valid / {ov_test} test residues positive in both (union still "
                 "built, counts may double-attribute)", flush=True)
        else:
            print("[compare] Beta strand + Turn union exclusivity holds", flush=True)
        concepts.append(("Beta strand + Turn", bt_val, bt_test))
    else:
        for c in [ZINC_BASE] + ZINC_NUMBERED:
            assert c in names, f"{c!r} not in eval store concepts"
            concepts.append((c, labels_val[c].to_numpy().astype(np.int64),
                            labels_test[c].to_numpy().astype(np.int64)))
        u_val, ov_val = build_union_labels(labels_val, ZINC_NUMBERED)
        u_test, ov_test = build_union_labels(labels_test, ZINC_NUMBERED)
        if ov_val or ov_test:
            print(f"[compare] WARNING: UNION(1-5) exclusivity violated -- {ov_val} valid / "
                 f"{ov_test} test residues positive in >1 numbered column (union still built, "
                 "counts may double-attribute)", flush=True)
        else:
            print("[compare] UNION(1-5) exclusivity holds (0 residues in >1 numbered column)",
                 flush=True)
        concepts.append((ZINC_UNION_NAME, u_val, u_test))

    # ---- representations ----
    sae = glp_model = scaler = None
    n_feat = {}
    if "sae" in reps:
        from interplm.sae.inference import load_sae_from_hf
        plm = "esm2-8m" if "8M" in cfg.data.model_name else "esm2-650m"
        sae = load_sae_from_hf(plm, cfg.data.layer)
        n_feat["sae"] = sae.dict_size
    if "glp" in reps:
        glp_model = GLPDenoiser(cfg.model).to(device).eval()
        glp_model.load_state_dict(torch.load(args.ckpt or ckpt_dir / "glp_final.pt", map_location=device))
        scaler = Scaler.load(args.scaler or ckpt_dir / "rep_statistics.pt")
        n_feat["glp"] = cfg.model.n_blocks * cfg.model.d_mlp
    u_fixed = max(cfg.probing.u_grid) if glp_model is not None else None

    def bank(rep: str, split: str, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        A = A_val if split == "valid" else A_test
        if rep == "sae":
            return bl.sae_bank(sae, A[rows].float()).numpy()[:, cols]
        elif rep == "glp":
            z = scaler.standardize(A[rows].float())
            f = meta_neuron_bank(glp_model, z, u_fixed, device=device).numpy()
            return f[:, cols]
        raise ValueError(rep)

    out_dir = Path(args.out_dir) if args.out_dir else out.parent / f"{out.name}_probing"
    run = RunLogger(out_dir)
    partial = run.metrics_dir / f"compare_feature_methods_{args.concept_group}_partial.jsonl"
    results: List[Dict] = []
    if partial.exists() and not args.restart:
        results = [json.loads(l) for l in partial.read_text().splitlines() if l.strip()]
        print(f"[compare] resuming: {len(results)} (concept, rep, method) rows already done",
             flush=True)
    done = {(r["concept"], r["representation"], r["method"]) for r in results}

    def log(row: Dict):
        results.append(row)
        with open(partial, "a") as f:
            f.write(json.dumps(row) + "\n")

    for concept, y_val, y_test in concepts:
        n_val, n_test = len(y_val), len(y_test)
        for rep in reps:
            def rbank(split, rows, cols, _rep=rep):
                return bank(_rep, split, rows, cols)

            # Same three methods on both representations. interplm_exact keeps its
            # "exhaustive, no shortlist" character on GLP, but can't reuse InterPLM's
            # literal threshold VALUES [0,0.15,0.5,0.6,0.8] -- those are calibrated to
            # the SAE's own normalized activation scale, meaningless on GLP's
            # differently-scaled dense signed meta-neurons -- so GLP uses the
            # rate-based grid (probing_f1.RATES) there instead, two-sided (GLP has no
            # non-negative-ReLU assumption to exploit). villegas_garcia only checks
            # SIGN (activation > 0), scale-invariant, unchanged across representations.
            methods = ["own", "interplm_exact", "villegas_garcia"]
            needed = [m for m in methods if (concept, rep, m) not in done]
            if not needed:
                continue

            # AUC for EVERY candidate, computed once per (concept, rep) on a balanced
            # subsample (same protocol as "own"'s old shortlist step) and reused to
            # report the true AUC of whichever feature each method picks -- this is
            # the fix for "F1-optimal but AUC-chance" picks like feature 6311
            # (docs/steering_history.md §15.1/§16): don't just trust F1, check the
            # SAME candidate's own AUC directly, not by cross-referencing a
            # differently-shortlisted separate probing run.
            sub = pr.balanced_indices((y_val > 0).astype(int), args.max_pos,
                                      args.neg_per_pos, seed=cfg.seed,
                                      protein_id=protein_id_val)
            sub_feats = rbank("valid", sub, np.arange(n_feat[rep]))
            aucs_full = pr.fast_auc_matrix(sub_feats, (y_val[sub] > 0).astype(int)[:, None])[0]
            del sub_feats

            def auc_of(feat_ids, signs=None):
                if signs is None:
                    signs = [1] * len(feat_ids)
                return [float(aucs_full[f] if s >= 0 else 1.0 - aucs_full[f])
                       for f, s in zip(feat_ids, signs)]

            for method in needed:
                t0 = time.time()
                if method == "own":
                    shortlist = np.argsort(-aucs_full)[:args.shortlist_k]
                    sel = pf.select_topn_by_f1(rbank, n_val, n_test, y_val, y_test, shortlist,
                                               concept, "own", top_n=args.top_n)
                    sel["auc"] = auc_of(sel["feature"], sel["sign"])
                    row = {"concept": concept, "representation": rep, "method": method, **sel}
                elif method == "interplm_exact":
                    if rep == "sae":
                        sel = pf.select_topn_by_f1(rbank, n_val, n_test, y_val, y_test,
                                                   np.arange(n_feat[rep]), concept, "interplm_exact",
                                                   top_n=args.top_n, thresholds=interplm_thresholds,
                                                   two_sided=False)
                    else:
                        sel = pf.select_topn_by_f1(rbank, n_val, n_test, y_val, y_test,
                                                   np.arange(n_feat[rep]), concept, "interplm_exact",
                                                   top_n=args.top_n, thresholds=None, two_sided=True)
                    sel["auc"] = auc_of(sel["feature"], sel["sign"])
                    row = {"concept": concept, "representation": rep, "method": method, **sel}
                else:  # villegas_garcia
                    vg = pf.villegas_garcia_select(rbank, n_val, y_val, np.arange(n_feat[rep]),
                                                   concept, split="valid",
                                                   precision_thr=args.villegas_precision_thr,
                                                   recall_thr=args.villegas_recall_thr,
                                                   top_n=args.top_n)
                    for t in vg["top"]:
                        t["auc"] = auc_of([t["feature"]])[0]
                    row = {"concept": concept, "representation": rep, "method": method, **vg}
                row["seconds"] = round(time.time() - t0, 1)
                row["n_pos_valid"] = int((y_val > 0).sum())
                log(row)
                print(f"[compare] {concept[:40]:<40} {rep:<4} {method:<16} "
                     f"({row['seconds']}s)  {row}", flush=True)

    (run.metrics_dir / f"compare_feature_methods_{args.concept_group}.json").write_text(
        json.dumps({"results": results, "concept_group": args.concept_group,
                   "representations": reps, "top_n": args.top_n}, indent=2))

    # ---- markdown summary + steering candidates ----
    lines = [f"# Feature-selection comparison: {args.concept_group} ({', '.join(reps)})\n"]
    for concept, _, _ in concepts:
        lines.append(f"## {concept}\n")
        for rep in reps:
            for method in ["own", "interplm_exact", "villegas_garcia"]:
                r = next((r for r in results if r["concept"] == concept
                         and r["representation"] == rep and r["method"] == method), None)
                if r is None:
                    continue
                if method == "villegas_garcia":
                    lines.append(f"- **{rep}/{method}**: {r['n_associated']} associated, "
                                f"top-{args.top_n}: {r['top']}")
                else:
                    lines.append(f"- **{rep}/{method}**: features={r['feature']} "
                                f"val_f1={[round(v, 4) for v in r['val_f1']]} "
                                f"test_f1={[round(v, 4) for v in r['test_f1']]} "
                                f"auc={[round(a, 4) for a in r.get('auc', [])]}")
        lines.append("")

    lines.append("## Steering candidates\n")
    lines.append("Best test-F1 pick per concept, per representation (own vs. interplm_exact, "
                 "both representations now -- villegas_garcia isn't F1-optimized on either, "
                 "so it's excluded from this best-by-F1 comparison; see the per-concept "
                 "listing above for its picks).\n")
    for concept, _, _ in concepts:
        cands = []
        for rep in reps:
            best = None
            for method in ["own", "interplm_exact"]:
                r = next((r for r in results if r["concept"] == concept
                         and r["representation"] == rep and r["method"] == method), None)
                if r is None or not r.get("test_f1"):
                    continue
                if best is None or r["test_f1"][0] > best[1]["test_f1"][0]:
                    best = (method, r)
            if best is not None:
                method, r = best
                if rep == "glp":
                    dmlp = cfg.model.d_mlp
                    loc = [f"block {f // dmlp} neuron {f % dmlp}" for f in r["feature"]]
                    cands.append(f"GLP ({method}): meta-neurons {r['feature']} [{'; '.join(loc)}] "
                               f"test_f1={[round(v, 4) for v in r['test_f1']]}")
                else:
                    cands.append(f"SAE ({method}): features {r['feature']} "
                               f"test_f1={[round(v, 4) for v in r['test_f1']]}")
        lines.append(f"- **{concept}**: " + " | ".join(cands))

    md_path = run.metrics_dir / f"compare_feature_methods_{args.concept_group}.md"
    md_path.write_text("\n".join(lines))
    print(f"[compare] wrote {md_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
