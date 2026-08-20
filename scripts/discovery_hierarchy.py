#!/usr/bin/env python
"""Discovery §2 — does the latent space reflect Swiss-Prot family hierarchy
(EF-hand 1-4, Ig-like subtypes, RRM 1/2, zinc-finger subtypes, ...)?

For each of GLP / SAE / raw_mlp (FFN), take the winning unit for every concept
listed in `configs/discovery_families.yaml`, pool its per-residue activation to a
per-protein vector on the held-out TEST split (reusing `probe_continuous.py`'s
`build_bank`/`_pooled_bank`/`_sample_proteins` — the same per-protein pooling
machinery continuous-property probing already validated), correlate every pair of
winning units across those pooled proteins, and test whether mean within-family
correlation exceeds mean between-family correlation (permutation test, self-pairs
excluded) more than chance. GLP-only proves nothing (spec) — SAE and raw_mlp are
run identically as the comparison.

DRAFT STATUS: configs/discovery_families.yaml has not yet been reviewed by a
domain expert (see that file's header) — treat this script's output as
provisional until that review lands.

Run AFTER run_probing.py (--metric auc) — needs its probing.json for winning
(method, concept) -> (u, feature_index).
"""
import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

# scripts/ is on sys.path[0] when run as `python scripts/discovery_hierarchy.py` —
# reuse probe_continuous.py's pooling machinery instead of re-deriving it.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_continuous as pc  # noqa: E402

from glp_plm.config import load_config
from glp_plm.data import swissprot_labels as sl
from glp_plm.eval import baselines as bl
from glp_plm.eval import discovery_hierarchy as dh
from glp_plm.experiments import RunLogger
from glp_plm.model import GLPDenoiser
from glp_plm.utils import get_device, resolve_config_data_dir, resolve_storage_dir, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval-dir")
    ap.add_argument("--ckpt")
    ap.add_argument("--scaler")
    ap.add_argument("--families", default="configs/discovery_families.yaml")
    ap.add_argument("--probing-json", help="default <out-dir>/metrics/probing.json")
    ap.add_argument("--max-proteins", type=int, default=500,
                    help="TEST-split protein subsample for pooling (bounds the GLP "
                         "meta-neuron pooled bank, which is n_blocks*d_mlp*len(u_grid) wide)")
    ap.add_argument("--protein-batch-size", type=int, default=16)
    ap.add_argument("--n-permutations", type=int, default=10000)
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

    families = dh.load_families(args.families)
    family_of = dh.family_of_concept(families)
    print(f"[discovery_hierarchy] {len(families)} families, {len(family_of)} concepts", flush=True)

    probing_path = Path(args.probing_json) if args.probing_json else run.metrics_dir / "probing.json"
    probing = json.loads(probing_path.read_text())

    ns = args.n_shards_per_split
    A_te, Y_te, names = sl.load_split(processed, "test", cfg.data.layer, n_shards=ns, binarize=True)
    spans_full = sl.protein_spans(processed, "test", ns)
    spans = pc._sample_proteins(spans_full, args.max_proteins, seed=cfg.seed)
    print(f"[discovery_hierarchy] pooling {len(spans)}/{len(spans_full)} test proteins", flush=True)

    from glp_plm.diffusion.scaler import Scaler
    scaler = Scaler.load(args.scaler or ckpt_dir / "rep_statistics.pt")
    model = GLPDenoiser(cfg.model).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt or ckpt_dir / "glp_final.pt", map_location=device))
    from interplm.sae.inference import load_sae_from_hf
    plm = "esm2-8m" if "8M" in cfg.data.model_name else "esm2-650m"
    sae = load_sae_from_hf(plm, cfg.data.layer)
    tag = f"s{ns}" if ns else "all"
    mlp_bank = bl.raw_mlp_cached_bank(processed, sl.split_shards(processed, "test", ns),
                                      cfg.data.layer, cfg.data.model_name, device,
                                      processed / "cache" / f"raw_mlp_test_{tag}.npy")
    u_grid = list(cfg.probing.u_grid)
    width = cfg.model.n_blocks * cfg.model.d_mlp

    # ---- per-method winning-unit lookup (concept -> global column index into the
    # method's full pooled bank), restricted to concepts covered by a family ----
    winners = {m: {} for m in ("glp", "sae", "raw_mlp")}
    for r in probing["results"]:
        if r["method"] not in winners or r["concept"] not in family_of:
            continue
        if r["method"] == "glp":
            col = u_grid.index(r["u"]) * width + r["feature_index"]
        else:
            col = r["feature_index"]
        winners[r["method"]][r["concept"]] = col

    # protein-level concept presence (any positive residue), for co-occurrence
    presence = {}
    for c in family_of:
        y = Y_te[c].to_numpy()
        presence[c] = np.array([bool((y[int(r.start):int(r.end)] > 0).any())
                                for r in spans.itertuples()])

    results, verdicts = [], {}
    for method in ("glp", "sae", "raw_mlp"):
        concepts = sorted(winners[method])
        if len(concepts) < 3:
            print(f"[discovery_hierarchy] {method}: only {len(concepts)} family concepts won — skipping")
            continue
        bank = pc.build_bank(method, "test", spans, A_te, model=model, scaler=scaler, sae=sae,
                             mlp_bank=mlp_bank, u_grid=u_grid, device=device,
                             pool_mode="mean", batch_proteins=args.protein_batch_size)
        cols = [winners[method][c] for c in concepts]
        unit_vecs = bank[:, cols]                              # (n_protein, n_concepts)
        corr = np.corrcoef(unit_vecs, rowvar=False)
        unit_of = {c: winners[method][c] for c in concepts}

        rows = dh.pairwise_rows(concepts, corr, family_of, unit_of, method)
        results.extend(r.to_dict() for r in rows)
        test = dh.permutation_test(concepts, corr, family_of, unit_of,
                                   n_perm=args.n_permutations, seed=cfg.seed)
        cooc = dh.cooccurrence(concepts, presence)
        # mean co-occurrence among the within-family, non-self pairs the test used
        n = len(concepts)
        iu, ju = np.triu_indices(n, k=1)
        within = np.array([family_of[concepts[i]] == family_of[concepts[j]] and
                           unit_of[concepts[i]] != unit_of[concepts[j]] for i, j in zip(iu, ju)])
        test["mean_within_family_cooccurrence"] = float(cooc[iu, ju][within].mean()) if within.any() else float("nan")
        test["n_concepts"] = len(concepts)
        verdicts[method] = test
        print(f"[discovery_hierarchy] {method}: within={test['within_family_mean']:.3f} "
             f"between={test['between_family_mean']:.3f} diff={test['difference']:.3f} "
             f"p={test['permutation_p_value']:.4f} "
             f"(excluded {test['n_excluded_self_pairs']} self-pairs, "
             f"cooccurrence={test['mean_within_family_cooccurrence']:.3f})", flush=True)
        del bank, unit_vecs
        import gc; gc.collect()

    import csv
    csv_path = run.metrics_dir / "discovery_hierarchy.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["concept_a", "concept_b", "same_family", "same_unit",
                                          "correlation", "representation"])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"[discovery_hierarchy] wrote {csv_path} ({len(results)} pairs)")

    test_path = run.metrics_dir / "discovery_hierarchy_test.json"
    test_path.write_text(json.dumps({"families_source": args.families, "n_proteins": len(spans),
                                     "results": verdicts}, indent=2))
    print(f"[discovery_hierarchy] wrote {test_path}")


if __name__ == "__main__":
    main()
