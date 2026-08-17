#!/usr/bin/env python
"""Assemble the GLProt probing spec's Part 7 deliverables from the raw metric JSONs
run_probing.py / probe_continuous.py / test_nonlinearity.py already wrote. Run last,
after whichever of those have been run (each section is skipped with a message if its
input is missing, so this is safe to run after only some of them).

Writes into `<output_dir>/metrics/` (alongside `probing.json` etc. — the established
home for this run's numeric artifacts, PLAN §5 repo layout):
  concepts_frozen.json      - copy of frozen_concepts.json (spec's literal filename;
                               the pipeline's own contract — freeze_concepts.py,
                               run_probing.py --concepts — still uses the original name)
  probing_discrete.csv      - concept, category, representation, selected_unit,
                               noise_level_u (GLP only), val_auc, test_auc
  probing_summary.json      - per-representation mean test AUC/rho + CI, paired
                               bootstrap (mean diff, CI, win counts), per-category table

(probing_continuous.csv and probing_falsification.csv are already written directly by
probe_continuous.py / test_nonlinearity.py — not duplicated here.)
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401

from glp_plm.config import load_config
from glp_plm.eval import stats as pstats
from glp_plm.eval.concept_selection import categorize
from glp_plm.eval.probing_1d import bootstrap_ci
from glp_plm.experiments import RunLogger
from glp_plm.utils import resolve_config_data_dir, resolve_storage_dir


def _write_discrete_csv(results, path: Path) -> int:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["concept", "category", "representation", "selected_unit",
                   "noise_level_u", "val_auc", "test_auc"])
        for r in results:
            w.writerow([r["concept"], categorize(r["concept"]) or "", r["method"],
                       r["feature_index"], r.get("u", ""),
                       f"{r['val_auc']:.6f}", f"{r['test_auc']:.6f}"])
    return len(results)


def _representation_summary(results, key, methods):
    agg = {}
    for m in methods:
        vals = [r[key] for r in results if r["method"] == m]
        if not vals:
            continue
        mean, lo, hi = bootstrap_ci(vals)
        agg[m] = {f"mean_{key}": mean, "ci_low": lo, "ci_high": hi, "n": len(vals)}
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval-dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(cfg.output_dir)
    run = RunLogger(out)

    if args.eval_dir:
        processed = Path(args.eval_dir)
    elif cfg.data.eval_dir:
        processed = resolve_config_data_dir(cfg.data.eval_dir)
    else:
        processed = resolve_storage_dir(out, "swissprot", "GLP_DATA_ROOT", symlink=False) / "processed"

    summary = {"glp_depth": f"{cfg.model.n_blocks}-block",
              "glp_meta_neuron_width": cfg.model.n_blocks * cfg.model.d_mlp}

    # ---- concepts_frozen.json (spec's literal name; alias of frozen_concepts.json) ----
    frozen = processed / "frozen_concepts.json"
    if frozen.exists():
        shutil.copy(frozen, run.metrics_dir / "concepts_frozen.json")
        print(f"[make_probing_outputs] concepts_frozen.json <- {frozen}")
    else:
        print(f"[make_probing_outputs] no {frozen} — run scripts/freeze_concepts.py first")

    # ---- discrete (probing.json, from run_probing.py --metric auc) ----
    discrete_path = run.metrics_dir / "probing.json"
    if discrete_path.exists():
        data = json.loads(discrete_path.read_text())
        results = data["results"]
        methods = sorted({r["method"] for r in results})
        n = _write_discrete_csv(results, run.metrics_dir / "probing_discrete.csv")
        print(f"[make_probing_outputs] probing_discrete.csv: {n} rows")

        primary = "glp" if "glp" in methods else methods[0]
        others = [m for m in methods if m != primary]
        summary["discrete"] = {
            "n_concepts": len({r["concept"] for r in results}),
            "u_grid": data.get("u_grid"),
            "representations": _representation_summary(results, "test_auc", methods),
            "paired_bootstrap": pstats.paired_comparisons(results, "test_auc", primary, others),
            "per_category": pstats.category_table(results, "test_auc", methods),
        }
    else:
        print(f"[make_probing_outputs] no {discrete_path} — skipping discrete section "
             f"(run scripts/run_probing.py --metric auc first)")

    # ---- continuous (probing_continuous.json, from probe_continuous.py) ----
    cont_path = run.metrics_dir / "probing_continuous.json"
    if cont_path.exists():
        data = json.loads(cont_path.read_text())
        results = data["results"]
        methods = sorted({r["method"] for r in results})
        primary = "glp" if "glp" in methods else methods[0]
        others = [m for m in methods if m != primary]
        summary["continuous"] = {
            "n_properties": len({r["property"] for r in results}),
            "pool": data.get("pool"), "u_grid": data.get("u_grid"),
            "representations": _representation_summary(results, "test_spearman", methods),
            "paired_bootstrap": pstats.paired_comparisons(results, "test_spearman", primary, others),
        }
    else:
        print(f"[make_probing_outputs] no {cont_path} — skipping continuous section "
             f"(run scripts/probe_continuous.py first)")

    # ---- falsification (falsification.json, from test_nonlinearity.py) ----
    fals_path = run.metrics_dir / "falsification.json"
    if fals_path.exists():
        data = json.loads(fals_path.read_text())
        summary["falsification"] = {"n_glp_wins": data.get("n_glp_wins"),
                                    "r2_distribution": data.get("r2_distribution")}
    else:
        print(f"[make_probing_outputs] no {fals_path} — skipping falsification section "
             f"(run scripts/test_nonlinearity.py first)")

    summary_path = run.metrics_dir / "probing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[make_probing_outputs] wrote {summary_path}")


if __name__ == "__main__":
    main()
