#!/usr/bin/env python
"""Re-score an already-generated Exp A steering CSV with the correct concept-hit detector,
no regeneration needed.

Exp A's concept_score used to be hardcoded to the C2H2 zinc-finger PROSITE regex
(`c2h2_concept_score`, `scripts/run_steering.py`), which has no meaning for any concept
other than zinc-finger -- real proteins don't contain C2H2 motifs at a Coiled coil, so
every row silently scored `motif_hit=False` regardless of whether steering worked
(`runs/.../steering_sae_zinc_coiledcoil_813` is the case that surfaced this). Fixed
going forward via `run_steering.py --concept-detector feature_activation`; this script
applies the same fix retroactively to a CSV that was already generated, by re-deriving
`concept_score` from its own `target_feature_activation_max` column -- already computed
by re-embedding each FINISHED discrete sequence fresh through ESM+SAE, independent of
the clamped internal activation used during generation -- against the feature's own
per-residue F1-calibrated sign/threshold from `compare_feature_methods.py`. Then it
reruns the Part 3 analysis (Pareto, three-regime breakdown) that `run_steering.py`
would have produced live under the corrected detector.

Usage:
  .venv/bin/python scripts/rescore_exp_a_concept.py \\
    --csv runs/uniref50_500m_4096_d24_steering/steering_sae_zinc_coiledcoil_813/expA_sae_zincfinger.csv \\
    --threshold 0.5 --sign 1
(threshold/sign: compare_feature_methods.py's own 'own'/interplm_exact output for this
feature+concept+representation -- e.g. Coiled coil/SAE/feature 813 -> threshold=0.5,
sign=+1, test F1 0.691, from compare_feature_methods_ss_partial.jsonl.)
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401

from run_steering import is_degenerate_sequence, pareto_exp_a, three_regime_breakdown
from glp_plm.experiments import RunLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="expA_sae_zincfinger.csv to rescore in place")
    ap.add_argument("--threshold", type=float, required=True,
                    help="per-residue activation threshold for a concept hit (from "
                         "compare_feature_methods.py's own/interplm_exact 'threshold' output)")
    ap.add_argument("--sign", type=int, default=1, choices=[1, -1],
                    help="sign convention matching compare_feature_methods.py's own 'sign' output")
    ap.add_argument("--neutral-alpha", type=float, default=0.0)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = csv_path.parent
    backup = csv_path.with_name(csv_path.stem + ".orig_c2h2_scored.csv")
    if not backup.exists():
        shutil.copy(csv_path, backup)
        print(f"[rescore] backed up the original (C2H2-scored) CSV to {backup}")
    else:
        print(f"[rescore] {backup} already exists — re-rescoring from it, not from {csv_path}")

    with open(backup) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{backup} has no rows")

    base_fields = [k for k in rows[0].keys() if k != "concept_detector"]
    insert_at = base_fields.index("length") + 1
    fieldnames = base_fields[:insert_at] + ["concept_detector"] + base_fields[insert_at:]

    rows_for_analysis = []
    alphas_seen = set()
    n_hits = 0
    for r in rows:
        feat_max = float(r["target_feature_activation_max"])
        hit = (args.sign * feat_max) >= args.threshold
        n_hits += int(hit)
        r["concept_detector"] = "feature_activation"
        r["motif_hit"] = hit
        r["concept_count"] = float(hit)
        r["concept_count_strict"] = float(hit)
        r["concept_count_loose"] = float(hit)
        alpha = float(r["alpha"])
        alphas_seen.add(alpha)
        rows_for_analysis.append({
            **r, "alpha": alpha, "corrected": r["corrected"] == "True",
            "naturalness_score": float(r["naturalness_score"]),
            "diffusion_loss": float(r["diffusion_loss"]),
            "concept_score": float(hit),
        })

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"[rescore] wrote corrected {csv_path} ({len(rows)} rows, {n_hits} hits, "
          f"detector=feature_activation threshold={args.threshold} sign={args.sign})")

    n_before = len(rows_for_analysis)
    degenerate = [r for r in rows_for_analysis if is_degenerate_sequence(r)]
    rows_for_analysis = [r for r in rows_for_analysis if not is_degenerate_sequence(r)]
    if degenerate:
        print(f"[rescore] excluded {len(degenerate)}/{n_before} degenerate sequences "
              "(see is_degenerate_sequence) from Part 3 analysis")

    alphas = sorted(alphas_seen)
    run = RunLogger(out_dir)
    naive_steered = [r for r in rows_for_analysis
                     if not r["corrected"] and r["alpha"] != args.neutral_alpha]
    pareto_exp_a(rows_for_analysis, out_dir, run, alphas, args.neutral_alpha)
    regimes = three_regime_breakdown(rows_for_analysis, naive_steered)
    anchor_rows = [r for r in rows_for_analysis
                   if r["alpha"] == args.neutral_alpha and not r["corrected"]]
    if anchor_rows:
        anchor_hits = sum(1 for r in anchor_rows if r["concept_score"] > 0)
        print(f"[rescore] unsteered anchor (no steering): {anchor_hits}/{len(anchor_rows)} "
              "feature-activation hits — the detector's own false-positive rate on this "
              "model's unsteered generation distribution.")
    if degenerate:
        run.log_summary(n_degenerate_excluded=len(degenerate))
    run.log_summary(three_regime=regimes, rescore_detector="feature_activation",
                    rescore_threshold=args.threshold, rescore_sign=args.sign)
    print(f"[rescore] done. figures/summary under {out_dir}")


if __name__ == "__main__":
    main()
