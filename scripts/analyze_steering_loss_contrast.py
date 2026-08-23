#!/usr/bin/env python
"""Contrast GLP's loss-restoration on an isolated activation (the diagnostic) vs. on the
finished generated sequence, for both steering mechanisms.

The premise (from the original GLP work): steering pushes an activation off-manifold, and one
GLP/SDEdit pass pulls it back (lowers the diffusion loss). We confirm this holds for an
activation *in isolation* but FAILS for the sequences steering actually produces — the
corrected arm's re-embedded loss is not lower than naive. This script computes both sides from
the persisted run artifacts so every number traces to a live file.

Per steering run it reads:
  - metrics/summary.json : diagnostic_{A,B}_means (naive/off-manifold loss by alpha) and
    diagnostic_{A,B}_means_corrected (after one SDEdit pass).            [activation in isolation]
  - expA_*/expB_* CSV    : per-generation `diffusion_loss` (finished sequence re-embedded),
    grouped by `corrected`.                                             [the output]

Writes <out>/steering_loss_contrast.json and steering_loss_contrast.png, and prints a table.
Method is auto-detected (A = SAE-clamp structural, B = diff-of-means) from the summary keys.
"""
import argparse
import glob
import json
import os

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from glp_plm.experiments import style

# (label, run_dir relative to --runs-root). Both artifacts live in the same dir.
DEFAULT_RUNS = [
    # diff-of-means (method B)
    ("hydrophobicity", "steering_crg/steering_diffmean_hydro_bidir_n40"),
    ("charge",         "steering_crg/steering_diffmean_charge_rel_charge_n40"),
    ("instability",    "steering_crg/steering_diffmean_instability_rel_instability_n40"),
    # SAE-clamp structural (method A)
    ("helix",       "uniref50_500m_4096_d24_steering/steering_sae_zinc_helix_5355"),
    ("coiled coil", "uniref50_500m_4096_d24_steering/steering_sae_zinc_coiledcoil_813"),
    ("beta strand", "uniref50_500m_4096_d24_steering/steering_sae_zinc_betastrand_2868"),
    ("zinc finger", "uniref50_500m_4096_d24_steering/steering_sae_zinc_union1435"),
]


def _find_csv(run_dir):
    # Prefer the canonical per-generation CSV, not re-scored/.orig variants.
    cands = sorted(glob.glob(os.path.join(run_dir, "expA_sae_zincfinger.csv")) +
                   glob.glob(os.path.join(run_dir, "expB_diffmean_*.csv")))
    if cands:
        return cands[0]
    cands = [c for c in glob.glob(os.path.join(run_dir, "exp[AB]_*.csv")) if ".orig" not in c]
    return cands[0] if cands else None


def analyze_run(label, run_dir):
    summ = json.load(open(os.path.join(run_dir, "metrics", "summary.json")))
    tag = "B" if "diagnostic_B_means" in summ else "A"
    method = "diff-of-means" if tag == "B" else "SAE-clamp (structural)"
    naive = summ[f"diagnostic_{tag}_means"]            # {"baseline":..., "<alpha>":loss}
    corr = summ[f"diagnostic_{tag}_means_corrected"]   # {"<alpha>":loss}
    baseline = float(naive["baseline"])
    alphas = sorted((float(a) for a in naive if a != "baseline"))
    amax = alphas[-1]
    diag_naive = float(naive[str(amax) if str(amax) in naive else f"{amax:g}"])
    # corrected dict keys match naive's alpha keys
    ck = str(amax) if str(amax) in corr else next(k for k in corr if abs(float(k) - amax) < 1e-9)
    diag_corr = float(corr[ck])

    csv = _find_csv(run_dir)
    df = pd.read_csv(csv)
    df = df[df["diffusion_loss"].notna()]
    fin_naive = df[~df["corrected"]]["diffusion_loss"]
    fin_corr = df[df["corrected"]]["diffusion_loss"]
    p = float(mannwhitneyu(fin_corr, fin_naive).pvalue)

    return {
        "label": label, "method": method, "run_dir": run_dir,
        "baseline_real_loss": baseline,
        "diagnostic": {"alpha_max": amax, "naive_loss": diag_naive, "corrected_loss": diag_corr,
                       "restoration": diag_naive - diag_corr},          # >0 = GLP pulled it back
        "finished": {"n": int(len(df)), "naive_loss": float(fin_naive.mean()),
                     "corrected_loss": float(fin_corr.mean()),
                     "delta_corr_minus_naive": float(fin_corr.mean() - fin_naive.mean()),
                     "mannwhitney_p": p},
    }


def make_figure(results, out_png):
    style.apply_style()
    import matplotlib.pyplot as plt
    c_naive = style.STEERING_ARM_COLORS["naive"]
    c_corr = style.STEERING_ARM_COLORS["corrected"]
    labels = [r["label"] for r in results]
    x = np.arange(len(results)); w = 0.38
    base = results[0]["baseline_real_loss"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))
    # Left: diagnostic (activation in isolation) at each run's max alpha — GLP restores it.
    axL.bar(x - w/2, [r["diagnostic"]["naive_loss"] for r in results], w, color=c_naive, label="naive (steered)")
    axL.bar(x + w/2, [r["diagnostic"]["corrected_loss"] for r in results], w, color=c_corr, label="GLP-corrected")
    axL.axhline(base, color=style.BASELINE, lw=1.2, ls="--")
    axL.set_yscale("log")
    axL.set_title("Activation in isolation (diagnostic, max $\\alpha$)")
    axL.set_ylabel("GLP diffusion loss (log)")
    axL.set_xticks(x); axL.set_xticklabels(labels, rotation=35, ha="right")
    axL.legend(loc="upper left")

    # Right: finished generated sequence — GLP does NOT restore it (corrected >= naive).
    axR.bar(x - w/2, [r["finished"]["naive_loss"] for r in results], w, color=c_naive, label="naive (steered)")
    axR.bar(x + w/2, [r["finished"]["corrected_loss"] for r in results], w, color=c_corr, label="GLP-corrected")
    axR.axhline(base, color=style.BASELINE, lw=1.2, ls="--")
    axR.annotate("real proteins", (len(results)-0.5, base), fontsize=8, color=style.MUTED, va="bottom", ha="right")
    axR.set_ylim(0.70, 0.90)
    axR.set_title("Finished generated sequence (re-embedded)")
    axR.set_ylabel("GLP diffusion loss")
    axR.set_xticks(x); axR.set_xticklabels(labels, rotation=35, ha="right")
    axR.legend(loc="upper left")

    fig.suptitle("GLP loss restoration: holds on the isolated activation, fails on the output",
                 fontsize=12, fontweight="bold")
    style.save_fig(fig, out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=os.environ.get("GLP_RUNS_ROOT", "runs"))
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    results = []
    for label, rel in DEFAULT_RUNS:
        d = os.path.join(args.runs_root, rel)
        if not os.path.exists(os.path.join(d, "metrics", "summary.json")):
            print(f"[skip] {label}: no summary.json at {d}"); continue
        try:
            results.append(analyze_run(label, d))
        except Exception as e:
            print(f"[skip] {label}: {type(e).__name__}: {e}")

    os.makedirs(args.out, exist_ok=True)
    json.dump({"baseline_real_loss": results[0]["baseline_real_loss"], "runs": results},
              open(os.path.join(args.out, "steering_loss_contrast.json"), "w"), indent=2)
    make_figure(results, os.path.join(args.out, "steering_loss_contrast.png"))

    print(f"\nGLP loss restoration: diagnostic (isolated activation) vs finished sequence. "
          f"Real-protein loss = {results[0]['baseline_real_loss']:.3f}\n")
    print(f"{'run':<15}{'method':<24}{'DIAG naive->corr (a)':<26}{'FINISHED naive->corr':<26}{'MWU p':>9}")
    for r in results:
        dg, fn = r["diagnostic"], r["finished"]
        diag = "%.2f->%.2f (a%g)" % (dg["naive_loss"], dg["corrected_loss"], dg["alpha_max"])
        fin = "%.3f->%.3f (%+.3f)" % (fn["naive_loss"], fn["corrected_loss"], fn["delta_corr_minus_naive"])
        print(f"{r['label']:<15}{r['method']:<24}{diag:<26}{fin:<26}{fn['mannwhitney_p']:>9.1e}")


if __name__ == "__main__":
    main()
