"""Auto-generated `run_summary.md` (factual tables only — PLAN §11 / CLAUDE.md repo layout).

`write_run_summary(run)` reads whatever `metrics/*.json` artifacts exist for a run and
renders them into a markdown report, section by section, skipping any section whose
inputs are missing — same idempotent/additive contract as `experiments.report.make_report`,
which calls this after every stage.

This is NOT a replacement for a hand-authored write-up (see `runs/local_clean/run_summary.md`):
it has no narrative interpretation, no limitations, no next steps — those require judgment
this function doesn't have. It exists so every run leaves a factual, tabular record behind
even when nobody gets around to writing the prose version.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from glp_plm.experiments.metrics import RunLogger


def _model_training_section(s: Dict[str, Any]) -> Optional[str]:
    if not any(k in s for k in ("n_blocks", "n_params", "final_loss")):
        return None
    n_params = f"{s['n_params']:,}" if isinstance(s.get("n_params"), int) else s.get("n_params", "—")
    final_loss = f"{s['final_loss']:.4f}" if isinstance(s.get("final_loss"), (int, float)) else "—"
    lines = ["## Model & training", "", "| n_blocks | n_params | final_loss |", "|---|---|---|",
             f"| {s.get('n_blocks', '—')} | {n_params} | {final_loss} |"]
    return "\n".join(lines)


def _generation_section(s: Dict[str, Any], run: RunLogger) -> Optional[str]:
    if "fd_generated_vs_real" not in s:
        return None
    lines = ["## Generation quality (Gate G1: FD + PCA)", ""]
    lines.append(
        f"FD(generated, real) = **{s['fd_generated_vs_real']:.4f}** at "
        f"{s.get('fd_num_steps', '?')} sampler steps, vs FD(train, val) lower bound "
        f"{s.get('fd_lower_bound', float('nan')):.4f} -> ratio **{s.get('fd_ratio', float('nan')):.2f}x**."
    )
    lines.append("")
    lines.append("| PCA stat | value | meaning |")
    lines.append("|---|---|---|")
    pca_meaning = {
        "pca_real_var_captured_frac": "top-2 PCs of real (baseline)",
        "pca_gen_var_captured_frac": "match to baseline = well-aligned",
        "pca_pc_mean_gap_norm": "0 = centered like real, in PC-std units",
        "pca_pc_std_ratio": "1.0 = matched spread",
    }
    for k, meaning in pca_meaning.items():
        if k in s:
            lines.append(f"| {k} | {s[k]:.4f} | {meaning} |")

    sweep_path = run.metrics_dir / "fd_vs_steps.json"
    if sweep_path.exists():
        sweep = json.loads(sweep_path.read_text())
        lines += ["", "FD depends strongly on sampler steps — quote it with its step count:", "",
                   "| sampler steps | FD | ratio to lower bound | spread match |", "|---|---|---|---|"]
        for row in sweep["sweep"]:
            lines.append(f"| {row['steps']} | {row['fd']:.2f} | {row['ratio']:.2f}x | "
                         f"{row['pc_std_ratio']:.3f} |")
        lines.append(f"\nLower bound FD(real, real) = {sweep['lower_bound']:.4f}.")
    return "\n".join(lines)


def _paired_comparisons(results: List[Dict[str, Any]], key: str, primary: str,
                        others: List[str]) -> List[Dict[str, Any]]:
    """Paired bootstrap on per-concept deltas (matched concepts) -> local_clean-style table.

    Thin wrapper over glp_plm.eval.stats (shared with scripts/make_probing_outputs.py,
    GLProt probing spec Part 5) — kept here so callers below don't need to change.
    """
    from glp_plm.eval import stats as pstats

    rows = pstats.paired_comparisons(results, key, primary, others)
    for r in rows:
        r["mean"], r["lo"], r["hi"] = r["mean_diff"], r["ci_low"], r["ci_high"]
    return rows


def _category_breakdown(results: List[Dict[str, Any]], key: str, methods: List[str]) -> Optional[str]:
    from glp_plm.eval import stats as pstats

    table = pstats.category_table(results, key, methods)
    return pstats.category_table_markdown(table, methods) if table else None


def _probing_section(run: RunLogger, metric: str) -> Optional[str]:
    suffix = "" if metric == "auc" else f"_{metric}"
    path = run.metrics_dir / f"probing{suffix}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    agg = data.get("aggregate", {})
    if not agg:
        return None
    key = f"test_{metric}"
    methods = sorted(agg, key=lambda m: -agg[m].get(f"mean_{key}", 0))
    n = agg[methods[0]].get("n_concepts", "?")
    title = "AUC" if metric == "auc" else "F1"
    lines = [f"## Probing — {title} ({n} concepts)", "",
            f"| method | mean test {metric} | 95% CI |", "|---|---|---|"]
    for m in methods:
        a = agg[m]
        bold = lambda x: f"**{x}**" if m == methods[0] else str(x)
        mean_str = bold(f"{a.get(f'mean_{key}', float('nan')):.4f}")
        lines.append(f"| {bold(m)} | {mean_str} | "
                     f"[{a.get('ci_low', float('nan')):.4f}, {a.get('ci_high', float('nan')):.4f}] |")

    comparisons = _paired_comparisons(data["results"], key, methods[0], methods[1:])
    if comparisons:
        lines += ["", "Paired bootstrap on per-concept differences (matched concepts):", "",
                 "| comparison | mean delta | 95% CI | wins | verdict |", "|---|---|---|---|---|"]
        for c in comparisons:
            lines.append(f"| {c['comparison']} | {c['mean']:+.4f} | [{c['lo']:+.4f}, {c['hi']:+.4f}] | "
                         f"{c['wins']}/{c['n']} | {c['verdict']} |")

    cat_table = _category_breakdown(data["results"], key, methods)
    if cat_table:
        lines += ["", "**Per-category** (PLAN §7 level c):", "", cat_table]
    return "\n".join(lines)


def _continuous_probing_section(run: RunLogger) -> Optional[str]:
    """Continuous-property probing (GRAVY, MW, pI, ...) — its OWN section, never
    merged into the discrete AUC/F1 tables above: different label type (Spearman rho
    on a continuous per-protein score, not AUC on a binary per-residue concept) and
    different unit (protein, not residue) — mixing them would silently average two
    incomparable statistics."""
    path = run.metrics_dir / "probing_continuous.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    agg = data.get("aggregate", {})
    if not agg:
        return None
    # Sorted by |rho| (association strength, sign-agnostic) — AUC's plain -mean sort
    # would put a strongly NEGATIVE correlation last, which misrepresents it.
    methods = sorted(agg, key=lambda m: -abs(agg[m].get("mean_test_spearman", 0)))
    n = agg[methods[0]].get("n_properties", "?")
    lines = [f"## Probing — continuous properties ({n} properties, pool={data.get('pool', '?')})", "",
            "| method | mean test ρ | 95% CI |", "|---|---|---|"]
    for m in methods:
        a = agg[m]
        bold = lambda x: f"**{x}**" if m == methods[0] else str(x)
        mean_str = bold(f"{a.get('mean_test_spearman', float('nan')):+.4f}")
        lines.append(f"| {bold(m)} | {mean_str} | "
                     f"[{a.get('ci_low', float('nan')):+.4f}, {a.get('ci_high', float('nan')):+.4f}] |")

    comparisons = _paired_comparisons(data["results"], "test_spearman", methods[0], methods[1:])
    if comparisons:
        lines += ["", "Paired bootstrap on per-property differences (matched properties):", "",
                 "| comparison | mean delta | 95% CI | wins | verdict |", "|---|---|---|---|---|"]
        for c in comparisons:
            lines.append(f"| {c['comparison']} | {c['mean']:+.4f} | [{c['lo']:+.4f}, {c['hi']:+.4f}] | "
                         f"{c['wins']}/{c['n']} | {c['verdict']} |")
    return "\n".join(lines)


def _artifacts_section(run: RunLogger) -> str:
    lines = ["## Artifacts", ""]
    metric_files = sorted(p.name for p in run.metrics_dir.glob("*") if p.is_file())
    if metric_files:
        lines.append("`metrics/`: " + ", ".join(f"`{f}`" for f in metric_files))
    fig_files = sorted(p.name for p in run.figures_dir.glob("*.png"))
    if fig_files:
        lines.append("")
        lines.append(f"`figures/`: {len(fig_files)} figure(s), see `figures/INDEX.md`.")
    return "\n".join(lines)


def write_run_summary(run_dir: str | Path, verbose: bool = True) -> Optional[Path]:
    run = RunLogger(run_dir)
    s = run.load_summary()
    if not s:
        return None

    header = [f"# `{run.dir.name}` — run summary", "",
             "*Auto-generated by `make_report` from `metrics/*.json` — factual tables only. "
             "For narrative interpretation, limitations, and next steps, write those by hand "
             "(see `runs/local_clean/run_summary.md` for a hand-authored example).*", "",
             f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  "
             f"**Git commit:** `{s.get('git_commit', '?')}`" + (" (dirty)" if s.get("git_dirty") else "")]

    sections = [
        _model_training_section(s),
        _generation_section(s, run),
        _probing_section(run, "auc"),
        _probing_section(run, "f1"),
        _continuous_probing_section(run),
    ]
    body = [sec for sec in sections if sec]
    body.append(_artifacts_section(run))

    text = "\n\n---\n\n".join(["\n".join(header)] + body) + "\n"
    out_path = run.dir / "run_summary.md"
    out_path.write_text(text)
    if verbose:
        print(f"[report] run_summary -> {out_path}")
    return out_path
