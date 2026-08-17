"""Shared probing statistics (GLProt probing spec Part 5): paired bootstrap over
concepts/properties and per-category means. Used by both `experiments/summary.py`
(run_summary.md tables) and `scripts/make_probing_outputs.py` (probing_summary.json)
so the two outputs can't drift apart — one source of truth for "how do we compare
representations."
"""

from __future__ import annotations

from typing import Any, Dict, List

from glp_plm.eval.probing_1d import bootstrap_ci


def _item_key(r: Dict[str, Any]) -> str:
    """A probing result row is either concept-keyed (discrete) or property-keyed
    (continuous) — never both."""
    return r.get("concept", r.get("property"))


def paired_comparisons(results: List[Dict[str, Any]], key: str, primary: str,
                       others: List[str]) -> List[Dict[str, Any]]:
    """Paired bootstrap on per-item deltas (matched concepts/properties) between
    `primary` and each of `others` (spec Part 5: mean difference, 95% CI, win count,
    e.g. "wins 88/104")."""
    by_item: Dict[str, Dict[str, float]] = {}
    for r in results:
        by_item.setdefault(_item_key(r), {})[r["method"]] = r[key]
    rows = []
    for other in others:
        diffs = [m[primary] - m[other] for m in by_item.values() if primary in m and other in m]
        if not diffs:
            continue
        mean, lo, hi = bootstrap_ci(diffs)
        wins = sum(1 for d in diffs if d > 0)
        verdict = "significant" if (lo > 0 or hi < 0) else "NOT significant"
        rows.append({"comparison": f"{primary} - {other}", "mean_diff": mean, "ci_low": lo,
                     "ci_high": hi, "wins": wins, "n": len(diffs), "verdict": verdict})
    return rows


def category_table(results: List[Dict[str, Any]], key: str, methods: List[str]) -> Dict[str, Dict[str, Any]]:
    """{category: {"n": n_items, method: mean}} — spec Part 5 "per-category means...
    with n per category, since some buckets are small.\""""
    from glp_plm.eval.concept_selection import categorize

    by_cat: Dict[str, Dict[str, List[float]]] = {}
    items_per_cat: Dict[str, set] = {}
    for r in results:
        item = _item_key(r)
        cat = categorize(item)
        if cat is None:
            continue
        by_cat.setdefault(cat, {}).setdefault(r["method"], []).append(r[key])
        items_per_cat.setdefault(cat, set()).add(item)
    table = {}
    for cat in sorted(by_cat, key=lambda c: -len(items_per_cat[c])):
        row: Dict[str, Any] = {"n": len(items_per_cat[cat])}
        for m in methods:
            if m in by_cat[cat]:
                row[m] = sum(by_cat[cat][m]) / len(by_cat[cat][m])
        table[cat] = row
    return table


def category_table_markdown(table: Dict[str, Dict[str, Any]], methods: List[str]) -> str:
    lines = ["| category | n | " + " | ".join(methods) + " |", "|---|---|" + "---|" * len(methods)]
    for cat, row in table.items():
        vals = [f"{row[m]:.3f}" if m in row else "—" for m in methods]
        lines.append(f"| {cat} | {row['n']} | " + " | ".join(vals) + " |")
    return "\n".join(lines)
