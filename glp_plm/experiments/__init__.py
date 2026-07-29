"""Experiment logging + figure generation.

- metrics.RunLogger : write/read structured metric artifacts (the source of truth)
- style             : shared matplotlib style + validated categorical palette
- figures           : figure registry (each declares the artifacts it needs)
- report.make_report: render every figure whose inputs exist (idempotent, additive),
                      then (re)write run_summary.md via summary.write_run_summary
- summary            : factual, tabular run_summary.md from metrics/*.json (no narrative)

Compute logs metrics; plotting reads them. See metrics.py for the layout.
"""

from glp_plm.experiments.metrics import RunLogger
from glp_plm.experiments.report import compare_runs, make_report
from glp_plm.experiments.summary import write_run_summary

__all__ = ["RunLogger", "make_report", "compare_runs", "write_run_summary"]
