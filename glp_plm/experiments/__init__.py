"""Experiment logging + figure generation.

- metrics.RunLogger : write/read structured metric artifacts (the source of truth)
- style             : shared matplotlib style + validated categorical palette
- figures           : figure registry (each declares the artifacts it needs)
- report.make_report: render every figure whose inputs exist (idempotent, additive)

Compute logs metrics; plotting reads them. See metrics.py for the layout.
"""

from glp_plm.experiments.metrics import RunLogger
from glp_plm.experiments.report import compare_runs, make_report

__all__ = ["RunLogger", "make_report", "compare_runs"]
