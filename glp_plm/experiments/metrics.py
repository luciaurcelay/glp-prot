"""Run-level metric logging — the source of truth that figures read from.

Design (built to scale across W1→W4): compute code (trainer, eval) writes structured
artifacts under `runs/<name>/metrics/`; the plotting layer (experiments.figures) only
*reads* these — it never recomputes. So expensive runs are logged once and figures can
be re-rendered/re-styled freely, and each pipeline phase just drops another metric file.

Artifacts:
  metrics/<phase>.jsonl   append-only scalar series, one JSON record per line
  metrics/summary.json    merged dict of final/summary scalars (fd, auc, %LR, …)
  metrics/*.npz           bulk arrays a figure needs (e.g. PCA projections)
  figures/                rendered PNAG/PDF (see experiments.report)
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from glp_plm.utils import resolve_storage_dir


def _git_provenance() -> Dict[str, Any]:
    """Best-effort {git_commit, git_dirty} for summary.json; {} outside a git repo."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip())
        return {"git_commit": commit, "git_dirty": dirty}
    except Exception:
        return {}


class RunLogger:
    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        # Redirected to $GLP_RUNS_ROOT on cluster (symlinked back under run_dir so
        # they stay browsable there); plain local dirs off-cluster (Mac PoC).
        self.metrics_dir = resolve_storage_dir(self.dir, "metrics", "GLP_RUNS_ROOT")
        self.figures_dir = resolve_storage_dir(self.dir, "figures", "GLP_RUNS_ROOT")
        # Stamped on every construction (each pipeline script makes its own RunLogger),
        # so summary.json reflects the commit as of the most recent stage that touched
        # this run — provenance for "which code produced these numbers" (2026-07-28).
        prov = _git_provenance()
        if prov:
            self.log_summary(**prov)

    # ---- scalar time-series ----
    def series_path(self, phase: str) -> Path:
        return self.metrics_dir / f"{phase}.jsonl"

    def log_step(self, phase: str, record: Dict[str, Any]) -> None:
        with open(self.series_path(phase), "a") as f:
            f.write(json.dumps(record) + "\n")

    def load_series(self, phase: str) -> List[Dict[str, Any]]:
        p = self.series_path(phase)
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    def reset_series(self, phase: str) -> None:
        self.series_path(phase).unlink(missing_ok=True)

    # ---- summary scalars (merged) ----
    @property
    def summary_path(self) -> Path:
        return self.metrics_dir / "summary.json"

    def log_summary(self, **kv: Any) -> None:
        s = self.load_summary()
        s.update(kv)
        self.summary_path.write_text(json.dumps(s, indent=2))

    def load_summary(self) -> Dict[str, Any]:
        if not self.summary_path.exists():
            return {}
        return json.loads(self.summary_path.read_text())

    # ---- bulk arrays ----
    def save_arrays(self, name: str, **arrays) -> Path:
        import numpy as np

        path = self.metrics_dir / f"{name}.npz"
        np.savez_compressed(path, **arrays)
        return path
