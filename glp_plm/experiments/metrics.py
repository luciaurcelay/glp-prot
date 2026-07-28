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
from pathlib import Path
from typing import Any, Dict, List


class RunLogger:
    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir)
        self.metrics_dir = self.dir / "metrics"
        self.figures_dir = self.dir / "figures"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

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
