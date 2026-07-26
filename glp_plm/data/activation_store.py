"""Memmap activation store + standardizer (PLAN §5/§6.1; Mac precache path).

Writes fp16 `(N, d)` activations to a single growable memmap shard on disk and
persists a fitted `Scaler` (`rep_statistics.pt`, `{mean, var}` — docs/INTERPLM_MAP.md).
Streaming producer-consumer (cluster) is a separate W2 concern (see producer_consumer).

Layout under `store_dir/`:
  activations.dat   # fp16 memmap, shape in meta
  meta.json         # {"n": N, "d": d, "dtype": "float16"}
  rep_statistics.pt # {mean, var} from Scaler.save
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch

from glp_plm.diffusion.scaler import Scaler


class ActivationStore:
    def __init__(self, store_dir: str | Path, d: int, dtype: str = "float16"):
        self.dir = Path(store_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.d = d
        self.dtype = dtype
        self.meta_path = self.dir / "meta.json"
        self.dat_path = self.dir / "activations.dat"

    # ---- writing (precache) ----
    def write(self, chunks: Iterable[np.ndarray | torch.Tensor], max_rows: Optional[int] = None) -> int:
        """Append (n_i, d) chunks into one memmap, up to max_rows. Returns N written."""
        rows = []
        total = 0
        for c in chunks:
            c = c.detach().cpu().numpy() if torch.is_tensor(c) else np.asarray(c)
            c = c.reshape(-1, self.d).astype(self.dtype)
            if max_rows is not None and total + len(c) > max_rows:
                c = c[: max_rows - total]
            rows.append(c)
            total += len(c)
            if max_rows is not None and total >= max_rows:
                break
        if not rows:
            raise ValueError("no activations to write")
        arr = np.concatenate(rows, axis=0)
        mm = np.memmap(self.dat_path, dtype=self.dtype, mode="w+", shape=arr.shape)
        mm[:] = arr
        mm.flush()
        self.meta_path.write_text(json.dumps({"n": int(arr.shape[0]), "d": self.d, "dtype": self.dtype}))
        return int(arr.shape[0])

    # ---- reading ----
    def _meta(self) -> dict:
        return json.loads(self.meta_path.read_text())

    def load(self) -> torch.Tensor:
        m = self._meta()
        mm = np.memmap(self.dat_path, dtype=m["dtype"], mode="r", shape=(m["n"], m["d"]))
        return torch.from_numpy(np.array(mm))  # writable copy (avoids non-writable warning)

    def __len__(self) -> int:
        return self._meta()["n"]

    # ---- scaler ----
    def fit_scaler(self, max_sample: int = 500_000) -> Scaler:
        x = self.load()
        if x.shape[0] > max_sample:
            idx = torch.randperm(x.shape[0])[:max_sample]
            x = x[idx]
        scaler = Scaler.fit(x.float())
        scaler.save(self.dir / "rep_statistics.pt")
        return scaler

    def load_scaler(self) -> Scaler:
        return Scaler.load(self.dir / "rep_statistics.pt")
