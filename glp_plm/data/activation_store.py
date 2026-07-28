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
    def write(self, chunks: Iterable[np.ndarray | torch.Tensor], max_rows: int,
              log_every: int = 2_000_000) -> int:
        """Stream (n_i, d) chunks straight into the memmap, up to max_rows.

        Writes incrementally — peak memory is one chunk, NOT the whole store. (An
        earlier version accumulated every chunk in RAM and concatenated, which needs
        ~2x the store size: fine at 2M activations, but it thrashes swap at 25M+.)
        Returns the number of rows actually written.
        """
        itemsize = np.dtype(self.dtype).itemsize
        mm = np.memmap(self.dat_path, dtype=self.dtype, mode="w+", shape=(max_rows, self.d))
        total = 0
        next_log = log_every
        try:
            for c in chunks:
                c = c.detach().cpu().numpy() if torch.is_tensor(c) else np.asarray(c)
                c = c.reshape(-1, self.d)
                if total + len(c) > max_rows:
                    c = c[: max_rows - total]
                if len(c) == 0:
                    break
                mm[total:total + len(c)] = c.astype(self.dtype, copy=False)
                total += len(c)
                if total >= next_log:
                    print(f"[store] {total:,}/{max_rows:,} activations written", flush=True)
                    next_log += log_every
                if total >= max_rows:
                    break
            mm.flush()
        finally:
            del mm                       # release the mapping before truncating
        if total == 0:
            raise ValueError("no activations to write")
        if total < max_rows:             # shrink the file to what we actually wrote
            with open(self.dat_path, "r+b") as f:
                f.truncate(total * self.d * itemsize)
        self.meta_path.write_text(json.dumps({"n": int(total), "d": self.d, "dtype": self.dtype}))
        return int(total)

    # ---- reading ----
    def _meta(self) -> dict:
        return json.loads(self.meta_path.read_text())

    def memmap(self, mode: str = "r") -> np.memmap:
        """Memmap over the store — no RAM copy (the store can exceed RAM).

        mode "r" = read-only; "c" = copy-on-write (writable view, still lazy — pages
        only materialize if written, which we never do).
        """
        m = self._meta()
        return np.memmap(self.dat_path, dtype=m["dtype"], mode=mode, shape=(m["n"], m["d"]))

    def load(self, copy: bool = False) -> torch.Tensor:
        """Torch view over the store.

        Default is a LAZY view sharing the memmap buffer, so a 16 GB store opens on a
        16 GB machine; rows are paged in on access. Do NOT call `.float()` on the
        result — that materializes everything. Convert per batch instead.
        `copy=True` forces a RAM copy (only for stores you know are small).
        """
        if copy:
            return torch.from_numpy(np.array(self.memmap()))
        # copy-on-write mapping: writable-looking (so torch doesn't warn) but lazy
        return torch.from_numpy(np.asarray(self.memmap(mode="c")))

    def __len__(self) -> int:
        return self._meta()["n"]

    # ---- scaler ----
    def fit_scaler(self, max_sample: int = 500_000, seed: int = 0) -> Scaler:
        """Fit per-dim mean/var on a random subsample (only the sample enters RAM)."""
        mm = self.memmap()
        n = mm.shape[0]
        if n > max_sample:
            idx = np.sort(np.random.default_rng(seed).choice(n, max_sample, replace=False))
            sample = np.asarray(mm[idx], dtype=np.float32)
        else:
            sample = np.asarray(mm, dtype=np.float32)
        scaler = Scaler.fit(torch.from_numpy(sample))
        scaler.save(self.dir / "rep_statistics.pt")
        return scaler

    def load_scaler(self) -> Scaler:
        return Scaler.load(self.dir / "rep_statistics.pt")
