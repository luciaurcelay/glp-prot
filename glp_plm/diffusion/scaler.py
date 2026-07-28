"""Per-dimension standardizer for activations (PLAN §6.2/§6.1; docs/INTERPLM_MAP.md).

GLP standardizes activations to zero-mean/unit-var before diffusion and inverts
after sampling. The released reference (denoiser.py:19-71) persists `{mean, var}`
and uses (x - mean)/sqrt(var). We store BOTH mean and var (so GLP-released
`rep_statistics.pt` can be loaded directly) and expose std = sqrt(var).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class Scaler(nn.Module):
    def __init__(self, mean: torch.Tensor, var: torch.Tensor, eps: float = 1e-8):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("var", var)
        self.eps = eps

    @property
    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var + self.eps)

    def standardize(self, x: torch.Tensor) -> torch.Tensor:
        return (x.to(self.mean.device) - self.mean) / self.std

    def destandardize(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.mean.device) * self.std + self.mean

    # ---- fit / io ----
    @classmethod
    def fit(cls, x: torch.Tensor) -> "Scaler":
        """Fit per-dim mean/var over a sample of shape (N, d)."""
        x = x.reshape(-1, x.shape[-1]).float()
        return cls(mean=x.mean(dim=0), var=x.var(dim=0, unbiased=False))

    def save(self, path: str | Path) -> None:
        # same key layout as the GLP reference (rep_statistics.pt)
        torch.save({"mean": self.mean, "var": self.var}, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "Scaler":
        d = torch.load(str(path), map_location="cpu")
        return cls(mean=d["mean"], var=d["var"])
