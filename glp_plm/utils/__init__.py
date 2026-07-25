"""Utilities: device selection and seeding."""

from __future__ import annotations

import os
import random


def get_device(prefer: str | None = None) -> str:
    """Resolve a device string. `prefer` from config wins if available."""
    import torch

    if prefer and prefer != "auto":
        if prefer == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if prefer == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
