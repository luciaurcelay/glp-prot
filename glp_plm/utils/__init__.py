"""Utilities: device selection, seeding, and cluster storage redirection."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path


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


def resolve_storage_dir(output_dir: str | Path, subdir: str, root_env: str, *,
                         symlink: bool = True) -> Path:
    """Resolve `<output_dir>/<subdir>`, redirected to cluster storage when `root_env`
    (e.g. GLP_DATA_ROOT, GLP_RUNS_ROOT) is set — see scripts/cluster/README.md.

    Off-cluster (Mac PoC, `root_env` unset) this is just `<output_dir>/<subdir>`,
    created locally: identical to the pre-cluster behavior, no code-path divergence.

    On cluster, the real directory is `$<root_env>/<basename(output_dir)>/<subdir>`
    — done unconditionally here (not a separate setup step) so the first script that
    touches a given subdir is always the one that redirects it; nothing can skip it
    and silently write large data onto the quota-capped repo filesystem.

    When `symlink` is true (default), `<output_dir>/<subdir>` becomes an idempotent
    symlink to the real directory, so the path stays browsable locally (matches how
    metrics/figures work — small, worth browsing). Pass `symlink=False` to redirect
    without creating that link — used for checkpoints, the activation store, and the
    embedded Swiss-Prot eval shards: all data-sized, kept off the local `runs/` tree
    on purpose and reached directly under `$<root_env>/<run>/` instead.
    """
    output_dir = Path(output_dir)
    root = os.environ.get(root_env)
    if not root:
        d = output_dir / subdir
        d.mkdir(parents=True, exist_ok=True)
        return d

    target = Path(root) / output_dir.name / subdir
    target.mkdir(parents=True, exist_ok=True)
    if not symlink:
        return target

    link = output_dir / subdir
    if link.is_symlink():
        pass  # already redirected (may point elsewhere on purpose)
    elif link.exists():
        print(f"WARNING: {link} exists and is not a symlink — leaving it alone; "
              f"it will NOT be redirected to ${root_env}.", file=sys.stderr)
        return link
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)
    return link


def resolve_config_store_dir(store_dir: str) -> Path:
    """Resolve a config `data.store_dir` (reuse an existing activation store) to a path.

    Absolute path is used as-is; a relative one resolves under $GLP_DATA_ROOT (falling
    back to ./data off-cluster), e.g. "uniref50_500m/train_store". Unlike
    `resolve_shared_dir` this does NOT create the directory: a reused store must already
    exist, so a wrong/missing path should surface as a clear ActivationStore error rather
    than a silently-created empty dir that looks like a store with zero activations.
    """
    p = Path(store_dir)
    if p.is_absolute():
        return p
    return Path(os.environ.get("GLP_DATA_ROOT", "data")) / store_dir


def resolve_shared_dir(subdir: str, root_env: str) -> Path:
    """Resolve a subdir shared across ALL runs — for content that doesn't depend on
    which run downloaded it (raw Swiss-Prot REST pulls, UniRef50), unlike
    `resolve_storage_dir`'s per-run subdirs (activation stores, embedded eval
    shards, checkpoints). No per-run nesting and no local symlink (nothing to
    browse it *as*, since it isn't any one run's).

    Lives at `$<root_env>/<subdir>` when that env var is set (cluster storage
    split — see scripts/cluster/README.md), else `./data/<subdir>` off-cluster.
    """
    root = os.environ.get(root_env, "data")
    d = Path(root) / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d


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
