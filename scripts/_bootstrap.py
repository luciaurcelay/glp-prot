"""Put the repo root and the InterPLM submodule on sys.path.

Imported first by every real script so `python scripts/x.py` works regardless of
cwd / site-machinery quirks (Homebrew's Python ships a sitecustomize that shadows
venv site hooks, and editable .pth handling proved unreliable here).
"""
import pathlib
import sys

_root = pathlib.Path(__file__).resolve().parent.parent
for _p in (_root, _root / "InterPLM"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
