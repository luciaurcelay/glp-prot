"""Make the repo root importable so `import glp_plm` works without installation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "InterPLM"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
