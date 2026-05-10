from __future__ import annotations

import os
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PHASE1_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = PACKAGE_ROOT.parents[2]
DINFER_PYTHON_ROOT = REPO_ROOT / "dInfer" / "python"

os.environ.setdefault("MPLCONFIGDIR", str(PHASE1_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PHASE1_ROOT / ".cache"))

for path in (REPO_ROOT, DINFER_PYTHON_ROOT, PHASE1_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

