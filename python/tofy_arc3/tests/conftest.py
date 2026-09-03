"""Make ``tofy_arc3`` importable when pytest is run from the repo root."""

import sys
from pathlib import Path

PYTHON_DIR = str(Path(__file__).resolve().parents[2])
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)
