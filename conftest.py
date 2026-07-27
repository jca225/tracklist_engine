"""Pytest config: make the repo root importable so package imports work
from tests without installing the project.

``eda/`` stays on ``sys.path`` for legacy flat imports inside eda helpers, but
**repo root must win** so top-level ``alignment/`` is not shadowed by
``eda/alignment/``.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
# Insert eda first, then repo — each insert(0) so repo ends up ahead of eda.
for p in (_REPO_ROOT / "eda", _REPO_ROOT):
    sp = str(p)
    if sp in sys.path:
        sys.path.remove(sp)
    sys.path.insert(0, sp)
