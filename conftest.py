"""Pytest config: make the repo root importable so `row_tokens`, `audio_pipeline`,
and the `data_analysis` helpers can all be imported from tests without installing
the project as a package.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
for p in (_REPO_ROOT, _REPO_ROOT / "data_analysis"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
