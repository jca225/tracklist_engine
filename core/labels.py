"""Generic display-label token utilities (substrate; stdlib-only)."""

from __future__ import annotations

import re


def labels_overlap(left: str, right: str, *, min_tokens: int = 2) -> bool:
    """True when two display labels share enough distinctive tokens."""

    def _tokens(label: str) -> set[str]:
        cleaned = re.sub(r"[^\w\s]", " ", label.lower())
        return {w for w in cleaned.split() if len(w) > 2}

    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return False
    shared = a & b
    if len(shared) >= min_tokens:
        return True
    shorter = min(len(a), len(b))
    return shorter > 0 and len(shared) / shorter >= 0.4
