"""Deterministic train/eval split for span supervision (P5 checklist)."""
from __future__ import annotations

import hashlib

from .records import SpanTarget


def _slot_bucket(slot_label: str) -> int:
    """Stable 0..99 bucket from slot label (holds w-slot layers together-ish)."""
    base = slot_label.split("w", 1)[0] if "w" in slot_label else slot_label
    digest = hashlib.sha256(base.encode()).hexdigest()
    return int(digest[:8], 16) % 100


def split_targets(
    targets: tuple[SpanTarget, ...],
    *,
    eval_fraction: float = 0.2,
) -> tuple[tuple[SpanTarget, ...], tuple[SpanTarget, ...]]:
    """Split by base slot number so mashup w-layers stay in the same fold."""
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError(f"eval_fraction must be in (0, 1), got {eval_fraction}")
    cutoff = int(round(100 * (1.0 - eval_fraction)))
    train: list[SpanTarget] = []
    eval_: list[SpanTarget] = []
    for t in targets:
        if _slot_bucket(t.slot_label) < cutoff:
            train.append(t)
        else:
            eval_.append(t)
    return tuple(train), tuple(eval_)
