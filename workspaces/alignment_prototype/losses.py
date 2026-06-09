"""Losses for supervised span alignment (P5)."""
from __future__ import annotations

import math

from .records import SpanPrediction, SpanTarget


def huber(x: float, delta: float = 1.0) -> float:
    ax = abs(x)
    if ax <= delta:
        return 0.5 * x * x
    return delta * (ax - 0.5 * delta)


def span_placement_loss(pred: SpanPrediction, target: SpanTarget) -> float:
    return (
        huber(pred.set_start_s - target.set_start_s)
        + huber(pred.set_end_s - target.set_end_s)
        + huber(pred.ref_start_s - target.ref_start_s)
        + huber((pred.ref_end_s or 0.0) - (target.ref_end_s or 0.0))
    )


def identity_ce_loss(
    pred: SpanPrediction,
    target: SpanTarget,
    candidates: tuple[tuple[str, str], ...],
) -> float:
    """Softmax CE over (recording_id, stem) candidates; stub uses hard match."""
    if not target.recording_id:
        return 0.0
    key = (pred.recording_id or "", pred.claimed_stem)
    tgt = (target.recording_id, target.claimed_stem)
    if key == tgt:
        return 0.0
    # Penalize wrong identity; magnitude 1.0 until learned logits exist.
    return 1.0 if candidates else 0.0


def batch_loss(
    preds: tuple[SpanPrediction, ...],
    targets: tuple[SpanTarget, ...],
    *,
    identity_weight: float = 1.0,
) -> float:
    if len(preds) != len(targets):
        raise ValueError(f"pred/target length mismatch: {len(preds)} vs {len(targets)}")
    total = 0.0
    for p, t in zip(preds, targets):
        cands = ((t.recording_id or "", t.claimed_stem),) if t.recording_id else ()
        total += span_placement_loss(p, t) + identity_weight * identity_ce_loss(p, t, cands)
    return total / max(len(targets), 1)
