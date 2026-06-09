"""Baseline span aligners — sanity checks until MERT features are wired."""
from __future__ import annotations

from dataclasses import dataclass

from .records import SpanPrediction, SpanTarget


class CopyGTBaseline:
    """Trivial baseline: predict targets unchanged (sanity-checks loss = 0)."""

    def predict(self, targets: tuple[SpanTarget, ...]) -> tuple[SpanPrediction, ...]:
        return tuple(
            SpanPrediction(
                slot_label=t.slot_label,
                recording_id=t.recording_id,
                claimed_stem=t.claimed_stem,
                set_start_s=t.set_start_s,
                set_end_s=t.set_end_s,
                ref_start_s=t.ref_start_s,
                ref_end_s=t.ref_end_s,
                confidence=1.0,
            )
            for t in targets
        )


@dataclass(frozen=True)
class _SpanStats:
    set_start_s: float
    set_end_s: float
    ref_start_s: float
    ref_end_s: float


def _mean_span_stats(targets: tuple[SpanTarget, ...]) -> _SpanStats:
    if not targets:
        return _SpanStats(0.0, 0.0, 0.0, 0.0)
    n = len(targets)
    return _SpanStats(
        set_start_s=sum(t.set_start_s for t in targets) / n,
        set_end_s=sum(t.set_end_s for t in targets) / n,
        ref_start_s=sum(t.ref_start_s for t in targets) / n,
        ref_end_s=sum((t.ref_end_s or 0.0) for t in targets) / n,
    )


class MeanSpanBaseline:
    """Predict corpus-mean spans; keeps GT identity (oracle-id placement ablation)."""

    def __init__(self, train_targets: tuple[SpanTarget, ...]) -> None:
        self._stats = _mean_span_stats(train_targets)

    def predict(self, targets: tuple[SpanTarget, ...]) -> tuple[SpanPrediction, ...]:
        s = self._stats
        return tuple(
            SpanPrediction(
                slot_label=t.slot_label,
                recording_id=t.recording_id,
                claimed_stem=t.claimed_stem,
                set_start_s=s.set_start_s,
                set_end_s=s.set_end_s,
                ref_start_s=s.ref_start_s,
                ref_end_s=s.ref_end_s,
                confidence=0.0,
            )
            for t in targets
        )


class NullIdentityBaseline:
    """Oracle spans + wrong identity — isolates identity CE from placement."""

    def predict(self, targets: tuple[SpanTarget, ...]) -> tuple[SpanPrediction, ...]:
        return tuple(
            SpanPrediction(
                slot_label=t.slot_label,
                recording_id=None,
                claimed_stem="regular",
                set_start_s=t.set_start_s,
                set_end_s=t.set_end_s,
                ref_start_s=t.ref_start_s,
                ref_end_s=t.ref_end_s,
                confidence=0.0,
            )
            for t in targets
        )
