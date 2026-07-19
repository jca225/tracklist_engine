"""Corpus-level weak-axis summaries for the gallery header."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from eda.alignment.spectrogram_review.spans import SpanCard


@dataclass(frozen=True)
class AxisStat:
    key: str
    n: int
    mean_traj: float
    sec_lost: float
    success_rate: float


def _agg(cards: Iterable[SpanCard], key_fn) -> list[AxisStat]:
    buckets: dict[str, list[SpanCard]] = defaultdict(list)
    for c in cards:
        buckets[key_fn(c)].append(c)
    out: list[AxisStat] = []
    for key, rows in buckets.items():
        n = len(rows)
        traj = sum(r.traj_strict for r in rows) / n
        lost = sum(r.sec_lost for r in rows)
        ok = sum(1 for r in rows if r.success) / n
        out.append(
            AxisStat(
                key=key,
                n=n,
                mean_traj=traj,
                sec_lost=lost,
                success_rate=ok,
            )
        )
    return sorted(out, key=lambda s: s.sec_lost, reverse=True)


def stem_stats(cards: Sequence[SpanCard]) -> list[AxisStat]:
    return _agg(cards, lambda c: c.gt_stem)


def cause_stats(cards: Sequence[SpanCard]) -> list[AxisStat]:
    fails = [c for c in cards if not c.success]
    return _agg(fails, lambda c: c.cause)


def span_class_stats(cards: Sequence[SpanCard]) -> list[AxisStat]:
    return _agg(cards, lambda c: c.span_class)


def weak_axes_blurb(cards: Sequence[SpanCard], *, top_n: int = 3) -> dict:
    """Compact summary for the HTML header (worst stems / causes / classes)."""
    stems = stem_stats(cards)
    causes = cause_stats(cards)
    classes = span_class_stats(cards)
    return {
        "n": len(cards),
        "n_fail": sum(1 for c in cards if not c.success),
        "n_ok": sum(1 for c in cards if c.success),
        "worst_stems": [
            {
                "key": s.key,
                "n": s.n,
                "mean_traj": round(s.mean_traj, 3),
                "sec_lost": round(s.sec_lost, 1),
                "success_rate": round(s.success_rate, 3),
            }
            for s in stems[:top_n]
        ],
        "worst_causes": [
            {
                "key": s.key,
                "n": s.n,
                "mean_traj": round(s.mean_traj, 3),
                "sec_lost": round(s.sec_lost, 1),
            }
            for s in causes[:top_n]
        ],
        "worst_span_classes": [
            {
                "key": s.key,
                "n": s.n,
                "mean_traj": round(s.mean_traj, 3),
                "sec_lost": round(s.sec_lost, 1),
            }
            for s in classes[:top_n]
        ],
    }
