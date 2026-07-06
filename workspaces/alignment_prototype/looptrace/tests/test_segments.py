#!/usr/bin/env python3
"""Phase-3 gate tests: the landmark Hough + cover DP must reach >= 90%
per-second accuracy (±0.25 s) on the synthetic fixtures across straight /
jump / loop / replay spans and all stretch modes (brief §6 tests)."""

from __future__ import annotations

import numpy as np
import pytest

from workspaces.alignment_prototype.looptrace import fixtures, landmarks
from workspaces.alignment_prototype.looptrace.run import decode_span


@pytest.fixture(scope="module")
def song() -> np.ndarray:
    return fixtures.make_song(90.0, seed=3)


@pytest.fixture(scope="module")
def song_hashes(song):
    return landmarks.audio_points(song)


SLOPES = [1.0 / 1.08, 1.0, 1.0 / 0.93, 0.93, 1.08]


def _gt_song_at(span: fixtures.MixSpan, t: float) -> float:
    segs = sorted(span.segments, key=lambda s: s.mix_start_s)
    for i, s in enumerate(segs):
        end = segs[i + 1].mix_start_s if i + 1 < len(segs) else s.mix_end_s
        if s.mix_start_s <= t <= end:
            frac = (t - s.mix_start_s) / max(s.mix_end_s - s.mix_start_s, 1e-9)
            return s.song_start_s + frac * (s.song_end_s - s.song_start_s)
    s = segs[-1]
    frac = (t - s.mix_start_s) / max(s.mix_end_s - s.mix_start_s, 1e-9)
    return s.song_start_s + frac * (s.song_end_s - s.song_start_s)


def _pred_song_at(segs: list[tuple[float, float, float]], t: float, slope: float):
    if not segs:
        return None
    for i, (m0, s0, s1) in enumerate(segs):
        m1 = segs[i + 1][0] if i + 1 < len(segs) else m0 + (s1 - s0) / max(slope, 1e-9)
        if m0 <= t <= m1:
            return s0 + (t - m0) * slope
    m0, s0, _ = min(segs, key=lambda q: abs(q[0] - t))
    return s0 + (t - m0) * slope


def _per_second_acc(span, segs, tol=0.25) -> float:
    dur = len(span.y) / fixtures.SR
    slope = span.stretch
    ts = np.arange(0.5, dur - 0.5, 1.0)
    ok = 0
    for t in ts:
        p = _pred_song_at(segs, float(t), slope)
        if p is not None and abs(p - _gt_song_at(span, float(t))) < tol:
            ok += 1
    return ok / max(1, len(ts))


def _decode(span, song_hashes):
    segs, _meta = decode_span(
        span.y,
        song_hashes,
        SLOPES,
        song_lags_s=list(span.song_lags_s),
    )
    return segs


def test_per_second_accuracy_battery(song, song_hashes):
    accs = []
    per_case = []
    for kind in ("straight", "jump", "loop2", "loop4", "replay"):
        for mode, factor in (("none", 1.0), ("repitch", 1.08), ("keylock", 0.93)):
            for seed in (1, 2):
                span = fixtures.make_mix_span(
                    song, kind, seed=seed, stretch_factor=factor, stretch_mode=mode
                )
                a = _per_second_acc(span, _decode(span, song_hashes))
                accs.append(a)
                per_case.append((kind, mode, seed, round(a, 2)))
    mean = float(np.mean(accs))
    worst = sorted(per_case, key=lambda c: c[3])[:6]
    assert mean >= 0.90, f"mean per-second acc {mean:.2f}; worst: {worst}"


def test_straight_span_is_one_segment(song, song_hashes):
    span = fixtures.make_mix_span(song, "straight", seed=4)
    segs = _decode(span, song_hashes)
    assert len(segs) == 1
    gt = span.segments[0]
    assert segs[0][1] == pytest.approx(gt.song_start_s, abs=0.3)


def test_jump_span_finds_both_diagonals(song, song_hashes):
    span = fixtures.make_mix_span(song, "jump", seed=6)
    segs = _decode(span, song_hashes)
    assert len(segs) == 2
    g1, g2 = span.segments
    assert segs[0][1] == pytest.approx(g1.song_start_s, abs=0.3)
    assert abs(segs[1][0] - g2.mix_start_s) < 1.5
    assert segs[1][1] == pytest.approx(
        g2.song_start_s + (segs[1][0] - g2.mix_start_s), abs=0.5
    )
