#!/usr/bin/env python3
"""Phase-2 gate tests: DJ loop/replay detection precision/recall >= 95% on
the synthetic battery (straight / 2x / 4x loops / replays / jumps / natural
distinct-take repeats, x none / repitch / key-locked stretch, x clean /
-6 dB per-iteration bleed), and collapse/expand round-trips."""

from __future__ import annotations

import numpy as np
import pytest

from alignment.looptrace import fixtures, loops
from alignment.looptrace.selfsim import SR


@pytest.fixture(scope="module")
def song() -> np.ndarray:
    return fixtures.make_song(90.0, seed=3)


def _detect(span: fixtures.MixSpan) -> list[loops.Loop]:
    return loops.detect_loops(
        span.y, song_lags_s=list(span.song_lags_s), slope=span.stretch
    )


def _battery(song):
    cases = []
    for kind in ("straight", "loop2", "loop4", "replay", "jump", "natural_repeat"):
        for mode, factor in (("none", 1.0), ("repitch", 1.08), ("keylock", 0.93)):
            for seed in (1, 2):
                for bleed in (None, -6.0):
                    cases.append(
                        fixtures.make_mix_span(
                            song,
                            kind,
                            seed=seed,
                            stretch_factor=factor,
                            stretch_mode=mode,
                            bleed_db=bleed,
                        )
                    )
    return cases


def _match(det: loops.Loop, gt: fixtures.GTLoop) -> bool:
    period_ok = abs(det.period_s - gt.period_s) / gt.period_s < 0.05
    ov = min(det.end_s, gt.end_s) - max(det.source_start_s, gt.start_s)
    return period_ok and ov / (gt.end_s - gt.start_s) > 0.5 and det.n_iter == gt.n_iter


def test_loop_detection_precision_recall(song):
    tp = fp = fn = 0
    misses, fps = [], []
    for span in _battery(song):
        det = _detect(span)
        gts = list(span.loops)
        used = set()
        for d in det:
            hit = next(
                (i for i, g in enumerate(gts) if i not in used and _match(d, g)), None
            )
            if hit is None:
                fp += 1
                fps.append((span.mode, d.period_s, d.n_iter))
            else:
                used.add(hit)
                tp += 1
        fn += len(gts) - len(used)
        if len(used) < len(gts):
            misses.append((span.mode, [(g.period_s, g.n_iter) for g in gts]))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    assert precision >= 0.95, f"precision {precision:.2f} (fp={fps})"
    assert recall >= 0.95, f"recall {recall:.2f} (miss={misses})"


def test_no_loops_on_non_loop_kinds(song):
    for kind in ("straight", "jump", "natural_repeat"):
        span = fixtures.make_mix_span(song, kind, seed=5)
        assert _detect(span) == []


@pytest.mark.xfail(
    reason="live-hardware order (tile, THEN key-locked master-tempo vocoder) "
    "defeats waveform cloneness; spectral-magnitude + drift verification "
    "measured unable to separate it from natural distinct-take repeats. "
    "The GT corpora are Ableton-produced (copies AFTER the stretch).",
    strict=True,
)
def test_keylock_after_tiling_is_a_known_gap(song):
    rng = np.random.default_rng(11)
    dur = len(song) / SR
    period, pre, post = 3.75, 6.0, 8.0
    t0 = rng.uniform(0, dur - (pre + period + post) - 1)

    def cut(a, ln):
        return song[int(a * SR) : int((a + ln) * SR)]

    y = np.concatenate(
        [cut(t0, pre), np.tile(cut(t0 + pre, period), 4), cut(t0 + pre + period, post)]
    )
    y = fixtures._stretch(y, 0.93, "keylock")
    det = loops.detect_loops(y)
    assert any(abs(d.period_s - period * 0.93) / (period * 0.93) < 0.05 for d in det)


def test_collapse_roundtrip(song):
    span = fixtures.make_mix_span(song, "loop4", seed=7)
    det = _detect(span)
    assert len(det) == 1
    lo = det[0]
    cs = loops.collapse(span.y, det)
    dur = len(span.y) / SR
    assert cs.collapsed_duration_s == pytest.approx(
        dur - (lo.n_iter - 1) * lo.period_s, abs=0.05
    )
    # to_original(to_collapsed(t)) folds any iteration onto the canonical one
    t_iter3 = lo.source_start_s + 2 * lo.period_s + 0.5
    assert cs.to_original(cs.to_collapsed(t_iter3)) == pytest.approx(
        lo.source_start_s + 0.5, abs=0.02
    )
    # outside the loop the map is a pure shift
    t_after = lo.end_s + 1.0
    assert cs.to_original(cs.to_collapsed(t_after)) == pytest.approx(t_after, abs=0.02)


def test_replay_collapse(song):
    span = fixtures.make_mix_span(song, "replay", seed=4)
    det = _detect(span)
    assert len(det) == 1
    lo = det[0]
    gt = span.loops[0]
    assert lo.period_s == pytest.approx(gt.period_s, rel=0.05)
    cs = loops.collapse(span.y, det)
    # the truncated second pass is dropped; only the first block remains
    dur = len(span.y) / SR
    assert cs.collapsed_duration_s < dur - 0.5 * (dur - gt.period_s)


def test_expand_segments_replicates_loop(song):
    span = fixtures.make_mix_span(song, "loop2", seed=9)
    det = _detect(span)
    assert len(det) == 1
    lo = det[0]
    cs = loops.collapse(span.y, det)
    seg_c = (cs.to_collapsed(lo.source_start_s), 100.0, 100.0 + lo.period_s)
    out = loops.expand_segments([seg_c], cs)
    # the source piece is replicated over all iterations (same song start —
    # the hard constraint); a trailing extension piece past the loop is
    # allowed (next-start convention extends the last segment to span end)
    reps = [(m, s0) for m, s0, _ in out if s0 == 100.0]
    assert len(reps) == lo.n_iter
    assert reps[0][0] == pytest.approx(lo.source_start_s, abs=0.02)
    assert reps[1][0] == pytest.approx(lo.source_start_s + lo.period_s, abs=0.02)
