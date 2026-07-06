"""Property-based tests for the .als timeline semantics (Hypothesis).

Each property is a law from docs/als_interpreter_plan.md §4: warp maps are
monotonic, the tempo integral matches numeric quadrature, audibility
invariants hold on arbitrary gain curves, envelope interpolation stays inside
the curve's value range, and clip splitting tiles the original span.
"""

from __future__ import annotations

from hypothesis import given, settings, strategies as st

from labeling.als.models import MixClipSpan, ParsedClip, WarpMarkers
from labeling.als.semantics import (
    ArrangementMapper,
    audible_from_curve,
    envelope_value,
    split_clip_at_mix_span_edges,
    tempo_beat_to_sec,
    tempo_sec_to_beat,
)

_finite = dict(allow_nan=False, allow_infinity=False)


@st.composite
def warp_markers(draw):
    """Monotonic warp markers: strictly increasing beats, non-decreasing secs."""
    n = draw(st.integers(2, 10))
    beat = draw(st.floats(-50.0, 50.0, **_finite))
    sec = draw(st.floats(0.0, 100.0, **_finite))
    pts = []
    for _ in range(n):
        pts.append((beat, sec))
        beat += draw(st.floats(0.25, 64.0, **_finite))
        sec += draw(st.floats(0.0, 32.0, **_finite))
    return WarpMarkers(points=tuple(pts))


@st.composite
def tempo_curve(draw):
    """Piecewise-linear BPM curve: increasing beats, BPM in a sane DJ range."""
    n = draw(st.integers(1, 8))
    beat = 0.0
    pts = []
    for _ in range(n):
        pts.append((beat, draw(st.floats(60.0, 200.0, **_finite))))
        beat += draw(st.floats(1.0, 128.0, **_finite))
    return tuple(pts)


@st.composite
def gain_curve(draw):
    n = draw(st.integers(2, 12))
    x = draw(st.floats(-100.0, 100.0, **_finite))
    pts = []
    for _ in range(n):
        pts.append((x, draw(st.floats(0.0, 2.0, **_finite))))
        x += draw(st.floats(0.01, 50.0, **_finite))
    return pts


@settings(deadline=None)
@given(
    warp_markers(),
    st.floats(-100.0, 500.0, **_finite),
    st.floats(0.0, 100.0, **_finite),
)
def test_beat_to_sec_monotonic(warp, beat, delta):
    assert warp.beat_to_sec(beat + delta) >= warp.beat_to_sec(beat) - 1e-9


def _numeric_sec(pts, beat, n=4000):
    """Trapezoid quadrature of 60/bpm — independent reference implementation."""
    if beat <= 0:
        return beat * 60.0 / pts[0][1]

    def bpm_at(x: float) -> float:
        if x <= pts[0][0]:
            return pts[0][1]
        for (b0, v0), (b1, v1) in zip(pts, pts[1:]):
            if b0 <= x <= b1:
                return v0 if b1 == b0 else v0 + (x - b0) / (b1 - b0) * (v1 - v0)
        return pts[-1][1]

    total = 0.0
    step = beat / n
    for i in range(n):
        x0, x1 = i * step, (i + 1) * step
        total += step * 0.5 * (60.0 / bpm_at(x0) + 60.0 / bpm_at(x1))
    return total


@settings(deadline=None, max_examples=60)
@given(tempo_curve(), st.floats(0.0, 800.0, **_finite))
def test_tempo_integral_matches_quadrature(pts, beat):
    exact = tempo_beat_to_sec(pts, beat)
    approx = _numeric_sec(pts, beat)
    assert abs(exact - approx) <= max(5e-3, 5e-3 * abs(exact))


@settings(deadline=None, max_examples=80)
@given(tempo_curve(), st.floats(0.0, 800.0, **_finite))
def test_tempo_sec_to_beat_right_inverts(pts, beat):
    # beat→sec is non-injective across zero-width steps, so assert the
    # right-inverse law sec(inverse(sec)) == sec rather than beat recovery
    sec = tempo_beat_to_sec(pts, beat)
    back = tempo_sec_to_beat(pts, sec)
    assert abs(tempo_beat_to_sec(pts, back) - sec) <= max(1e-6, 1e-6 * abs(sec))


@settings(deadline=None, max_examples=60)
@given(
    tempo_curve(), st.floats(0.0, 2000.0, **_finite), st.floats(0.0, 100.0, **_finite)
)
def test_tempo_sec_to_beat_monotonic(pts, sec, delta):
    assert tempo_sec_to_beat(pts, sec + delta) >= tempo_sec_to_beat(pts, sec) - 1e-9


def test_tempo_sec_to_beat_song_boundary_placement():
    # The seeder scenario that was botched Jun-16: 120 BPM for the first 30
    # mix-seconds, then a step to 90. The tempo step must be WRITTEN at beat
    # 30·120/60 = 60 (not at beat 30), and a song starting 40 s later lands at
    # beat 60 + 40·90/60 = 120.
    pts = ((0.0, 120.0), (60.0, 120.0), (60.0, 90.0))
    assert abs(tempo_sec_to_beat(pts, 30.0) - 60.0) < 1e-9
    assert abs(tempo_sec_to_beat(pts, 70.0) - 120.0) < 1e-9
    assert abs(tempo_beat_to_sec(pts, 120.0) - 70.0) < 1e-9


@settings(deadline=None)
@given(gain_curve(), st.floats(0.0, 0.5, **_finite))
def test_audible_from_curve_invariants(curve, thr):
    frac, start, end = audible_from_curve(curve, thr=thr)
    assert 0.0 <= frac <= 1.0
    if frac == 0.0:
        assert start is None and end is None
    else:
        assert start is not None and end is not None
        assert start <= end + 1e-9
        lo, hi = curve[0][0], curve[-1][0]
        assert lo - 1e-6 <= start and end <= hi + 1e-6


@settings(deadline=None)
@given(gain_curve(), st.floats(-200.0, 300.0, **_finite))
def test_envelope_value_stays_in_range(curve, x):
    vals = [v for _, v in curve]
    assert min(vals) - 1e-9 <= envelope_value(curve, x) <= max(vals) + 1e-9


def _clip(arr_start: float, arr_end: float) -> ParsedClip:
    return ParsedClip(
        group_name="g",
        track_name="t",
        path="/stems/001__A/vocals.flac",
        arr_start=arr_start,
        arr_end=arr_end,
        loop_start=0.0,
        loop_end=arr_end - arr_start,
        pitch_coarse=0,
        pitch_fine=0,
        warp=WarpMarkers(points=((0.0, 0.0), (1.0, 1.0))),
    )


_IDENTITY_WARP = WarpMarkers(points=((0.0, 0.0), (1.0, 1.0)))


def test_split_is_identity_on_monotonic_mapper():
    mapper = ArrangementMapper(
        spans=(MixClipSpan(0.0, 64.0, 0.0, _IDENTITY_WARP),), mix_duration_s=64.0
    )
    clip = _clip(16.0, 48.0)
    assert split_clip_at_mix_span_edges(clip, mapper) == (clip,)


def test_split_tiles_span_at_mix_splice():
    # second 1-mix span restarts mix-seconds at 0 → jump-back at arr beat 32
    mapper = ArrangementMapper(
        spans=(
            MixClipSpan(0.0, 32.0, 0.0, _IDENTITY_WARP),
            MixClipSpan(32.0, 64.0, 0.0, _IDENTITY_WARP),
        ),
        mix_duration_s=64.0,
    )
    # NOTE endpoints must straddle the jump-back visibly: the splice detector
    # compares endpoint seconds, so a clip like [16, 48] (sec 16 → 16) looks
    # monotonic end-to-end and is deliberately left unsplit.
    clip = _clip(20.0, 44.0)
    parts = split_clip_at_mix_span_edges(clip, mapper)
    assert len(parts) == 2
    assert parts[0].arr_start == 20.0 and parts[1].arr_end == 44.0
    assert abs(parts[0].arr_end - 32.0) < 0.01 and abs(parts[1].arr_start - 32.0) < 0.01
    # content offset follows the cut: loop_start advances by the trimmed amount
    assert (
        abs(parts[1].loop_start - (clip.loop_start + (parts[1].arr_start - 20.0)))
        < 1e-6
    )
    # each part is monotonic in mix-seconds
    for p in parts:
        lo = mapper.arr_to_set_sec(p.arr_start + 1e-6)
        hi = mapper.arr_to_set_sec(p.arr_end - 1e-6)
        assert lo is not None and hi is not None and hi >= lo
