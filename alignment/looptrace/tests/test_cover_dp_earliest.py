"""TDD gate for the earliest-fiber-instance prior inside looptrace's cover_dp.

Looptrace is the canonical acappella decoder (joint_ref_decode --decoder
looptrace). Its cover_dp selects the ref-position instance by DP over candidate
diagonals (each Diagonal.intercept_s = a fiber instance), and NOTES.md records
that "lands on the wrong repeat INSTANCE" is looptrace's dominant recoverable
error. The instance-separability finding shows the GT instance is the EARLIEST
ref-position member ~0.93 of the time. This adds a per-candidate position prior
that breaks a near-tie toward the earliest intercept without overriding a
genuine support gap — the looptrace analogue of path_decode's r0_slope.
"""

from __future__ import annotations

import numpy as np

from alignment.looptrace.segments import Diagonal, cover_dp


def _pts(slope: float, b: float, ts: np.ndarray) -> np.ndarray:
    """Landmark match points (t_mix, t_song) lying exactly on t_song=slope*t+b."""
    return np.column_stack([ts, slope * ts + b])


# Two fiber siblings: early instance at intercept 5 s, late at 100 s. The late
# one carries slightly MORE landmark density (noise), so the position-agnostic
# baseline decodes to it.
_SLOPE = 1.0
_SPAN = 40.0
_EARLY_B = 5.0
_LATE_B = 100.0


def _near_tie_points() -> tuple[np.ndarray, list[Diagonal]]:
    early = _pts(_SLOPE, _EARLY_B, np.arange(0.0, _SPAN, 1.0))  # 40 pts
    late = _pts(_SLOPE, _LATE_B, np.arange(0.0, _SPAN, 0.8))  # 50 pts (denser)
    pts = np.vstack([early, late])
    diags = [Diagonal(_EARLY_B, 40), Diagonal(_LATE_B, 50)]
    return pts, diags


def _lopsided_points() -> tuple[np.ndarray, list[Diagonal]]:
    """Late instance is genuinely much stronger (not a fiber tie)."""
    early = _pts(_SLOPE, _EARLY_B, np.arange(0.0, _SPAN, 4.0))  # 10 pts
    late = _pts(_SLOPE, _LATE_B, np.arange(0.0, _SPAN, 0.67))  # ~60 pts
    pts = np.vstack([early, late])
    diags = [Diagonal(_EARLY_B, 10), Diagonal(_LATE_B, 60)]
    return pts, diags


def _decoded_intercept(segs: list[tuple[float, float, float]]) -> float:
    """Ref intercept of the longest decoded segment (song_start - slope*mix_start)."""
    assert segs, "expected at least one segment"
    seg = max(segs, key=lambda s: s[2] - s[1])
    return seg[1] - _SLOPE * seg[0]


def test_default_decodes_the_denser_late_instance():
    pts, diags = _near_tie_points()
    segs, _ev = cover_dp(pts, _SLOPE, _SPAN, diagonals=diags)
    assert abs(_decoded_intercept(segs) - _LATE_B) < 1.0


def test_earliest_prior_flips_near_tie_to_earliest_instance():
    pts, diags = _near_tie_points()
    segs, _ev = cover_dp(pts, _SLOPE, _SPAN, diagonals=diags, earliest_slope=0.005)
    assert abs(_decoded_intercept(segs) - _EARLY_B) < 1.0


def test_earliest_prior_does_not_override_a_real_support_gap():
    pts, diags = _lopsided_points()
    segs, _ev = cover_dp(pts, _SLOPE, _SPAN, diagonals=diags, earliest_slope=0.005)
    assert abs(_decoded_intercept(segs) - _LATE_B) < 1.0


def test_earliest_slope_zero_is_exact_baseline():
    pts, diags = _near_tie_points()
    base = cover_dp(pts, _SLOPE, _SPAN, diagonals=diags)
    zero = cover_dp(pts, _SLOPE, _SPAN, diagonals=diags, earliest_slope=0.0)
    assert base[0] == zero[0]
    assert base[1] == zero[1]
