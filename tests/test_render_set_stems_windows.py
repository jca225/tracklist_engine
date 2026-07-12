"""Unit tests for render_set_stems overlap-window planning (WS2).

The separator sees each core tile padded by `overlap_sec` of two-sided context
(clamped at the mix edges); only the core is kept and concatenated. These tests
pin the tiling invariants that keep the concatenated stems seamless and
gap-free — the actual seam-healing was validated in workspaces/streaming_mir.
"""

from __future__ import annotations

from scripts.render_set_stems import Window, plan_windows


def _cores_tile_exactly(windows: list[Window], duration: float) -> None:
    assert windows[0].core_start == 0.0
    assert abs(windows[-1].core_end - duration) < 1e-9
    for a, b in zip(windows, windows[1:]):
        assert abs(a.core_end - b.core_start) < 1e-9  # no gap, no overlap in cores
    total = sum(w.core_end - w.core_start for w in windows)
    assert abs(total - duration) < 1e-9  # cores cover the whole mix


def test_cores_tile_and_cover() -> None:
    w = plan_windows(duration=100.0, core_sec=30.0, overlap_sec=10.0)
    assert len(w) == 4  # [0,30] [30,60] [60,90] [90,100]
    _cores_tile_exactly(w, 100.0)


def test_interior_windows_have_two_sided_margin() -> None:
    w = plan_windows(duration=100.0, core_sec=30.0, overlap_sec=10.0)
    mid = w[1]  # core [30,60]
    assert mid.win_start == 20.0  # 30 - 10
    assert mid.win_end == 70.0  # 60 + 10


def test_edges_clamp_to_mix() -> None:
    w = plan_windows(duration=100.0, core_sec=30.0, overlap_sec=10.0)
    assert w[0].win_start == 0.0  # first window can't look left of 0
    assert w[-1].win_end == 100.0  # last window can't look past the end


def test_overlap_zero_reproduces_hard_cut() -> None:
    w = plan_windows(duration=90.0, core_sec=30.0, overlap_sec=0.0)
    for win in w:
        assert win.win_start == win.core_start  # window == core → legacy behaviour
        assert win.win_end == win.core_end


def test_short_mix_single_window() -> None:
    w = plan_windows(duration=12.0, core_sec=360.0, overlap_sec=10.0)
    assert len(w) == 1
    assert w[0] == Window(core_start=0.0, core_end=12.0, win_start=0.0, win_end=12.0)
