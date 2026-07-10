"""Tests for analysis.grid_repair — beat-grid repair and grid snapping."""

from __future__ import annotations

from analysis.grid_repair import RepairedGrid, repair_beat_grid, snap_to_grid


def _grid(n: int, period: float = 0.5, start: float = 0.0) -> list[float]:
    return [start + k * period for k in range(n)]


class TestRepairBeatGrid:
    def test_clean_grid_is_untouched(self) -> None:
        g = _grid(64)
        r = repair_beat_grid(g)
        assert not r.changed
        assert list(r.times) == g

    def test_true_tempo_drift_survives(self) -> None:
        # 3% gradual accelerando — musically real, must not be "repaired"
        times, t, p = [], 0.0, 0.5
        for _ in range(64):
            times.append(t)
            t += p
            p *= 0.9995
        r = repair_beat_grid(times)
        assert not r.changed

    def test_doublet_removed(self) -> None:
        g = _grid(32)
        g.insert(10, g[10] - 0.05)  # spurious extra beat 50ms early
        r = repair_beat_grid(g)
        assert r.n_removed == 1
        assert r.n_inserted == 0
        assert len(r.times) == 32

    def test_double_time_run_halved(self) -> None:
        # 32 clean beats, then a 12-interval double-time run (6 real beats'
        # worth). The clean majority defines the median period.
        g = _grid(32)
        t = g[-1]
        for _ in range(12):
            t += 0.25
            g.append(t)
        r = repair_beat_grid(g)
        assert r.n_removed == 6
        ivals = [b - a for a, b in zip(r.times, r.times[1:])]
        assert all(i >= 0.5 / 1.5 for i in ivals)

    def test_missed_beat_interpolated(self) -> None:
        g = _grid(32)
        missing = g.pop(10)
        r = repair_beat_grid(g)
        assert r.n_inserted == 1
        assert any(abs(t - missing) < 1e-9 for t in r.times)
        assert len(r.times) == 32

    def test_beatless_breakdown_not_filled(self) -> None:
        # a gap of 10.3 periods — not near an integer multiple: leave it
        first, second = _grid(16), _grid(16, start=15.65)
        r = repair_beat_grid(first + second)
        assert r.n_inserted == 0

    def test_long_exact_gap_filled_evenly(self) -> None:
        g = _grid(32)
        del g[10:13]  # three consecutive beats missing -> 4-period gap
        r = repair_beat_grid(g)
        assert r.n_inserted == 3
        assert len(r.times) == 32

    def test_idempotent(self) -> None:
        g = _grid(32)
        g.insert(5, g[5] - 0.04)
        del g[20]
        once = repair_beat_grid(g)
        twice = repair_beat_grid(once.times)
        assert not twice.changed
        assert twice.times == once.times

    def test_degenerate_inputs(self) -> None:
        assert repair_beat_grid(()) == RepairedGrid((), 0, 0)
        assert repair_beat_grid((1.0, 1.5)).times == (1.0, 1.5)


class TestSnapToGrid:
    def test_snaps_near_marker_to_line(self) -> None:
        grid = _grid(10, period=2.0)
        assert snap_to_grid([4.3], grid) == (4.0,)

    def test_far_marker_left_alone(self) -> None:
        # outside the grid span: nearest line is beyond max_dist
        grid = _grid(10, period=2.0)
        assert snap_to_grid([25.0], grid) == (25.0,)

    def test_explicit_tolerance(self) -> None:
        grid = _grid(10, period=2.0)
        assert snap_to_grid([4.3], grid, max_dist_s=0.2) == (4.3,)

    def test_collision_keeps_raw_second_marker(self) -> None:
        grid = _grid(10, period=2.0)
        out = snap_to_grid([3.8, 4.2], grid)
        assert out == (4.0, 4.2)

    def test_empty_grid_passthrough(self) -> None:
        assert snap_to_grid([1.0, 2.0], []) == (1.0, 2.0)
