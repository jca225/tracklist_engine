#!/usr/bin/env python3
"""Set tempo curve + seconds<->beats clock for tempo-automated .als seeding.

With real tempo automation the arrangement's musical beats are no longer equal
to seconds (the 60-BPM "1 beat = 1 s" trick is gone). A clip at set-time t must
be placed at beat = integral of BPM(tau)/60 over [0, t]. For a piecewise-constant
tempo curve that integral is a cumulative sum, so SetClock precomputes the beat
position at each segment boundary and interpolates within a segment.

The curve itself comes from the bed spans' per-track BPM (annotator [NNNbpm] tags
x tempo_ratio), boundaries at song transitions — see seed_tempo_test.py, which
shares this module. A DJ set holds one BPM per track then steps at a transition.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass


@dataclass(frozen=True)
class SetClock:
    """Maps set-time seconds <-> arrangement beats for a piecewise-constant tempo.

    `segs` are (start_s, end_s, bpm), contiguous and sorted. `beat_at` holds the
    cumulative beat count at each segment's start_s; total length closes the last
    segment. Outside the covered range we extrapolate at the edge segment's tempo.
    """
    starts: tuple[float, ...]
    ends: tuple[float, ...]
    bpms: tuple[float, ...]
    beat_at: tuple[float, ...]  # beats accrued at each segment start

    @classmethod
    def from_segments(cls, segs: list[tuple[float, float, float]]) -> "SetClock":
        if not segs:
            raise ValueError("empty tempo curve")
        segs = sorted(segs, key=lambda s: s[0])
        starts, ends, bpms, beat_at, acc = [], [], [], [], 0.0
        for (s, e, b) in segs:
            starts.append(float(s))
            ends.append(float(e))
            bpms.append(float(b))
            beat_at.append(acc)
            acc += (float(e) - float(s)) * float(b) / 60.0
        return cls(tuple(starts), tuple(ends), tuple(bpms), tuple(beat_at))

    @property
    def total_beats(self) -> float:
        return self.beat_at[-1] + (self.ends[-1] - self.starts[-1]) * self.bpms[-1] / 60.0

    def beats(self, t: float) -> float:
        """Arrangement beat at set-time `t` seconds."""
        if t <= self.starts[0]:
            return (t - self.starts[0]) * self.bpms[0] / 60.0
        i = bisect_right(self.starts, t) - 1
        i = min(i, len(self.starts) - 1)
        return self.beat_at[i] + (t - self.starts[i]) * self.bpms[i] / 60.0

    def sec(self, beat: float) -> float:
        """Inverse: set-time seconds at arrangement `beat`."""
        if beat <= 0.0:
            return self.starts[0] + beat * 60.0 / self.bpms[0]
        i = bisect_right(self.beat_at, beat) - 1
        i = min(max(i, 0), len(self.beat_at) - 1)
        return self.starts[i] + (beat - self.beat_at[i]) * 60.0 / self.bpms[i]


def _selftest() -> None:
    # 3 segments: 120 bpm for 10s, 60 bpm for 10s, 140 bpm for 5s
    segs = [(0.0, 10.0, 120.0), (10.0, 20.0, 60.0), (20.0, 25.0, 140.0)]
    c = SetClock.from_segments(segs)
    # 120 bpm * 10s = 20 beats; +60bpm*10s=10 beats; +140bpm*5s=11.667
    assert abs(c.beats(10.0) - 20.0) < 1e-9, c.beats(10.0)
    assert abs(c.beats(20.0) - 30.0) < 1e-9, c.beats(20.0)
    assert abs(c.total_beats - (20.0 + 10.0 + 35.0 / 3.0)) < 1e-9, c.total_beats
    # round-trip seconds -> beats -> seconds at several points
    for t in (0.0, 3.3, 10.0, 15.7, 20.0, 24.9):
        assert abs(c.sec(c.beats(t)) - t) < 1e-6, (t, c.sec(c.beats(t)))
    # monotonic
    bs = [c.beats(t) for t in range(0, 25)]
    assert all(b2 > b1 for b1, b2 in zip(bs, bs[1:]))
    print("SetClock self-test OK:",
          f"beats(10)={c.beats(10.0):.3f} beats(25)={c.total_beats:.3f}")


if __name__ == "__main__":
    _selftest()
