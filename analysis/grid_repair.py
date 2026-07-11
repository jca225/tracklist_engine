"""Beat-grid repair + grid snapping — detect-then-correct for tracker flips.

beat_this raw output carries two artifact classes that poison every
bar-indexed consumer downstream (measure grids, per-measure MERT pooling,
placement lattices, per-measure BPM):

  * INSERTED beats — a run of half-length intervals where the tracker locked
    onto double time (or emitted a doublet). Fix: drop beats that arrive too
    soon after the last kept beat.
  * MISSED beats — an interval close to an integer multiple of the local
    period. Fix: interpolate the missing beats evenly.

Measured on the BB11-era sample (eda/alignment/placement_structure): 29% of
raw grids had >5% of intervals in these classes. Repair is intentionally
conservative: it corrects against the grid's own MEDIAN period, so a global
half/double-time lock (every interval consistent, just at the wrong
metrical level) is out of scope — that is a metrical-level choice, not an
artifact, and folding it would fight legitimate genre tempo conventions.

True tempo drift survives repair untouched: the tolerance band (factor 1.5)
admits any musically plausible local tempo movement.

Pure functions, stdlib only.
"""

from __future__ import annotations

import bisect
import statistics
from dataclasses import dataclass

GRID_REPAIR_VERSION = "grid_repair_v1"

_TOLERANCE_FACTOR = 1.5  # interval within [ref/1.5, ref*1.5] = plausible


@dataclass(frozen=True)
class RepairedGrid:
    times: tuple[float, ...]
    n_removed: int  # spurious beats dropped
    n_inserted: int  # missed beats interpolated

    @property
    def changed(self) -> bool:
        return bool(self.n_removed or self.n_inserted)


def repair_beat_grid(times: tuple[float, ...] | list[float]) -> RepairedGrid:
    """Remove doubled beats, interpolate missed ones, against the median period.

    Works for beat grids and downbeat/measure grids alike — each carries its
    own median period. Idempotent: repairing a repaired grid is a no-op.
    """
    ts = [float(t) for t in times]
    if len(ts) < 4:
        return RepairedGrid(times=tuple(ts), n_removed=0, n_inserted=0)
    ref = statistics.median(b - a for a, b in zip(ts, ts[1:]))
    if ref <= 0:
        return RepairedGrid(times=tuple(ts), n_removed=0, n_inserted=0)

    # Pass 1 — drop beats arriving sooner than ref/1.5 after the last kept
    # beat (handles doublets and double-time runs: every other beat survives).
    kept = [ts[0]]
    removed = 0
    for t in ts[1:]:
        if t - kept[-1] < ref / _TOLERANCE_FACTOR:
            removed += 1
        else:
            kept.append(t)

    # Pass 2 — fill gaps close to an integer multiple of the period with
    # evenly spaced beats. Gaps that are long but NOT near a multiple (e.g. a
    # beatless breakdown) are left alone: interpolating there would invent
    # beats where the tracker heard none for a reason.
    out = [kept[0]]
    inserted = 0
    for t in kept[1:]:
        gap = t - out[-1]
        mult = gap / ref
        k = round(mult)
        if k >= 2 and abs(mult - k) <= 0.25:
            lo = out[-1]
            for j in range(1, k):
                out.append(lo + gap * j / k)
            inserted += k - 1
        out.append(t)

    return RepairedGrid(times=tuple(out), n_removed=removed, n_inserted=inserted)


def snap_to_grid(
    times: tuple[float, ...] | list[float],
    grid: tuple[float, ...] | list[float],
    *,
    max_dist_s: float | None = None,
) -> tuple[float, ...]:
    """Snap each time to its nearest grid line, when one is close enough.

    Default tolerance is half the median grid interval — i.e. always snap
    inside the grid span (the nearest line is at most half an interval away),
    but never drag a marker that sits outside the grid (beatless intro/outro)
    onto its edge. Preserves order and deduplicates collisions.
    """
    g = sorted(float(x) for x in grid)
    if len(g) < 2:
        return tuple(float(t) for t in times)
    if max_dist_s is None:
        max_dist_s = statistics.median(b - a for a, b in zip(g, g[1:])) / 2
    out: list[float] = []
    for t in times:
        t = float(t)
        i = bisect.bisect_left(g, t)
        best = min(
            (
                c
                for c in (g[i - 1] if i > 0 else None, g[i] if i < len(g) else None)
                if c is not None
            ),
            key=lambda c: abs(c - t),
        )
        snapped = best if abs(best - t) <= max_dist_s else t
        if not out or snapped > out[-1]:
            out.append(snapped)
        elif t > out[-1]:
            out.append(t)  # collision after snap — keep the raw position
    return tuple(out)
