"""Typed time domains + TimeMap — the coordinate algebra (plan workstream A2).

Five time coordinates coexist in this repo (mix seconds, ref seconds,
arrangement beats, content beats, warped/unwarped variants) and their
conversions produced the single largest fix-commit class (22 fixes: the
-430 s warp-anchor shift, loop-SECONDS vs warp-mapped beats, master-tempo
confusion, clamp-vs-extrapolate drift — docs/entropy_reduction_plan.md).

Two tools, used together:

* **Branded floats** (`MixSec`, `RefSec`, `ArrBeat`, `ContentBeat`) make the
  domain part of a signature under mypy. Known limit: arithmetic degrades to
  ``float`` — so cross-domain conversion NEVER happens by arithmetic, only
  through a TimeMap.
* **`TimeMap`** — a strictly-monotonic piecewise-linear mapping between two
  named domains, with explicit out-of-domain policy, a true ``inverse()``,
  and ``compose()``. Warp markers, tempo maps, span stretches, and DTW paths
  all become TimeMaps (API shape follows pretty_midi's paired
  ``tick_to_time``/``time_to_tick``; strict-monotonic filtering follows
  synctoolbox's ``make_path_strictly_monotonic``).

Stdlib-only (bisect), scalar API; vectorize at the call site if profiling
demands it.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Iterable, Literal, NewType

MixSec = NewType("MixSec", float)
"""Seconds on the DJ-set (mix) clock."""

RefSec = NewType("RefSec", float)
"""Seconds on a reference track's own clock."""

ArrBeat = NewType("ArrBeat", float)
"""Beats on a Live arrangement timeline."""

ContentBeat = NewType("ContentBeat", float)
"""Beats inside one clip's content (loop-local)."""

Policy = Literal["extrapolate", "clamp", "raise"]


class TimeMapError(ValueError):
    """Invalid construction (non-monotonic anchors) or out-of-domain apply."""


@dataclass(frozen=True)
class TimeMap:
    """Strictly-increasing piecewise-linear bijection ``src -> dst``.

    ``anchors`` are (src, dst) pairs, strictly increasing in BOTH coordinates
    (validated — a backward jump like the `6c084eb` loop bug fails here, at
    construction, not three consumers later). ``policy`` states what happens
    outside the anchor range: linear ``extrapolate`` (default; matches
    labeling/als WarpMarkers), ``clamp`` to the edge value, or ``raise``.
    Domains are carried as strings and checked on ``compose`` so a
    mix->ref map cannot be chained onto another mix->ref map.
    """

    src_domain: str
    dst_domain: str
    anchors: tuple[tuple[float, float], ...]
    policy: Policy = "extrapolate"

    def __post_init__(self) -> None:
        if len(self.anchors) < 2:
            raise TimeMapError(
                f"TimeMap {self.src_domain}->{self.dst_domain}: need >=2 anchors, "
                f"got {len(self.anchors)}"
            )
        for (a0, b0), (a1, b1) in zip(self.anchors, self.anchors[1:]):
            if not (a1 > a0 and b1 > b0):
                raise TimeMapError(
                    f"TimeMap {self.src_domain}->{self.dst_domain}: anchors must "
                    f"be strictly increasing in both coordinates; "
                    f"({a0}, {b0}) -> ({a1}, {b1}) is not"
                )

    # -- application ---------------------------------------------------------

    def apply(self, x: float) -> float:
        """Map one src value to dst under the out-of-domain policy."""
        xs = [a for a, _ in self.anchors]
        lo_x, lo_y = self.anchors[0]
        hi_x, hi_y = self.anchors[-1]
        if x < lo_x or x > hi_x:
            if self.policy == "raise":
                raise TimeMapError(
                    f"{x} outside {self.src_domain} domain [{lo_x}, {hi_x}]"
                )
            if self.policy == "clamp":
                return lo_y if x < lo_x else hi_y
        # segment index: rightmost anchor <= x (extrapolate uses edge segments)
        i = min(max(bisect_right(xs, x) - 1, 0), len(self.anchors) - 2)
        (x0, y0), (x1, y1) = self.anchors[i], self.anchors[i + 1]
        return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

    __call__ = apply

    # -- algebra -------------------------------------------------------------

    def inverse(self) -> TimeMap:
        """dst -> src. Total because anchors are strictly monotonic."""
        return TimeMap(
            src_domain=self.dst_domain,
            dst_domain=self.src_domain,
            anchors=tuple((b, a) for a, b in self.anchors),
            policy=self.policy,
        )

    def compose(self, other: TimeMap) -> TimeMap:
        """``self.compose(other)``: src(self) -> dst(other), i.e. other∘self.

        Domain-checked: ``self.dst_domain`` must equal ``other.src_domain``
        (this is where a mix->ref map refuses to chain onto mix->ref).
        Anchored at the union of self's anchors and other's knots pulled back
        through self, so the composition is exact, not resampled.
        """
        if self.dst_domain != other.src_domain:
            raise TimeMapError(
                f"cannot compose {self.src_domain}->{self.dst_domain} with "
                f"{other.src_domain}->{other.dst_domain}: domain mismatch"
            )
        inv = self.inverse()
        knots = {a for a, _ in self.anchors}
        lo, hi = self.anchors[0][1], self.anchors[-1][1]
        for k, _ in other.anchors:
            if lo <= k <= hi:
                knots.add(inv.apply(k))
        xs = sorted(knots)
        anchors = tuple((x, other.apply(self.apply(x))) for x in xs)
        return TimeMap(
            src_domain=self.src_domain,
            dst_domain=other.dst_domain,
            anchors=anchors,
            policy=self.policy,
        )

    # -- constructors --------------------------------------------------------

    @staticmethod
    def linear(
        src_domain: str,
        dst_domain: str,
        *,
        src_start: float,
        dst_start: float,
        rate: float,
        length: float = 1.0,
    ) -> TimeMap:
        """One straight segment: dst = dst_start + (src - src_start) * rate."""
        if rate <= 0:
            raise TimeMapError(f"rate must be positive, got {rate}")
        return TimeMap(
            src_domain,
            dst_domain,
            (
                (src_start, dst_start),
                (src_start + length, dst_start + length * rate),
            ),
        )

    @staticmethod
    def from_pairs(
        src_domain: str,
        dst_domain: str,
        pairs: Iterable[tuple[float, float]],
        *,
        policy: Policy = "extrapolate",
    ) -> TimeMap:
        """Build from possibly-redundant pairs; drops non-advancing ones.

        The synctoolbox ``make_path_strictly_monotonic`` move: DTW paths and
        duplicated warp markers (the Aftershock bug) contain stalls — keep
        the first pair, then only pairs strictly advancing in BOTH coords.
        """
        kept: list[tuple[float, float]] = []
        for a, b in sorted(pairs):
            if not kept or (a > kept[-1][0] and b > kept[-1][1]):
                kept.append((float(a), float(b)))
        return TimeMap(src_domain, dst_domain, tuple(kept), policy=policy)
