#!/usr/bin/env python3
"""Deterministic decision rules (no learned components).

CLONE tie-break (Phase 1 item 5): when a segment's song-region has
indistinguishable clone-mates, pick (a) the image nearest the previous
segment's song position, then (b) the earliest instance in the song.
"""

from __future__ import annotations


def clone_tiebreak(candidates: list[float], prev_song_pos: float | None) -> float:
    """Pick one song-time among clone-equivalent candidates.

    (a) nearest the previous segment's song position — a DJ jumping within a
    song tends to stay local; (b) tie / no previous segment: the FIRST
    instance in the song. Deterministic: ties on distance break earliest."""
    if not candidates:
        raise ValueError("no candidates")
    if prev_song_pos is not None:
        return min(candidates, key=lambda c: (abs(c - prev_song_pos), c))
    return min(candidates)
