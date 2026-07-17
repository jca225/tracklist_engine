"""Tests for fiber_assignment.equivalence_classes.

Table-driven, no audio, no I/O — plain hand-constructable inputs only.
"""

from __future__ import annotations

from workspaces.alignment_prototype.fiber_assignment import equivalence_classes


def _row(track_id: str, ref_start_s: float) -> dict:
    """Minimal GT row for testing (straight-clip, no ref_segments)."""
    return {"track_id": track_id, "ref_start_s": ref_start_s}


def test_equivalence_classes():
    """Two GT rows of the SAME recording whose ref-starts fall in ONE validated
    fiber (n_instances>=2) share a class id; a third single-instance row is its
    own distinct class.

    Fiber structure: dict mapping track_id ->
        list of (start_s, end_s, fiber_id, n_instances).
    A ref_start_s falls in the interval [start_s, end_s).
    fiber_id < 0 means ungrouped/silence; n_instances < 2 means unvalidated.
    """
    # Two acappella plays of the same recording land in the SAME repeated chorus
    # fiber — validated (n_instances=2).
    row_a = _row("tid_chorus", 30.0)  # ref_start at 30s — in fiber 0, chorus repeat
    row_b = _row("tid_chorus", 90.0)  # ref_start at 90s — in fiber 0, chorus repeat

    # A third row for a DIFFERENT recording with no validated fiber match.
    row_c = _row("tid_verse", 5.0)  # ref_start at 5s — unvalidated (n_instances=1)

    fibers: dict[str, list[tuple[float, float, int, int]]] = {
        "tid_chorus": [
            # (start_s, end_s, fiber_id, n_instances)
            (25.0, 55.0, 0, 2),  # covers row_a at 30s — validated repeat
            (85.0, 115.0, 0, 2),  # covers row_b at 90s — same fiber, validated
        ],
        "tid_verse": [
            (0.0, 20.0, 1, 1),  # covers row_c at 5s — n_instances=1, NOT validated
        ],
    }

    result = equivalence_classes([row_a, row_b, row_c], fibers)

    # row_a and row_b share a validated fiber — same class id
    assert result[0] == result[1], (
        f"Rows in the same validated fiber must share a class id; "
        f"got {result[0]} and {result[1]}"
    )

    # row_c is unvalidated (n_instances=1) — must get its own singleton class
    assert result[2] != result[0], (
        f"Unvalidated row must get its own singleton class; "
        f"got {result[2]} (same as validated class {result[0]})"
    )

    # All three rows must have an entry
    assert len(result) == 3, f"Expected 3 entries, got {len(result)}"


def test_singleton_for_negative_fiber_id():
    """A row whose ref_start lands in a fiber_id < 0 interval is a singleton
    even when n_instances >= 2 (ungrouped/silence — the low-recall-honest rule)."""
    row = _row("tid_silence", 10.0)
    fibers: dict[str, list[tuple[float, float, int, int]]] = {
        "tid_silence": [
            (5.0, 20.0, -1, 3),  # fiber_id < 0 — ungrouped; must NOT collapse
        ],
    }
    result = equivalence_classes([row], fibers)
    assert len(result) == 1

    # A second row for the same silence interval must also be a singleton and
    # must NOT share a class with the first (two separate singletons).
    row2 = _row("tid_silence", 12.0)
    result2 = equivalence_classes([row, row2], fibers)
    assert len(result2) == 2
    assert result2[0] != result2[1], (
        "Two rows in a silence fiber (fiber_id<0) must each be their own singleton"
    )


def test_no_fiber_entry_is_singleton():
    """A row for a track_id with no fiber data at all is a singleton."""
    row_a = _row("tid_unknown", 0.0)
    row_b = _row("tid_unknown", 50.0)
    result = equivalence_classes([row_a, row_b], fibers={})
    assert len(result) == 2
    assert result[0] != result[1], "No-fiber rows must each be singletons"


def test_ref_start_outside_all_intervals_is_singleton():
    """A row whose ref_start_s falls outside all fiber intervals is a singleton."""
    row = _row("tid_chorus", 200.0)  # well past all defined intervals
    fibers: dict[str, list[tuple[float, float, int, int]]] = {
        "tid_chorus": [
            (25.0, 55.0, 0, 2),
            (85.0, 115.0, 0, 2),
        ],
    }
    result = equivalence_classes([row], fibers)
    assert len(result) == 1


def test_multiseg_row_uses_first_ref_start():
    """For a row with ref_segments, equivalence_classes uses
    ref_segments[0]['ref_start_s'] as the ref position (matching the brief's
    spec), not the top-level ref_start_s.

    The test is structured so it FAILS if _ref_start were changed to return the
    top-level 999.0:  999.0 falls outside all defined fiber intervals, so the
    multiseg row would become a singleton and would NOT share a class id with the
    plain 30.0 row — the assertion ``result[0] == result[1]`` would fail.
    """
    # Multiseg row: top-level ref_start_s=999.0 must be IGNORED; the position
    # used is ref_segments[0]["ref_start_s"]=30.0, which falls in fiber 0.
    row_ms = {
        "track_id": "tid_ms",
        "ref_start_s": 999.0,  # top-level — must be IGNORED
        "ref_segments": [
            {"ref_start_s": 30.0},
            {"ref_start_s": 60.0},
        ],
    }
    # Plain row at 30.0 (no ref_segments) — falls in the same validated fiber 0.
    # If _ref_start correctly reads ref_segments[0] for the multiseg row (→30.0),
    # both rows land in fiber 0 and share a class id.
    # If _ref_start incorrectly returns 999.0, row_ms is a singleton (999.0 is
    # outside all intervals) and the assertion below fails.
    row_plain = {
        "track_id": "tid_ms",
        "ref_start_s": 30.0,
    }
    fibers: dict[str, list[tuple[float, float, int, int]]] = {
        "tid_ms": [
            (25.0, 55.0, 0, 2),  # covers 30.0 — validated repeat (fiber 0)
            (85.0, 115.0, 0, 2),  # second interval of the same fiber
        ],
    }
    result = equivalence_classes([row_ms, row_plain], fibers)
    assert len(result) == 2
    assert result[0] == result[1], (
        f"Multiseg row (ref_segments[0].ref_start_s=30.0) and plain row "
        f"(ref_start_s=30.0) must share the same fiber class id; "
        f"got {result[0]} and {result[1]} — likely _ref_start returned the "
        f"top-level 999.0 instead of the first segment's 30.0"
    )
