"""Tests for fiber equivalence classes — TDD gate for f0.1."""

from __future__ import annotations

from workspaces.alignment_prototype.fiber_assignment import equivalence_classes


def test_equivalence_classes():
    """Two GT rows of the SAME recording whose ref-starts fall in ONE
    validated fiber (n_instances>=2) share a class; a third, unflagged
    single-instance row is its own class.
    """
    gt_rows = [
        {"track_id": "rec1", "ref_segments": [{"ref_start_s": 10.0}]},
        {
            "track_id": "rec1",
            "ref_segments": [{"ref_start_s": 15.0}],
        },  # same recording, same fiber
        {
            "track_id": "rec2",
            "ref_segments": [{"ref_start_s": 5.0}],
        },  # different recording
    ]

    # Fibers per recording: rec1 has a validated fiber (fiber_id=1, n_instances=2);
    # rec2 has a single-instance fiber (fiber_id=2, n_instances=1)
    class Fiber:
        def __init__(self, fiber_id: int, n_instances: int):
            self.fiber_id = fiber_id
            self.n_instances = n_instances

    fibers = {
        "rec1": {
            10.0: Fiber(fiber_id=1, n_instances=2),
            15.0: Fiber(fiber_id=1, n_instances=2),
        },
        "rec2": {
            5.0: Fiber(fiber_id=2, n_instances=1),
        },
    }

    result = equivalence_classes(gt_rows, fibers)
    assert isinstance(result, dict)
    # Rows 0 and 1 (both rec1, both fiber_id=1 with n_instances=2) share a class
    assert result[("rec1", 10.0)] == result[("rec1", 15.0)]
    # Row 2 (rec2, single-instance fiber) is its own class
    assert result[("rec2", 5.0)] != result[("rec1", 10.0)]
