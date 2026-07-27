"""Fiber equivalence classes from GT — pure module for f0 scorer."""

from __future__ import annotations


def equivalence_classes(gt_rows: list[dict], fibers) -> dict[str, int]:
    """Map each GT (track_id, ref_start) to a fiber-class id.

    Rows sharing a *validated* fiber (n_instances >= 2) get the same class id;
    all other rows get unique ids (singletons). This preserves low-recall
    honesty: unflagged repeats are not collapsed.

    Args:
        gt_rows: List of GT dicts with keys: track_id, ref_segments[0][ref_start_s].
        fibers: Dict mapping track_id -> {ref_start_s -> Fiber(fiber_id, n_instances)}.
                Fiber objects must expose .fiber_id and .n_instances attributes.

    Returns:
        Dict mapping (track_id, ref_start_s) key to an integer class id.
        Rows sharing a validated fiber_id (n_instances >= 2) get the same id;
        all others are unique singletons.
    """
    result = {}
    fiber_to_class = {}  # fiber_id (for validated fibers) -> assigned class id
    next_singleton_id = 0

    for row in gt_rows:
        track_id = row.get("track_id")
        ref_segments = row.get("ref_segments", [])
        if not ref_segments:
            continue
        ref_start = ref_segments[0].get("ref_start_s")
        if ref_start is None:
            continue

        # Look up the fiber for this (track_id, ref_start) pair
        fiber_dict = fibers.get(track_id, {})
        fiber = fiber_dict.get(ref_start)

        # Row key for the result dict
        row_key = (track_id, ref_start)

        if fiber is None or fiber.fiber_id < 0 or fiber.n_instances < 2:
            # Unflagged or single-instance: gets its own singleton class
            result[row_key] = next_singleton_id
            next_singleton_id += 1
        else:
            # Validated fiber: all rows with this fiber_id get the same class
            fid = fiber.fiber_id
            if fid not in fiber_to_class:
                fiber_to_class[fid] = next_singleton_id
                next_singleton_id += 1
            result[row_key] = fiber_to_class[fid]

    return result
