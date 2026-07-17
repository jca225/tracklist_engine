"""Fiber equivalence classes from GT rows (pure, no audio, no I/O).

Maps each GT row to an integer equivalence-class id based on which fiber
(self-repeat class) its reference position falls in. The function is
fiber-consistent: rows that share a *validated* fiber collapse to the same
class id; everything else gets its own unique singleton class.

**Low-recall honesty (SALAMI P .88 / R .06):** the fiber detector is precise
but misses most real repeats. A row flagged n_instances=1 (or fiber_id<0) may
still be a genuine repeat — but we *never* collapse it on that guess. Only
validated fibers (n_instances >= 2, fiber_id >= 0) collapse. This is an
intentional design constraint: do not paper over it.

``fibers`` structure
--------------------
A plain dict mapping ``track_id -> list[tuple[float, float, int, int]]``:

    {
        track_id: [
            (start_s, end_s, fiber_id, n_instances),
            ...
        ]
    }

Each tuple is one contiguous interval of the reference audio.
- ``start_s``, ``end_s``: the interval in reference-track seconds (half-open:
  ``start_s <= ref_start_s < end_s``).
- ``fiber_id``: the repeat-class label from the detector. ``fiber_id < 0``
  means ungrouped / silence — treated as a singleton regardless of
  ``n_instances``.
- ``n_instances``: how many distinct intervals share this ``fiber_id`` for this
  track. ``n_instances < 2`` means the fiber has not been validated as a true
  repeat; treat as singleton.

This shape is the simplest structure the test can build by hand. Downstream
tasks (F0.2 / F0.3) adapt real ``fibers.detect`` output — which produces
``fiber_intervals(labels, hz, min_len_s)`` as ``[(start_s, end_s, label)]``
and ``fiber_ambiguity(…)`` with ``{fiber_id, n_instances, …}`` — into this
dict. The mapping is a thin pre-computation:
  1. Call ``fiber_intervals`` to get the interval list.
  2. Count how many intervals share each label → ``n_instances``.
  3. Build the tuple list.

``gt_rows``
-----------
Each row is a plain dict with at minimum ``track_id``.  The ref position is
taken from ``row["ref_segments"][0]["ref_start_s"]`` when ``ref_segments`` is
present (multiseg rows), otherwise from ``row["ref_start_s"]``.

Return value
------------
A ``dict[int, int]`` mapping row index (0-based position in ``gt_rows``) to
an integer class id.  Rows sharing a validated fiber get the SAME class id;
all others get a distinct id that no other row shares.
"""

from __future__ import annotations


def _ref_start(row: dict) -> float | None:
    """Extract the canonical ref position from a GT row.

    Multiseg rows carry ``ref_segments``; use the first segment's start.
    Falls back to the top-level ``ref_start_s``. Returns None if neither is
    present (callers treat as singleton).
    """
    segs = row.get("ref_segments")
    if segs:
        first = segs[0]
        if "ref_start_s" in first:
            return float(first["ref_start_s"])
    val = row.get("ref_start_s")
    return float(val) if val is not None else None


def _fiber_key(
    row: dict,
    fibers: dict[str, list[tuple[float, float, int, int]]],
) -> tuple[str, int] | None:
    """Return ``(track_id, fiber_id)`` if the row's ref position falls in a
    validated fiber interval; otherwise return None (singleton).

    A fiber is *validated* when both:
    - ``fiber_id >= 0`` (not ungrouped/silence)
    - ``n_instances >= 2`` (more than one instance detected)
    """
    track_id: str | None = row.get("track_id")
    if track_id is None:
        return None

    intervals = fibers.get(track_id)
    if not intervals:
        return None

    ref_s = _ref_start(row)
    if ref_s is None:
        return None

    for start_s, end_s, fiber_id, n_instances in intervals:
        if start_s <= ref_s < end_s:
            # Found the enclosing interval — check validation criteria.
            if fiber_id >= 0 and n_instances >= 2:
                return (track_id, fiber_id)
            # Interval found but not validated → singleton (stop scanning).
            return None

    # No interval contained the ref position → singleton.
    return None


def equivalence_classes(
    gt_rows: list[dict],
    fibers: dict[str, list[tuple[float, float, int, int]]],
) -> dict[int, int]:
    """Map each GT row index to an equivalence-class id.

    Args:
        gt_rows: List of GT row dicts (each must have ``track_id`` and a ref
            position source — see module docstring).
        fibers: Fiber interval map — see module docstring for exact shape.

    Returns:
        ``dict[int, int]`` where keys are 0-based row indices and values are
        integer class ids.  Rows sharing a validated fiber get the SAME class
        id; all other rows each get a unique singleton id.
    """
    class_map: dict[tuple[str, int], int] = {}
    result: dict[int, int] = {}
    next_id = 0

    for idx, row in enumerate(gt_rows):
        key = _fiber_key(row, fibers)
        if key is not None:
            # Validated fiber — reuse or allocate a shared class id.
            if key not in class_map:
                class_map[key] = next_id
                next_id += 1
            result[idx] = class_map[key]
        else:
            # Singleton — allocate a unique id not shared with anyone.
            result[idx] = next_id
            next_id += 1

    return result
