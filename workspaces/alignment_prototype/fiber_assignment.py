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

import numpy as np
from scipy.optimize import linear_sum_assignment


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

    # Intervals are assumed non-overlapping: the first enclosing interval is the
    # unique owner of ref_s.  If it is not validated we stop immediately (a later
    # overlapping valid interval would be a data-integrity error, not a fallback).
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


def assign(
    pred_segments: list[float],
    gt_rows: list[dict],
    classes: dict[int, int],
    *,
    tol: float = 2.0,
) -> list[tuple[int, int, float]]:
    """Optimally match predictions to GT rows using fiber-consistent costs.

    Uses ``scipy.optimize.linear_sum_assignment`` (Hungarian algorithm) to find
    the minimum-cost one-to-one matching.

    **Cost rule:**

    - **0.0** — the prediction's ref time is within ``tol`` seconds of ANY GT
      row that shares the same fiber equivalence class (``classes[gt_idx]``).
      Within-class occurrence swaps are FREE: the aligner cannot distinguish
      which chorus repeat the DJ played, so penalising the wrong-but-equivalent
      assignment would be unfair.
    - **true gap in seconds** — ``|pred_ref_s - gt_ref_s|`` — when the
      prediction does not land on any member of the correct class.

    **Singleton guarantee:** a single-instance span's class contains only
    itself, so "any class member" reduces to exactly that one GT row's ref
    position.  A miss is never forgiven — the cost is the true seconds gap.
    This guards the 746 s single-instance miss regression seen on 2026-07-17.

    Args:
        pred_segments: List of predicted ref-start times (floats, in seconds).
            Shape: ``[pred_ref_s, ...]`` — one float per predicted span.
            F0.3 (``trajectory_acc``) will feed this from ``(mix_off,
            ref_start, ref_end)`` tuples by passing ``[t[1] for t in pred]``.
        gt_rows: List of GT row dicts (same order and schema as passed to
            ``equivalence_classes``).  Each row must have a ref position
            readable by ``_ref_start``.
        classes: Output of ``equivalence_classes(gt_rows, fibers)`` — maps
            0-based GT row index to an integer class id.
        tol: Tolerance in seconds for the cost-0 decision (default 2.0 s,
            matching the scorer's canonical tolerance).  A prediction is
            considered to "land on" a class member when
            ``|pred_ref_s - member_ref_s| < tol``.

    Returns:
        List of ``(pred_idx, gt_idx, cost)`` triples — one per matched pair.
        The matching is one-to-one (Hungarian).  ``linear_sum_assignment``
        operates on the cost matrix as-is and returns a partial matching of
        size ``min(n_pred, n_gt)`` — it does NOT pad or trim the matrix:

        - When ``n_pred > n_gt``: some predictions are left unmatched and are
          not present in the returned triples.
        - When ``n_pred < n_gt``: some GT rows are left unmatched and are not
          present in the returned triples.

        The returned list has exactly ``min(len(pred_segments), len(gt_rows))``
        triples.
    """
    n_pred = len(pred_segments)
    n_gt = len(gt_rows)

    if n_pred == 0 or n_gt == 0:
        return []

    # Pre-build: for each GT class id, the set of ref positions of all members
    # (used to evaluate whether a pred "lands on any member of the class").
    class_to_ref_positions: dict[int, list[float]] = {}
    for gt_idx, row in enumerate(gt_rows):
        class_id = classes[gt_idx]
        ref_s = _ref_start(row)
        if ref_s is not None:
            class_to_ref_positions.setdefault(class_id, []).append(ref_s)

    # Build cost matrix: shape (n_pred, n_gt).
    cost_matrix = np.empty((n_pred, n_gt), dtype=np.float64)

    for p_idx, pred_ref_s in enumerate(pred_segments):
        for g_idx in range(n_gt):
            gt_ref_s = _ref_start(gt_rows[g_idx])
            class_id = classes[g_idx]
            member_positions = class_to_ref_positions.get(class_id, [])

            # Cost is 0 if the pred lands within tol of ANY member of the
            # correct class (validated fiber) OR of the GT row itself.
            # For singletons the only member IS the GT row itself, so this
            # reduces to checking |pred - gt| < tol — which evaluates to a
            # non-zero cost when the pred is 746 s away.
            if any(abs(pred_ref_s - m) < tol for m in member_positions):
                cost_matrix[p_idx, g_idx] = 0.0
            else:
                # True ref-time gap to this specific GT row.
                cost_matrix[p_idx, g_idx] = (
                    abs(pred_ref_s - gt_ref_s) if gt_ref_s is not None else 1e9
                )

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    return [
        (int(p), int(g), float(cost_matrix[p, g])) for p, g in zip(row_ind, col_ind)
    ]
