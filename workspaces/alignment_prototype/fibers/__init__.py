"""fibers — self-repeat equivalence classes over a reference track.

Extracted 2026-07-09 from the flat ``ref_fibers.py`` / ``harmony_fibers.py``
modules (the ``labeling/als`` precedent: package + import shims at the old
paths). One concept, one home:

  detect.py   — the production detector (HuBERT + silence gate + diagonal
                verification + average linkage) and its soft/fp/multi variants
  harmony.py  — the CENS time-lag harmony detector (opt-in union arm; see
                external/fiber_validation_findings.md phase 3)
  gt_als.py   — fiber ground-truth determination via Ableton: render a
                fiber-overlay .als for hand correction, parse the corrected
                session back into pairwise fiber GT

Design rules (do not relearn): fibers MUST be HuBERT + silence-gated for
vocals, never chroma (blobs to one fiber); average linkage, never transitive
merges; precision is the contract — a same-fiber verdict must almost never
be a lie (SALAMI external check: P 0.88 / R 0.065).
"""

from __future__ import annotations

from workspaces.alignment_prototype.fibers.detect import (
    FIBER_VERSION,
    FiberParams,
    _avg_linkage,
    _diag_sim,
    _long_repeats,
    compute_fibers,
    compute_fibers_fp,
    compute_fibers_multi,
    compute_fibers_soft,
    fiber_ambiguity,
    fiber_at,
    fiber_intervals,
    same_fiber,
)
from workspaces.alignment_prototype.fibers.harmony import harmony_fibers

__all__ = [
    "FIBER_VERSION",
    "FiberParams",
    "_avg_linkage",
    "_diag_sim",
    "_long_repeats",
    "compute_fibers",
    "compute_fibers_fp",
    "compute_fibers_multi",
    "compute_fibers_soft",
    "fiber_ambiguity",
    "fiber_at",
    "fiber_intervals",
    "harmony_fibers",
    "same_fiber",
]
