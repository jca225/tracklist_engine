"""Pseudo-label -> TRM offset-target extraction (trm_flywheel_design.md §1-2).

The real pseudo-label flywheel trains the TRM decoder on the REAL DJ-set
distribution instead of synthetic mashups, closing the measured sim2real gap
(attic/EXPERIMENTS.md "TRM decoder graft — sim2real gap MEASURED 2026-07-18":
synthetic-only train-fit 0.87 but real eval flat ~0.09 < 0.306 control). The
pseudo-labels are the agentic loop's AUTO_COMMIT spans on UNLABELED sets.

**The whole trick is that no new target machinery is needed.** A high-confidence
`PredictedTimeline` span carries exactly the fields `path_decode._gt_pieces` /
`targets.raster_targets` read from a hand-GT row (`set_start_s`, `set_end_s`,
`ref_segments`, `ref_start_s`, a slope). So a predicted span is a drop-in "row":
convert it with `pseudo_gt_row`, then the EXISTING `raster_targets` +
`offset_coords.encode_offset_labels` produce the TRM's offset-class targets.

This module is the ONLY new code on the data path, and it is pure (timing fields
only — no audio, no GPU). `pseudo_gt_row` also enforces the cheap structural
quality gates (G0 AUTO_COMMIT-only, G4 segment-shape sanity) from the design;
the belief-layer gates (G1 calibration, G2 agreement, G3 fiber-instance abstain)
live upstream in `agentic/` and decide which spans reach `driver_mode ==
"auto_commit"` in the first place.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .offset_coords import OffsetVocab, encode_offset_labels
from .targets import raster_targets

# G0: only the agentic loop's top autonomy rung mints a pseudo-label.
AUTO_COMMIT_MODE = "auto_commit"


def _segments_ok(segs: list[dict], s0: float, s1: float) -> bool:
    """G4 structural sanity on a predicted span's ref_segments. Reject decode
    pathologies: a segment must play the ref FORWARD (ref_end >= ref_start) and
    have a real mix extent. An empty list is allowed (linear-span fallback via
    ref_start_s, validated by the caller)."""
    for s in segs:
        try:
            rs, re = float(s["ref_start_s"]), float(s["ref_end_s"])
        except (KeyError, TypeError, ValueError):
            return False
        if re < rs:  # backwards segment — not a forward play
            return False
    return True


def pseudo_gt_row(span: dict) -> dict | None:
    """Convert one AUTO_COMMIT PredictedTimeline span into a GT-row-shaped dict
    the trajectory target machinery consumes unchanged, or ``None`` if the span
    fails a structural quality gate (G0/G4).

    GT-only fields default honestly: no `gain_curve` (a machine-placed clip has
    no hand fader, so every played frame is audible = unity gain) and
    `unalignable=False` (a span that reached AUTO_COMMIT is not an annotator
    abstain). `ref_stretch` (the driver's slope key) is mapped to `tempo_ratio`
    (the key `_gt_pieces` reads). `track_id` is preserved so
    `TrajectorySpanDataset` can resolve the span's audio.
    """
    # G0 — AUTO_COMMIT only.
    if str(span.get("driver_mode")) != AUTO_COMMIT_MODE:
        return None

    try:
        s0 = float(span["set_start_s"])
        s1 = float(span["set_end_s"])
    except (KeyError, TypeError, ValueError):
        return None
    # G4 — degenerate mix window.
    if s1 <= s0:
        return None

    segs = list(span.get("ref_segments") or ())
    # G4 — need a usable ref anchor: either segments or a linear ref_start_s.
    if not segs and span.get("ref_start_s") is None:
        return None
    if not _segments_ok(segs, s0, s1):
        return None

    slope = span.get("tempo_ratio")
    if slope is None:
        slope = span.get("ref_stretch")
    tempo_ratio = float(slope) if slope not in (None, "") else 1.0

    row: dict[str, Any] = {
        "slot_label": span.get("slot_label"),
        "recording_id": span.get("recording_id"),
        "track_id": span.get("track_id"),
        "claimed_stem": span.get("claimed_stem") or "regular",
        "set_start_s": s0,
        "set_end_s": s1,
        "ref_start_s": (
            float(span["ref_start_s"]) if span.get("ref_start_s") is not None else None
        ),
        "ref_end_s": (
            float(span["ref_end_s"]) if span.get("ref_end_s") is not None else None
        ),
        "tempo_ratio": tempo_ratio,
        "ref_segments": [
            {
                "mix_start_s": float(s["mix_start_s"]),
                "ref_start_s": float(s["ref_start_s"]),
                "ref_end_s": float(s["ref_end_s"]),
            }
            for s in segs
        ],
        # GT-only fields default honestly (see docstring).
        "gain_curve": None,
        "unalignable": False,
        # provenance so a materialized pseudo-GT set is auditable / never
        # mistaken for hand GT downstream.
        "pseudo_label": True,
        "agentic_quality": span.get("agentic_quality"),
    }
    return row


def pseudo_span_to_offset_labels(
    span: dict,
    bin_s: float,
    vocab: OffsetVocab,
) -> np.ndarray | None:
    """Full pure path: AUTO_COMMIT span -> per-frame TRM offset-class labels.

    Rasterizes the span onto its own bin-center grid with the EXISTING
    `raster_targets` (so the pseudo-label optimizes the same quantity the
    referee measures), then encodes offsets with the EXISTING
    `offset_coords.encode_offset_labels`. Returns ``None`` for spans rejected by
    `pseudo_gt_row`. Audio-free: it needs only the timing fields.
    """
    row = pseudo_gt_row(span)
    if row is None:
        return None

    s0, s1 = row["set_start_s"], row["set_end_s"]
    # Mirror TrajectorySpanDataset's bin raster: bins snapped to the global
    # bin_s grid, bin-center times (b0 + arange(tm) + 0.5) * bin_s.
    b0 = max(0, int(round(s0 / bin_s)))
    n_bins = max(1, int(round((s1 - s0) / bin_s)))
    times_s = (b0 + np.arange(n_bins) + 0.5) * bin_s

    # ref_dur_s bounds the in-ref window. Without the loaded ref audio we do not
    # know its true length; use the span's own decoded ref reach (max ref_end
    # across segments, or the linear ref_end_s) as a conservative bound so
    # frames past the decoded ref become IGNORE rather than fabricated positions.
    ref_reach = 0.0
    for s in row["ref_segments"]:
        ref_reach = max(ref_reach, float(s["ref_end_s"]))
    if row.get("ref_end_s") is not None:
        ref_reach = max(ref_reach, float(row["ref_end_s"]))
    ref_dur_s = ref_reach if ref_reach > 0 else None

    tgt = raster_targets(row, times_s, ref_dur_s=ref_dur_s)
    # `raster_targets` interpolates the trajectory at ABSOLUTE mix times, but the
    # TRM answer is encoded in the SPAN-RELATIVE (frame-index) offset coordinate
    # that `frames_to_segments` / `trm.trm_offset_targets` decode in
    # (offset_bin = ref_bin - frame_index). Compute the offset against
    # mix-relative frame-center times (0, bin_s, 2*bin_s, ...), NOT the absolute
    # `tgt.times_s`, so a straight span lands at a single constant offset class.
    rel_times_s = (np.arange(len(tgt.times_s)) + 0.5) * bin_s
    labels = encode_offset_labels(tgt.ref_pos_s, rel_times_s, tgt.kind, bin_s, vocab)
    return labels


__all__ = [
    "AUTO_COMMIT_MODE",
    "pseudo_gt_row",
    "pseudo_span_to_offset_labels",
]
