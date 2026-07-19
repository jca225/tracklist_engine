"""Pseudo-label -> TRM offset-target extraction (trm_flywheel_design.md §1-2).

The flywheel trains the TRM on REAL pseudo-labels the agentic loop accepts on
unlabeled sets, closing the measured sim2real gap (attic: "TRM decoder graft").
A high-confidence PredictedTimeline span is structurally a GT row for the target
machinery: its `ref_segments` are the same shape `_gt_pieces` reads, so a straight
predicted span rasterizes to a run of constant offset — exactly one TRM segment.

These tests pin the ONLY new data-path code (`pseudo_labels.pseudo_gt_row` +
`pseudo_span_to_offset_labels`) with zero audio/GPU: they exercise the timing
fields only, and prove the round-trip
    predicted span -> GT-row -> raster_targets -> encode_offset_labels
    -> offset_labels_to_ref_bins -> frames_to_segments
reproduces the input placement.
"""

from __future__ import annotations

import numpy as np

from workspaces.alignment_prototype.trajectory import offset_coords as oc
from workspaces.alignment_prototype.trajectory import pseudo_labels as pl

BIN_S = 0.5
VOCAB = oc.OffsetVocab(lo_bin=-8, hi_bin=1200, res_bin=1)


def _straight_span() -> dict:
    """A confident, straight (single-segment) predicted acappella span:
    mix [100, 130] plays ref [40, 70] at slope 1 -> constant 60 s offset."""
    return {
        "slot_label": "3w2",
        "recording_id": "abc123x",
        "track_id": "trk999",
        "claimed_stem": "acappella",
        "set_start_s": 100.0,
        "set_end_s": 130.0,
        "ref_start_s": 40.0,
        "ref_end_s": 70.0,
        "ref_stretch": 1.0,
        "ref_segments": [
            {"mix_start_s": 100.0, "ref_start_s": 40.0, "ref_end_s": 70.0},
        ],
        "driver_mode": "auto_commit",
        "agentic_quality": 0.82,
    }


# --- pseudo_gt_row: span dict -> GT-row-shaped dict (G4 sanity in here) -------


def test_pseudo_gt_row_preserves_timing_and_defaults_gt_only_fields():
    row = pl.pseudo_gt_row(_straight_span())
    assert row is not None
    # timing carried through unchanged
    assert row["set_start_s"] == 100.0 and row["set_end_s"] == 130.0
    assert row["ref_segments"][0]["ref_start_s"] == 40.0
    # ref_stretch -> tempo_ratio (the slope key _gt_pieces reads)
    assert float(row["tempo_ratio"]) == 1.0
    # GT-only fields default: no fader (unity), not an annotator abstain
    assert row.get("gain_curve") in (None, (), [])
    assert bool(row.get("unalignable", False)) is False
    # audio-resolution key preserved for TrajectorySpanDataset
    assert row["track_id"] == "trk999"
    assert row["claimed_stem"] == "acappella"


def test_pseudo_gt_row_rejects_non_auto_commit():
    span = _straight_span()
    span["driver_mode"] = "review"
    assert pl.pseudo_gt_row(span) is None


def test_pseudo_gt_row_rejects_degenerate_span():
    # G4: zero/negative mix window, no usable ref -> not a training target
    span = _straight_span()
    span["set_end_s"] = span["set_start_s"]
    assert pl.pseudo_gt_row(span) is None


def test_pseudo_gt_row_rejects_backwards_segment():
    # G4: ref_end < ref_start inside a forward segment is a decode pathology
    span = _straight_span()
    span["ref_segments"] = [
        {"mix_start_s": 100.0, "ref_start_s": 70.0, "ref_end_s": 40.0}
    ]
    assert pl.pseudo_gt_row(span) is None


# --- pseudo_span_to_offset_labels: the full path to TRM targets ---------------


def test_straight_span_encodes_to_constant_offset():
    labels = pl.pseudo_span_to_offset_labels(_straight_span(), BIN_S, VOCAB)
    live = labels[(labels != VOCAB.null_index) & (labels != oc.IGNORE_INDEX)]
    off_bins = live - VOCAB.zero_index
    # The offset is SPAN-RELATIVE (ref_pos - mix_relative_time), the coordinate
    # frames_to_segments / trm.trm_offset_targets decode in. At the span start
    # mix-relative time is ~0 and ref is 40 s, so a straight span sits at a
    # single constant offset class of ref_start / bin_s = 80 bins.
    assert live.size > 0
    assert np.all(off_bins == round(40.0 / BIN_S))


def test_roundtrip_reproduces_one_segment_at_the_input_placement():
    span = _straight_span()
    labels = pl.pseudo_span_to_offset_labels(span, BIN_S, VOCAB)
    ref_bin, null_mask = oc.offset_labels_to_ref_bins(labels, VOCAB)
    from workspaces.alignment_prototype.trajectory.targets import frames_to_segments

    segs = frames_to_segments(ref_bin, null_mask, BIN_S)
    assert len(segs) == 1
    mix0, ref0, ref1 = segs[0]
    assert mix0 == 0.0  # relative to span start
    assert abs(ref0 - 40.0) < BIN_S  # recovered the input ref_start


def test_rejected_span_returns_none_from_offset_path():
    span = _straight_span()
    span["driver_mode"] = "escalate"
    assert pl.pseudo_span_to_offset_labels(span, BIN_S, VOCAB) is None
