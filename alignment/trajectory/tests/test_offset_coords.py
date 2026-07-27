"""Offset-coordinate answer encoding (TRM bake-off §2.3).

TRM needs a fixed-size answer grid, but Tr varies per span. Encode the answer as
per-frame *clip-start offset* bins over a FIXED vocabulary: a straight clip is a
run of constant offset; discontinuities are the DJ's jumps; NULL stays NULL. The
round-trip (ref-position raster -> offset labels -> ref bins -> segments) must
reproduce the same segments the conv path decodes."""

from __future__ import annotations

import numpy as np

from alignment.trajectory import offset_coords as oc
from alignment.trajectory.targets import (
    KIND_IGNORE,
    KIND_NULL,
    KIND_POSITION,
)

BIN_S = 0.5


def test_vocab_size_is_fixed_and_independent_of_span():
    v = oc.OffsetVocab(lo_bin=-8, hi_bin=1200, res_bin=1)
    # NULL + IGNORE are not offset classes; vocab is (hi-lo+1) offset bins + NULL
    assert v.size == (1200 - (-8) + 1) + 1
    assert v.null_index == v.size - 1


def test_straight_clip_encodes_to_constant_offset():
    # ref plays forward at slope 1 starting at 40 s -> offset == 40 s / BIN_S bins
    tm = 20
    times_s = np.arange(tm) * BIN_S  # mix-relative
    ref_pos_s = 40.0 + times_s
    kind = np.full(tm, KIND_POSITION, dtype=np.int64)
    v = oc.OffsetVocab(lo_bin=-8, hi_bin=1200, res_bin=1)
    labels = oc.encode_offset_labels(ref_pos_s, times_s, kind, BIN_S, v)
    off_bins = labels[labels != v.null_index] - v.zero_index
    assert np.all(off_bins == round(40.0 / BIN_S))  # constant 80-bin offset


def test_null_and_ignore_map_to_their_indices():
    tm = 6
    times_s = np.arange(tm) * BIN_S
    ref_pos_s = 10.0 + times_s
    kind = np.array(
        [KIND_POSITION, KIND_NULL, KIND_POSITION, KIND_IGNORE, KIND_NULL, KIND_POSITION]
    )
    v = oc.OffsetVocab(lo_bin=-8, hi_bin=1200, res_bin=1)
    labels = oc.encode_offset_labels(ref_pos_s, times_s, kind, BIN_S, v)
    assert labels[1] == v.null_index
    assert labels[4] == v.null_index
    assert labels[3] == oc.IGNORE_INDEX  # excluded from the loss


def test_roundtrip_straight_span_reproduces_one_segment():
    tm = 30
    times_s = np.arange(tm) * BIN_S
    ref_pos_s = 12.0 + times_s  # single straight segment, ref 12 s -> 12+15 s
    kind = np.full(tm, KIND_POSITION, dtype=np.int64)
    v = oc.OffsetVocab(lo_bin=-8, hi_bin=1200, res_bin=1)
    labels = oc.encode_offset_labels(ref_pos_s, times_s, kind, BIN_S, v)
    ref_bin, null_mask = oc.offset_labels_to_ref_bins(labels, v)
    from alignment.trajectory.targets import frames_to_segments

    segs = frames_to_segments(ref_bin, null_mask, BIN_S)
    assert len(segs) == 1
    mix0, ref0, ref1 = segs[0]
    assert mix0 == 0.0
    assert abs(ref0 - 12.0) < BIN_S
    assert abs(ref1 - (12.0 + tm * BIN_S)) < 2 * BIN_S


def test_roundtrip_two_segments_with_a_jump():
    # segment A: ref 20s slope1 for 10 frames; jump; segment B: ref 5s slope1
    tm = 20
    times_s = np.arange(tm) * BIN_S
    ref_pos_s = np.empty(tm)
    ref_pos_s[:10] = 20.0 + times_s[:10]
    ref_pos_s[10:] = 5.0 + (times_s[10:] - times_s[10])
    kind = np.full(tm, KIND_POSITION, dtype=np.int64)
    v = oc.OffsetVocab(lo_bin=-8, hi_bin=1200, res_bin=1)
    labels = oc.encode_offset_labels(ref_pos_s, times_s, kind, BIN_S, v)
    ref_bin, null_mask = oc.offset_labels_to_ref_bins(labels, v)
    from alignment.trajectory.targets import frames_to_segments

    segs = frames_to_segments(ref_bin, null_mask, BIN_S)
    assert len(segs) == 2
