"""GT stem overrides must bridge Ableton labels -> timeline via recording_id
AND retain every stem form a slot is played in (multi-form slots must not
collapse to a single form -- see slots 066/068/137, 003/007).
"""

from __future__ import annotations

from workspaces.alignment_prototype.fp_segments.stem_overrides import (
    lane_stem,
    timeline_stem_overrides,
)

INSTR = frozenset({"instrumental"})
ACAP = frozenset({"acappella"})
REG = frozenset({"regular"})


def test_multiform_slot_retains_both_forms_no_lane_dropped(tmp_path):
    """BB12-066 class: one recording played instrumental-then-regular across a
    transition must admit the instrumental lane (seg 1) AND the regular lane
    (seg 2). The old nearest-only collapse dropped whichever form was farther."""
    gt = tmp_path / "gt.yaml"
    gt.write_text(
        """
set_id: demo
tracks:
  - slot_label: "066"
    track_id: shared
    claimed_stem: instrumental
    set_start_s: 1552.47
    set_end_s:   1599.14
  - slot_label: "066"
    track_id: shared
    set_start_s: 1599.14
    set_end_s:   1638.7
"""
    )
    span = {
        "slot_label": "66",
        "recording_id": "shared",
        "claimed_stem": "regular",
        "set_start_s": 1552.47,
        "set_end_s": 1638.7,
    }
    ov = timeline_stem_overrides([span], gt)
    # Neither form is dropped: the span is admitted to BOTH lanes.
    assert lane_stem(ov, span, INSTR) == "instrumental"
    assert lane_stem(ov, span, REG) == "regular"


def test_recording_id_bridge_ignores_ableton_slot_collisions(tmp_path):
    """GT slot '007' must not remap an unrelated timeline slot '7'."""
    gt = tmp_path / "gt.yaml"
    gt.write_text(
        """
set_id: demo
tracks:
  - slot_label: "007"
    track_id: gt_instr_rid
    claimed_stem: instrumental
    set_start_s: 200.0
    set_end_s:   240.0
  - slot_label: "015"
    track_id: other_rid
    claimed_stem: acappella
    set_start_s: 300.0
    set_end_s:   340.0
"""
    )
    unrelated = {
        "slot_label": "7",
        "recording_id": "unrelated_timeline_rid",
        "claimed_stem": "regular",
        "set_start_s": 50.0,
        "set_end_s": 90.0,
    }
    matched = {
        "slot_label": "12w1",
        "recording_id": "gt_instr_rid",
        "claimed_stem": "regular",
        "set_start_s": 201.0,
        "set_end_s": 240.0,
    }
    ov = timeline_stem_overrides([unrelated, matched], gt)
    # slot '7' carries an unrelated recording -> no GT routing, no instrumental.
    assert lane_stem(ov, unrelated, INSTR) is None
    # slot '12w1' carries gt_instr_rid -> instrumental form applies.
    assert lane_stem(ov, matched, INSTR) == "instrumental"


def test_absent_slot_falls_back_to_span_claimed_stem(tmp_path):
    """A span whose recording has no GT row falls back to its own claimed_stem
    (preserves behavior when no override applies / no --stem-overrides file)."""
    gt = tmp_path / "gt.yaml"
    gt.write_text(
        """
set_id: demo
tracks:
  - slot_label: "015"
    track_id: other_rid
    claimed_stem: acappella
    set_start_s: 300.0
    set_end_s:   340.0
"""
    )
    span = {
        "slot_label": "99",
        "recording_id": "no_gt_rid",
        "claimed_stem": "instrumental",
        "set_start_s": 10.0,
        "set_end_s": 40.0,
    }
    ov = timeline_stem_overrides([span], gt)
    assert lane_stem(ov, span, INSTR) == "instrumental"
    assert lane_stem(ov, span, ACAP) is None


def test_form_not_overlapping_span_is_not_admitted(tmp_path):
    """Time-ranged routing: a form whose GT window does not overlap the span is
    not admitted, so a re-entry at a different time picks the right form."""
    gt = tmp_path / "gt.yaml"
    gt.write_text(
        """
set_id: demo
tracks:
  - slot_label: "050"
    track_id: shared
    claimed_stem: instrumental
    set_start_s: 100.0
    set_end_s:   150.0
  - slot_label: "050"
    track_id: shared
    set_start_s: 500.0
    set_end_s:   560.0
"""
    )
    late_span = {
        "slot_label": "50",
        "recording_id": "shared",
        "claimed_stem": "regular",
        "set_start_s": 500.0,
        "set_end_s": 560.0,
    }
    ov = timeline_stem_overrides([late_span], gt)
    # Only the regular form overlaps [500, 560]; the instrumental form is far away.
    assert lane_stem(ov, late_span, INSTR) is None
    assert lane_stem(ov, late_span, REG) == "regular"
