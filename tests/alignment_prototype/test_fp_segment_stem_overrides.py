"""GT stem overrides must bridge Ableton labels → timeline via recording_id."""

from __future__ import annotations

from workspaces.alignment_prototype.fp_segments.stem_overrides import (
    timeline_stem_overrides,
)


def test_recording_id_bridge_ignores_ableton_slot_collisions(tmp_path):
    """BB12-style: GT slot '007' must not remap timeline slot '7' for another rid."""
    gt = tmp_path / "gt.yaml"
    gt.write_text(
        """
set_id: demo
tracks:
  - slot_label: "007"
    track_id: gt_instr_rid
    claimed_stem: instrumental
    set_start_s: 200.0
  - slot_label: "015"
    track_id: other_rid
    claimed_stem: acappella
    set_start_s: 300.0
"""
    )
    spans = [
        {
            "slot_label": "7",
            "recording_id": "unrelated_timeline_rid",
            "claimed_stem": "regular",
            "set_start_s": 50.0,
        },
        {
            "slot_label": "12w1",
            "recording_id": "gt_instr_rid",
            "claimed_stem": "regular",
            "set_start_s": 201.0,
        },
    ]
    overrides = timeline_stem_overrides(spans, gt)
    assert overrides.get("7") is None
    assert overrides["12w1"] == "instrumental"


def test_multi_span_recording_picks_nearest_set_start(tmp_path):
    gt = tmp_path / "gt.yaml"
    gt.write_text(
        """
set_id: demo
tracks:
  - slot_label: "100"
    track_id: shared
    claimed_stem: instrumental
    set_start_s: 1000.0
"""
    )
    spans = [
        {
            "slot_label": "20",
            "recording_id": "shared",
            "claimed_stem": "regular",
            "set_start_s": 100.0,
        },
        {
            "slot_label": "40w2",
            "recording_id": "shared",
            "claimed_stem": "acappella",
            "set_start_s": 1005.0,
        },
    ]
    overrides = timeline_stem_overrides(spans, gt)
    assert overrides == {"40w2": "instrumental"}
