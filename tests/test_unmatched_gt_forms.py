"""Form-level GT coverage: when a recording has multiple GT rows (a slot played
across a stem transition, e.g. BB12 066 instrumental->regular), the scorer maps
each predicted span to its NEAREST GT row by set_start. Rows that no span selects
are silently unscored -- ``unmatched_gt_forms`` surfaces exactly those, which
``never_matched_recordings`` (recording-level) cannot see.
"""

from __future__ import annotations

from alignment.never_matched import unmatched_gt_forms


def _row(tid, slot, stem, ss, se):
    return {
        "track_id": tid,
        "slot_label": slot,
        "claimed_stem": stem,
        "set_start_s": ss,
        "set_end_s": se,
    }


def test_multiform_slot_second_form_is_unmatched_when_one_span():
    """066 class: one recording, two forms, ONE predicted span -> the span picks
    the nearest form (instrumental@1552); the regular form@1599 is unscored."""
    gt = [
        _row("shared", "066", "instrumental", 1552.0, 1599.0),
        _row("shared", "066", "regular", 1599.0, 1638.0),
    ]
    spans = [{"recording_id": "shared", "slot_label": "66", "set_start_s": 1560.0}]
    out = unmatched_gt_forms(gt, spans)
    assert len(out) == 1
    assert out[0]["claimed_stem"] == "regular"
    assert out[0]["set_start_s"] == 1599.0
    # recording-level accounting is blind to this (the recording IS matched):
    from alignment.never_matched import never_matched_recordings

    assert never_matched_recordings(gt, spans) == []


def test_both_forms_covered_when_a_span_matches_each():
    gt = [
        _row("shared", "066", "instrumental", 1552.0, 1599.0),
        _row("shared", "066", "regular", 1599.0, 1638.0),
    ]
    spans = [
        {"recording_id": "shared", "slot_label": "66", "set_start_s": 1555.0},
        {"recording_id": "shared", "slot_label": "66", "set_start_s": 1620.0},
    ]
    assert unmatched_gt_forms(gt, spans) == []


def test_single_form_matched_recording_is_covered():
    gt = [_row("solo", "010", "regular", 100.0, 140.0)]
    spans = [{"recording_id": "solo", "slot_label": "10", "set_start_s": 102.0}]
    assert unmatched_gt_forms(gt, spans) == []


def test_never_matched_recording_forms_are_all_unmatched():
    gt = [
        _row("gone", "097", "acappella", 300.0, 340.0),
        _row(None, "200", "regular", 500.0, 540.0),  # trackless -> skipped
    ]
    spans = [{"recording_id": "other", "slot_label": "5", "set_start_s": 10.0}]
    out = unmatched_gt_forms(gt, spans)
    assert {e["claimed_stem"] for e in out} == {"acappella"}  # trackless skipped
