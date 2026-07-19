"""Tests for GT slot_label remapping to pi tracklist labels."""

from __future__ import annotations

from labeling.remap_gt_slot_labels import RemapEvent, remap_slots


def test_remap_heiress_and_saints():
    gt = {
        "tracks": [
            {
                "track": "Heiress",
                "slot_label": "132",
                "track_id": "94tc2y5",
                "claimed_stem": "instrumental",
            },
            {
                "track": "Saints",
                "slot_label": "076",
                "track_id": "1r8f4fc5",
                "claimed_stem": "instrumental",
            },
        ]
    }
    pi = [
        {
            "slot_label": "037",
            "recording_id": "94tc2y5",
            "claimed_stem": "instrumental",
        },
        {
            "slot_label": "022",
            "recording_id": "1r8f4fc5",
            "claimed_stem": "instrumental",
        },
    ]
    out, events = remap_slots(gt, pi)
    assert out["tracks"][0]["slot_label"] == "037"
    assert out["tracks"][1]["slot_label"] == "022"
    assert len(events) == 2
    assert all(e.status == "remapped" for e in events)


def test_disambiguate_duplicate_track_id_by_claimed_stem():
    gt = {
        "tracks": [
            {
                "track": "Song A vocal",
                "slot_label": "010",
                "track_id": "abc123",
                "claimed_stem": "acappella",
                "set_start_s": 100.0,
            },
            {
                "track": "Song A inst",
                "slot_label": "011",
                "track_id": "abc123",
                "claimed_stem": "instrumental",
                "set_start_s": 200.0,
            },
        ]
    }
    pi = [
        {
            "slot_label": "003",
            "recording_id": "abc123",
            "claimed_stem": "instrumental",
            "row_index": 3,
            "cue_time_seconds": 195,
        },
        {
            "slot_label": "002",
            "recording_id": "abc123",
            "claimed_stem": "acappella",
            "row_index": 2,
            "cue_time_seconds": 98,
        },
    ]
    out, events = remap_slots(gt, pi)
    assert out["tracks"][0]["slot_label"] == "002"
    assert out["tracks"][1]["slot_label"] == "003"
    assert len([e for e in events if e.status == "remapped"]) == 2


def test_needs_human_when_still_ambiguous():
    gt = {
        "tracks": [
            {
                "track": "Dup",
                "slot_label": "099",
                "track_id": "xyz",
                "claimed_stem": "regular",
                "set_start_s": 100.0,
            },
        ]
    }
    pi = [
        {
            "slot_label": "001",
            "recording_id": "xyz",
            "claimed_stem": "regular",
            "row_index": 5,
            "cue_time_seconds": 100,
        },
        {
            "slot_label": "002",
            "recording_id": "xyz",
            "claimed_stem": "regular",
            "row_index": 5,
            "cue_time_seconds": 100,
        },
    ]
    out, events = remap_slots(gt, pi)
    assert out["tracks"][0]["slot_label"] == "099"
    assert len(events) == 1
    assert events[0].status == "needs_human"


def test_unchanged_when_already_matches():
    gt = {
        "tracks": [
            {
                "track": "OK",
                "slot_label": "005",
                "track_id": "tid1",
                "claimed_stem": "regular",
            },
        ]
    }
    pi = [
        {"slot_label": "005", "recording_id": "tid1", "claimed_stem": "regular"},
    ]
    out, events = remap_slots(gt, pi)
    assert out["tracks"][0]["slot_label"] == "005"
    assert events == []
