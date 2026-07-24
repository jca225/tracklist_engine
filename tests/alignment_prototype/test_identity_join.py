"""Guards the identity-join finding: slot_labels are NOT a cross-source key
(GT vs set_track_slots number slots differently). Locks in the fix for the "4%
identity" hand-rolled-join trap.
"""

from __future__ import annotations

from workspaces.alignment_prototype import identity_join


def test_bridge_uses_idmap():
    # a bare spine tlp number bridges to its canonical recording in the BB12 map
    # (self-maps are the identity; the point is bridge() never raises + is total)
    assert identity_join.bridge("nonexistent-id", "1fsnxchk") == "nonexistent-id"


def test_slots_do_not_share_songs():
    """The regression guard: if this ever rises toward 1.0, slot_labels became a
    valid join key and the module's premise (and the scorer's time-join) should
    be revisited. On current data it is near-0."""
    tl = {
        "spans": [
            {"slot_label": "1", "recording_id": "a", "name": "Manse - Freeze Time"},
            {"slot_label": "2", "recording_id": "b", "name": "Mako - Our Story"},
        ]
    }
    gt = {
        "tracks": [
            {"slot_label": "1", "track_id": "x", "track": "Post Malone - Congrats"},
            {"slot_label": "2", "track_id": "y", "track": "Avicii - Fade"},
        ]
    }
    assert identity_join.slots_share_songs(tl, gt) < 0.5


def test_slots_share_songs_detects_real_alignment():
    """Sanity: when labels DO map to the same song, the metric is high — so a
    near-0 result is a real signal, not a broken metric."""
    tl = {
        "spans": [{"slot_label": "1", "recording_id": "a", "name": "Avicii - Levels"}]
    }
    gt = {"tracks": [{"slot_label": "1", "track_id": "x", "track": "Avicii - Levels"}]}
    assert identity_join.slots_share_songs(tl, gt) == 1.0


def test_recording_bags_bridge():
    tl = {"spans": [{"recording_id": "a"}, {"recording_id": "b"}]}
    gt = {"tracks": [{"slot_label": "1", "track_id": "a"}, {"slot_label": "mix"}]}
    assert identity_join.pred_recordings(tl, "1fsnxchk") == {"a", "b"}
    assert identity_join.gt_recordings(gt, "1fsnxchk") == {"a"}
