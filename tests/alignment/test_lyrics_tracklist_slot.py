"""Manifest slot resolution for lyrics position priors (BB10 rid-keyed timelines)."""

from __future__ import annotations

from alignment.lyrics_align import (
    _slot_order,
    resolve_tracklist_slot,
    tracklist_max_slot,
)


def test_resolve_tracklist_slot_prefers_manifest_ableton_label():
    assert (
        resolve_tracklist_slot(
            "h5u3jtp", {"slot_label": "001w2", "track_id": "h5u3jtp"}
        )
        == "001w2"
    )
    assert resolve_tracklist_slot("001w2", None) == "001w2"
    assert resolve_tracklist_slot("h5u3jtp", {"track_id": "h5u3jtp"}) == "h5u3jtp"


def test_rid_style_slots_collapse_max_slot_manifest_does_not():
    rid_slots = ["h5u3jtp", "89vy0x5", "20l5t595", "mvnq63x"]
    man_slots = ["001w2", "027w2", "015w1", "002w1", "040"]
    # Digit-stripping rid keys invent a huge false max (BB10 smoke: ~698k).
    assert tracklist_max_slot(rid_slots) > 1000
    assert tracklist_max_slot(man_slots) == 40
    mix_dur = 3664.0
    epos_rid = _slot_order("h5u3jtp")[0] / tracklist_max_slot(rid_slots) * mix_dur
    epos_man = _slot_order("001w2")[0] / tracklist_max_slot(man_slots) * mix_dur
    assert epos_rid < 1.0
    assert 50.0 < epos_man < 150.0
