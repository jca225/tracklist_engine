from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from lxml import etree

from workspaces.source_detection import als_audit


def test_parse_als_prefers_exact_primary_mix_lane(monkeypatch, tmp_path):
    root = etree.fromstring(
        b"""
        <Ableton><LiveSet><Tracks>
          <AudioTrack><Name Value="2-mix"/></AudioTrack>
          <AudioTrack><Name Value="1-mix"/></AudioTrack>
        </Tracks></LiveSet></Ableton>
        """
    )
    selected = {}

    monkeypatch.setattr(als_audit.als_io, "load_als_xml", lambda _path: root)
    monkeypatch.setattr(als_audit.als_io, "parse_layer_clips", lambda _root: [])

    def choose(_root, mix_track, **_kwargs):
        selected["name"] = als_audit.als_io.track_display_name(mix_track)
        return SimpleNamespace(arr_to_set_sec=lambda _beat: None)

    monkeypatch.setattr(als_audit.als_io, "select_arrangement_mapper", choose)

    facts, dropped = als_audit.parse_als(tmp_path / "set.als", tmp_path, {})

    assert facts == []
    assert dropped == []
    assert selected["name"] == "1-mix"


def test_placed_window_score_uses_declared_tolerance_neighborhood():
    curve = np.array([0.1, 0.2, 0.7, 0.3, 0.1])

    assert als_audit._placed_window_score(curve, center=1, radius=1) == 0.7
    assert als_audit._placed_window_score(curve, center=4, radius=1) == 0.3
    assert als_audit._placed_window_score(curve, center=9, radius=1) == -2.0
