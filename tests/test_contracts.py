"""core.contracts — loader validation, slot normal form, join guard."""

from __future__ import annotations

import json

import msgspec
import pytest

from core.contracts import (
    SetMismatchError,
    join_guard,
    load_timeline,
    normalize_slot_label,
)


def test_normalize_slot_label_canonical_form():
    assert normalize_slot_label("001w2") == "1w2"
    assert normalize_slot_label("042") == "42"
    assert normalize_slot_label("7") == "7"
    assert normalize_slot_label(" 013w1 ") == "13w1"
    assert normalize_slot_label("mix") is None
    assert normalize_slot_label("") is None
    assert normalize_slot_label(None) is None


def test_join_guard_same_set_passes():
    assert join_guard("2nvzlh2k", "2nvzlh2k") == "2nvzlh2k"


def test_join_guard_cross_set_raises():
    with pytest.raises(SetMismatchError):
        join_guard("2nvzlh2k", "1fsnxchk", context="test")


def test_join_guard_empty_raises():
    with pytest.raises(SetMismatchError):
        join_guard("2nvzlh2k", "")


def _minimal_timeline(**overrides) -> dict:
    doc = {
        "set_id": "2nvzlh2k",
        "spans": [
            {
                "slot_label": "001w2",
                "recording_id": "abc123",
                "set_start_s": 10.0,
                "set_end_s": 20.0,
                "ref_start_s": 5.0,
                "extra_provenance_field": {"ignored": True},  # writer not migrated
            }
        ],
    }
    doc.update(overrides)
    return doc


def test_load_timeline_valid(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps(_minimal_timeline()))
    tl = load_timeline(p)
    assert tl.set_id == "2nvzlh2k"
    assert tl.spans[0].slot == "1w2"  # normalized at the boundary
    assert tl.spans[0].recording_id == "abc123"


def test_load_timeline_missing_required_field_fails(tmp_path):
    doc = _minimal_timeline()
    del doc["spans"][0]["set_start_s"]
    p = tmp_path / "t.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(msgspec.ValidationError):
        load_timeline(p)


def test_load_timeline_mistyped_field_fails(tmp_path):
    doc = _minimal_timeline()
    doc["spans"][0]["set_start_s"] = "ten seconds"
    p = tmp_path / "t.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(msgspec.ValidationError):
        load_timeline(p)


def test_load_timeline_empty_set_id_fails(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps(_minimal_timeline(set_id="")))
    with pytest.raises(msgspec.ValidationError):
        load_timeline(p)


def test_load_manifest_all_eras(tmp_path):
    from core.contracts import load_manifest

    doc = {
        "set_id": "w1mgcjt",
        "tracks": [
            {  # current era
                "track_id": "abc",
                "track_audio_id": 5,
                "label": "001w1",
                "slot_label": "001w1",
                "stem": "regular",
                "stems": {"vocals": "/x/vocals.flac"},
            },
            {"track_id": "tlp123", "version_tag": "Remix"},  # pre-axes era
        ],
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(doc))
    m = load_manifest(p)
    assert m.sid == "w1mgcjt"
    assert m.tracks[0].slot == "1w1"
    assert m.tracks[1].slot is None
    assert m.by_track_id()["abc"].stems == {"vocals": "/x/vocals.flac"}
