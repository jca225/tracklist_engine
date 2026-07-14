"""TDD tests for export_votes — per-probe AlignmentResults -> probe_votes.json.

Three levels:
  1. Unit — span_to_votes_doc() on synthetic span dicts (no audio, no files).
  2. File I/O — write_votes_file() / load_predicted_timeline() round-trip.
  3. End-to-end round-trip — export_votes() output feeds run_phase1() and
     produces a valid pws_timeline.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from workspaces.pws_aligner.export_votes import (
    ProbeEntry,
    SpanVotesDoc,
    _DEFAULT_CONFIDENCE,
    _KNOWN_PROBES,
    export_votes,
    span_to_votes_doc,
    timeline_to_votes_docs,
    write_votes_file,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_SPAN_FULL = {
    "slot_label": "3",
    "recording_id": "rec_abc",
    "claimed_stem": "regular",
    "set_start_s": 100.0,
    "set_end_s": 140.0,
    "ref_start_s": 12.0,
    "ref_end_s": 52.0,
    "confidence": 0.72,
    "name": "Track Alpha",
    "probe_proposals": {
        "mert_decode": 100.0,  # same as set_start_s — won the merge
        "fp": 98.0,  # fp proposed slightly earlier set_start
        "lyrics": 101.5,  # lyrics proposed slightly later
    },
}

_SPAN_NO_PROPOSALS = {
    "slot_label": "7",
    "recording_id": "rec_xyz",
    "claimed_stem": "acappella",
    "set_start_s": 200.0,
    "set_end_s": 230.0,
    "ref_start_s": 5.0,
    "ref_end_s": None,
    "confidence": 0.4,
    "name": "Track Beta",
    # no probe_proposals key at all
}


# ---------------------------------------------------------------------------
# 1. Unit tests for span_to_votes_doc
# ---------------------------------------------------------------------------


def test_span_to_votes_doc_basic_fields():
    """Pass-through fields are copied correctly."""
    doc = span_to_votes_doc(_SPAN_FULL, span_index=0)

    assert isinstance(doc, SpanVotesDoc)
    assert doc.span_id == "3"
    assert doc.slot_label == "3"
    assert doc.recording_id == "rec_abc"
    assert doc.claimed_stem == "regular"
    assert doc.set_start_s == 100.0
    assert doc.set_end_s == 140.0
    assert doc.ref_start_s == 12.0
    assert doc.ref_end_s == 52.0
    assert doc.confidence == pytest.approx(0.72)
    assert doc.name == "Track Alpha"


def test_span_to_votes_doc_probe_count():
    """All _KNOWN_PROBES are emitted (even those absent from proposals)."""
    doc = span_to_votes_doc(_SPAN_FULL, span_index=0)
    probe_names_in_doc = [p.probe for p in doc.probes]
    # Every known probe must appear
    for name in _KNOWN_PROBES:
        assert name in probe_names_in_doc, f"{name!r} missing from probes"


def test_span_to_votes_doc_present_probe_not_abstain():
    """A probe that appears in probe_proposals must be non-abstaining."""
    doc = span_to_votes_doc(_SPAN_FULL, span_index=0)
    by_name = {p.probe: p for p in doc.probes}

    for probe_name in ("mert_decode", "fp", "lyrics"):
        entry = by_name[probe_name]
        assert not entry.abstain, f"{probe_name!r} should not abstain"
        assert entry.confidence == pytest.approx(_DEFAULT_CONFIDENCE)
        assert entry.recording_id == "rec_abc"


def test_span_to_votes_doc_absent_probe_abstains():
    """A known probe absent from probe_proposals must abstain."""
    doc = span_to_votes_doc(_SPAN_FULL, span_index=0)
    by_name = {p.probe: p for p in doc.probes}

    for probe_name in ("stem_hubert", "instr_fp"):
        entry = by_name[probe_name]
        assert entry.abstain, f"{probe_name!r} should abstain (absent from proposals)"
        assert entry.confidence == 0.0
        assert entry.recording_id is None


def test_span_to_votes_doc_offset_s_formula():
    """offset_s = ref_start_s - probe_set_start_s for each non-abstaining probe."""
    doc = span_to_votes_doc(_SPAN_FULL, span_index=0)
    by_name = {p.probe: p for p in doc.probes}

    ref_start_s = _SPAN_FULL["ref_start_s"]
    proposals = _SPAN_FULL["probe_proposals"]

    for probe_name in ("mert_decode", "fp", "lyrics"):
        expected_offset = ref_start_s - proposals[probe_name]
        assert by_name[probe_name].offset_s == pytest.approx(expected_offset), (
            f"{probe_name!r} offset_s mismatch"
        )


def test_span_to_votes_doc_features_empty():
    """features is always empty (neuro sharpness proxies not available offline)."""
    doc = span_to_votes_doc(_SPAN_FULL, span_index=0)
    for entry in doc.probes:
        assert entry.features == (), f"{entry.probe!r} features must be empty tuple"


def test_span_to_votes_doc_no_proposals_all_abstain():
    """When probe_proposals is absent, all known probes must abstain."""
    doc = span_to_votes_doc(_SPAN_NO_PROPOSALS, span_index=6)
    assert all(p.abstain for p in doc.probes), (
        "all probes must abstain with no proposals"
    )
    assert all(p.recording_id is None for p in doc.probes)


def test_span_to_votes_doc_missing_optional_fields():
    """Missing ref_end_s / name / claimed_stem fall back gracefully."""
    minimal = {
        "slot_label": "1",
        "set_start_s": 0.0,
        "set_end_s": 30.0,
        "ref_start_s": 0.0,
        "confidence": 0.0,
        "probe_proposals": {},
    }
    doc = span_to_votes_doc(minimal, span_index=0)
    assert doc.ref_end_s is None
    assert doc.name == ""
    assert doc.claimed_stem == "regular"
    assert doc.recording_id is None


# ---------------------------------------------------------------------------
# 2. to_dict() serialization
# ---------------------------------------------------------------------------


def test_span_votes_doc_to_dict_schema():
    """to_dict() matches the probe_votes.json schema exactly."""
    doc = span_to_votes_doc(_SPAN_FULL, span_index=0)
    d = doc.to_dict()

    required_top = {
        "span_id",
        "slot_label",
        "recording_id",
        "claimed_stem",
        "set_start_s",
        "set_end_s",
        "ref_start_s",
        "ref_end_s",
        "confidence",
        "name",
        "probes",
    }
    assert required_top <= d.keys(), f"missing keys: {required_top - d.keys()}"

    required_probe = {
        "probe",
        "recording_id",
        "offset_s",
        "confidence",
        "abstain",
        "features",
    }
    for entry in d["probes"]:
        assert required_probe <= entry.keys(), (
            f"probe missing keys: {required_probe - entry.keys()}"
        )
        assert isinstance(entry["features"], list)


# ---------------------------------------------------------------------------
# 3. File I/O
# ---------------------------------------------------------------------------


def _make_fake_timeline(tmp_path: Path, set_id: str = "fake_set") -> Path:
    """Write a minimal predicted_timeline.json with two spans."""
    payload = {
        "set_id": set_id,
        "spans": [
            {
                "slot_label": "1",
                "recording_id": "r1",
                "claimed_stem": "regular",
                "set_start_s": 10.0,
                "set_end_s": 50.0,
                "ref_start_s": 8.0,
                "ref_end_s": 48.0,
                "confidence": 0.8,
                "name": "Track One",
                "probe_proposals": {"mert_decode": 10.0, "fp": 11.5},
            },
            {
                "slot_label": "2",
                "recording_id": "r2",
                "claimed_stem": "acappella",
                "set_start_s": 50.0,
                "set_end_s": 90.0,
                "ref_start_s": 3.0,
                "ref_end_s": None,
                "confidence": 0.6,
                "name": "Track Two",
                "probe_proposals": {"mert_decode": 50.0, "lyrics": 49.0},
            },
        ],
        "skipped": [],
    }
    path = tmp_path / f"{set_id}_predicted_timeline.json"
    path.write_text(json.dumps(payload))
    return path


def test_write_votes_file_produces_valid_json(tmp_path):
    """write_votes_file emits a JSON array."""
    timeline_path = _make_fake_timeline(tmp_path)
    raw = json.loads(timeline_path.read_text())
    docs = timeline_to_votes_docs(raw)

    out = tmp_path / "out" / "fake_set_probe_votes.json"
    write_votes_file(docs, out)

    assert out.exists()
    loaded = json.loads(out.read_text())
    assert isinstance(loaded, list)
    assert len(loaded) == 2


def test_write_votes_file_span_fields_roundtrip(tmp_path):
    """All span-level fields survive write -> load -> compare."""
    timeline_path = _make_fake_timeline(tmp_path)
    raw = json.loads(timeline_path.read_text())
    docs = timeline_to_votes_docs(raw)

    out = tmp_path / "fake_set_probe_votes.json"
    write_votes_file(docs, out)

    loaded = json.loads(out.read_text())
    s1 = loaded[0]
    assert s1["slot_label"] == "1"
    assert s1["recording_id"] == "r1"
    assert s1["set_start_s"] == 10.0
    assert s1["ref_start_s"] == 8.0
    assert s1["name"] == "Track One"

    s2 = loaded[1]
    assert s2["slot_label"] == "2"
    assert s2["claimed_stem"] == "acappella"
    assert s2["ref_end_s"] is None


# ---------------------------------------------------------------------------
# 4. End-to-end round-trip: export_votes -> run_phase1 -> pws_timeline.json
# ---------------------------------------------------------------------------


def test_export_votes_to_run_phase1_roundtrip(tmp_path):
    """Full pipeline: fake predicted_timeline -> export_votes -> run_phase1.

    This test drives both sides of the schema contract: the exporter must
    produce a votes file that run_phase1's loader accepts without error, and
    the resulting pws_timeline.json must carry required span fields.
    """
    from workspaces.pws_aligner.run_phase1 import run_phase1

    set_id = "roundtrip_set"
    timeline_path = _make_fake_timeline(tmp_path, set_id)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # Step 1: export votes
    votes_path = export_votes(
        set_id,
        timeline_path=timeline_path,
        out_dir=out_dir,
    )
    assert votes_path.exists(), "export_votes must write the votes file"
    votes_raw = json.loads(votes_path.read_text())
    assert isinstance(votes_raw, list)
    assert len(votes_raw) == 2

    # Verify schema of the votes file before feeding to run_phase1
    for span_doc in votes_raw:
        assert "span_id" in span_doc
        assert "slot_label" in span_doc
        assert "probes" in span_doc
        for probe in span_doc["probes"]:
            assert "probe" in probe
            assert "recording_id" in probe
            assert "offset_s" in probe
            assert "confidence" in probe
            assert "abstain" in probe
            assert "features" in probe
            assert isinstance(probe["features"], list)

    # Step 2: run_phase1 consumes the votes file
    tl_path = run_phase1(
        set_id,
        votes_path=votes_path,
        out_dir=out_dir,
        bin_s=2.0,
    )
    assert tl_path.exists(), "run_phase1 must write the timeline"

    tl = json.loads(tl_path.read_text())
    assert tl["set_id"] == set_id
    spans = tl["spans"]
    assert len(spans) == 2

    # Required fields in every pws_timeline span
    for s in spans:
        assert "slot_label" in s
        assert "recording_id" in s
        assert "set_start_s" in s
        assert "ref_start_s" in s
        assert "confidence" in s
        assert "abstain" in s


def test_export_votes_all_abstain_roundtrip(tmp_path):
    """All-abstain span (no probe_proposals) goes through run_phase1 without error."""
    from workspaces.pws_aligner.run_phase1 import run_phase1

    set_id = "abstain_set2"
    payload = {
        "set_id": set_id,
        "spans": [
            {
                "slot_label": "1",
                "recording_id": "r1",
                "claimed_stem": "regular",
                "set_start_s": 20.0,
                "set_end_s": 60.0,
                "ref_start_s": 5.0,
                "ref_end_s": 45.0,
                "confidence": 0.0,
                "name": "Track Z",
                # no probe_proposals — all probes will abstain
            },
        ],
        "skipped": [],
    }
    timeline_path = tmp_path / f"{set_id}_predicted_timeline.json"
    timeline_path.write_text(json.dumps(payload))

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    votes_path = export_votes(set_id, timeline_path=timeline_path, out_dir=out_dir)
    votes_raw = json.loads(votes_path.read_text())
    assert all(p["abstain"] for span in votes_raw for p in span["probes"])

    tl_path = run_phase1(set_id, votes_path=votes_path, out_dir=out_dir)
    tl = json.loads(tl_path.read_text())
    assert tl["spans"][0]["abstain"] is True
    assert tl["spans"][0]["recording_id"] == ""  # empty-string sentinel for abstain
