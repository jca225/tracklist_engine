"""Segment bank → shadow timeline materialization."""

from __future__ import annotations

import json
from pathlib import Path

from core.contracts import load_timeline
from alignment.fp_segments.materialize import (
    materialize_timeline,
)


def test_materialize_applies_decoded_segments_and_preserves_abstains(tmp_path: Path):
    baseline = {
        "set_id": "demo",
        "spans": [
            {
                "slot_label": "1",
                "recording_id": "r1",
                "claimed_stem": "instrumental",
                "set_start_s": 10.0,
                "set_end_s": 40.0,
                "ref_start_s": 0.0,
                "ref_end_s": 30.0,
                "start_source": "baseline",
                "ref_segments": [],
            },
            {
                "slot_label": "2",
                "recording_id": "r2",
                "claimed_stem": "regular",
                "set_start_s": 50.0,
                "set_end_s": 80.0,
                "ref_start_s": 5.0,
                "ref_end_s": 35.0,
                "start_source": "baseline",
                "ref_segments": [],
            },
        ],
    }
    bank = {
        "set_id": "demo",
        "spans": [
            {
                "slot_label": "1",
                "recording_id": "r1",
                "stem": "instrumental",
                "status": "decoded",
                "segments": [
                    {
                        "mix_start_s": 12.0,
                        "mix_end_s": 22.0,
                        "ref_start_s": 1.0,
                        "ref_end_s": 11.0,
                        "slope": 1.0,
                        "evidence": 10.0,
                        "confidence": 0.5,
                    },
                    {
                        "mix_start_s": 30.0,
                        "mix_end_s": 38.0,
                        "ref_start_s": 20.0,
                        "ref_end_s": 28.0,
                        "slope": 1.0,
                        "evidence": 8.0,
                        "confidence": 0.4,
                    },
                ],
            },
            {
                "slot_label": "2",
                "recording_id": "r2",
                "stem": "instrumental",
                "status": "abstained",
                "segments": [],
            },
        ],
    }
    out = tmp_path / "shadow.json"
    materialize_timeline(baseline, bank, out)
    load_timeline(out)
    doc = json.loads(out.read_text())
    s0, s1 = doc["spans"]
    assert s0["set_start_s"] == 12.0
    assert s0["set_end_s"] == 38.0
    assert s0["start_source"] == "fp_segment_dp"
    assert s0["recording_id"] == "r1"
    assert len(s0["ref_segments"]) == 2
    assert s0["ref_segments"][0]["mix_start_s"] == 12.0
    assert s0["ref_segments"][0]["ref_start_s"] == 1.0
    assert s0["ref_segments"][0]["ref_end_s"] == 11.0
    # Abstained: baseline preserved.
    assert s1["set_start_s"] == 50.0
    assert s1["start_source"] == "baseline"
    assert s1["ref_segments"] == []


def test_gate_rejects_far_proposal_and_keeps_near(tmp_path: Path):
    baseline = {
        "set_id": "demo",
        "spans": [
            {
                "slot_label": "1",
                "recording_id": "r1",
                "claimed_stem": "instrumental",
                "set_start_s": 100.0,
                "set_end_s": 140.0,
                "ref_start_s": 0.0,
                "ref_end_s": 40.0,
                "start_source": "baseline",
                "ref_segments": [],
            },
            {
                "slot_label": "2",
                "recording_id": "r2",
                "claimed_stem": "instrumental",
                "set_start_s": 200.0,
                "set_end_s": 240.0,
                "ref_start_s": 0.0,
                "ref_end_s": 40.0,
                "start_source": "baseline",
                "ref_segments": [],
            },
        ],
    }
    bank = {
        "set_id": "demo",
        "spans": [
            {
                "slot_label": "1",
                "recording_id": "r1",
                "stem": "instrumental",
                "status": "decoded",
                "segments": [
                    {
                        "mix_start_s": 2000.0,
                        "mix_end_s": 2040.0,
                        "ref_start_s": 0.0,
                        "ref_end_s": 40.0,
                        "slope": 1.0,
                        "evidence": 99.0,
                        "confidence": 0.9,
                    }
                ],
            },
            {
                "slot_label": "2",
                "recording_id": "r2",
                "stem": "instrumental",
                "status": "decoded",
                "segments": [
                    {
                        "mix_start_s": 205.0,
                        "mix_end_s": 235.0,
                        "ref_start_s": 2.0,
                        "ref_end_s": 32.0,
                        "slope": 1.0,
                        "evidence": 20.0,
                        "confidence": 0.5,
                    },
                    {
                        # Far false extra — must not be applied under the gate.
                        "mix_start_s": 900.0,
                        "mix_end_s": 930.0,
                        "ref_start_s": 0.0,
                        "ref_end_s": 30.0,
                        "slope": 1.0,
                        "evidence": 50.0,
                        "confidence": 0.8,
                    },
                ],
            },
        ],
    }
    out = tmp_path / "gated.json"
    materialize_timeline(baseline, bank, out, gate_s=90.0)
    doc = json.loads(out.read_text())
    meta = doc["fp_segment_materialize"]
    assert meta["gate_s"] == 90.0
    assert meta["spans_applied"] == 1
    assert meta["spans_gated_reject"] == 1
    # Far proposal rejected.
    assert doc["spans"][0]["set_start_s"] == 100.0
    assert doc["spans"][0]["start_source"] == "baseline"
    # Near proposal applied; far extra dropped.
    assert doc["spans"][1]["set_start_s"] == 205.0
    assert doc["spans"][1]["set_end_s"] == 235.0
    assert len(doc["spans"][1]["ref_segments"]) == 1
    assert doc["spans"][1]["start_source"] == "fp_segment_dp"


def test_ref_segments_only_preserves_set_start(tmp_path: Path):
    baseline = {
        "set_id": "demo",
        "spans": [
            {
                "slot_label": "1",
                "recording_id": "r1",
                "claimed_stem": "instrumental",
                "set_start_s": 100.0,
                "set_end_s": 140.0,
                "ref_start_s": 0.0,
                "ref_end_s": 40.0,
                "start_source": "baseline",
                "ref_segments": [],
            }
        ],
    }
    bank = {
        "set_id": "demo",
        "spans": [
            {
                "slot_label": "1",
                "recording_id": "r1",
                "stem": "instrumental",
                "status": "decoded",
                "segments": [
                    {
                        "mix_start_s": 105.0,
                        "mix_end_s": 130.0,
                        "ref_start_s": 10.0,
                        "ref_end_s": 35.0,
                        "slope": 1.0,
                        "evidence": 12.0,
                        "confidence": 0.4,
                    }
                ],
            }
        ],
    }
    out = tmp_path / "ref_only.json"
    materialize_timeline(baseline, bank, out, gate_s=90.0, ref_segments_only=True)
    doc = json.loads(out.read_text())
    span = doc["spans"][0]
    assert span["set_start_s"] == 100.0
    assert span["set_end_s"] == 140.0
    assert span["ref_start_s"] == 10.0
    assert span["ref_end_s"] == 35.0
    assert span["start_source"] == "fp_segment_dp_ref"
    assert len(span["ref_segments"]) == 1
    assert doc["fp_segment_materialize"]["ref_segments_only"] is True
