"""ChromaProbe: correlation peak -> clamped [0,1] confidence with abstention on a
weak match, and the probe satisfies the contract without touching audio."""

from __future__ import annotations

from pathlib import Path

import workspaces.alignment_prototype.harness.chroma_probe as cp
from workspaces.alignment_prototype.harness import (
    AlignmentResult,
    CandidatePool,
    MixContext,
    RefContext,
)


def test_strong_peak_becomes_confident_result():
    r = cp.chroma_result_to_alignment(
        30.0, peak=0.82, stretch=1.05, recording_id="rec1"
    )
    assert not r.abstain
    assert r.offset_s == 30.0 and r.tempo_ratio == 1.05
    assert r.confidence == 0.82 and r.source == "chroma"


def test_weak_peak_abstains():
    r = cp.chroma_result_to_alignment(5.0, peak=0.31, stretch=1.0, recording_id="rec1")
    assert r.abstain and r.confidence == 0.0


def test_peak_clamped_into_unit_interval():
    # A correlation score slightly out of [0,1] must not violate the contract.
    hi = cp.chroma_result_to_alignment(0.0, peak=1.4, stretch=1.0, recording_id="r")
    assert hi.confidence == 1.0 and not hi.abstain
    lo = cp.chroma_result_to_alignment(0.0, peak=-0.2, stretch=1.0, recording_id="r")
    assert lo.abstain  # clamps to 0 -> below floor -> abstain


def test_run_wires_extractors_and_detect_offset(monkeypatch):
    monkeypatch.setattr(
        cp, "detect_offset", lambda win, ref, stretches: (42.0, 0.9, 0.98)
    )
    probe = cp.ChromaProbe(
        mix_chroma=lambda m: "MIXFEATS", ref_chroma=lambda r: "REFFEATS"
    )
    out = probe(
        MixContext(audio_path=Path("/tmp/mix.wav"), span_start_s=10.0, span_end_s=40.0),
        RefContext(recording_id="recY", audio_path=Path("/tmp/ref.wav")),
        CandidatePool(),
    )
    assert isinstance(out, AlignmentResult)
    assert out.recording_id == "recY" and out.offset_s == 42.0
    assert out.tempo_ratio == 0.98 and not out.abstain and out.source == "chroma"


def test_run_abstains_on_weak_correlation(monkeypatch):
    monkeypatch.setattr(
        cp, "detect_offset", lambda win, ref, stretches: (0.0, 0.2, 1.0)
    )
    probe = cp.ChromaProbe(mix_chroma=lambda m: "x", ref_chroma=lambda r: "y")
    out = probe(
        MixContext(audio_path=Path("/tmp/m.wav")),
        RefContext(recording_id="r", audio_path=Path("/tmp/r.wav")),
        CandidatePool(),
    )
    assert out.abstain
