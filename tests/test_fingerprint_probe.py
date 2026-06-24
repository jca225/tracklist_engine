"""FingerprintProbe: the raw (votes, sharpness) signal maps to a bounded, monotone
confidence with explicit abstention — and the probe satisfies the harness contract
without touching audio."""

from __future__ import annotations

from pathlib import Path

import pytest

import workspaces.alignment_prototype.harness.fingerprint_probe as fp
from workspaces.alignment_prototype.harness import (
    AlignmentResult,
    CandidatePool,
    MixContext,
    RefContext,
)


def test_votes_to_confidence_monotone_and_bounded():
    cs = [fp.votes_to_confidence(v) for v in (0, 1, 10, 50, 100, 400, 5000)]
    assert cs[0] == 0.0
    assert all(0.0 <= c <= 1.0 for c in cs)
    assert cs == sorted(cs)  # non-decreasing in votes
    assert cs[-1] > 0.99  # saturates toward 1 for a strong hit


def test_strong_hit_becomes_confident_result():
    r = fp.fp_result_to_alignment(
        12.5, votes=400, stretch=1.02, sharpness=3.0, recording_id="rec1"
    )
    assert not r.abstain
    assert r.offset_s == 12.5 and r.tempo_ratio == 1.02
    assert r.confidence > 0.9 and r.source == "fp"


def test_low_votes_abstains():
    r = fp.fp_result_to_alignment(
        9.0, votes=3, stretch=1.0, sharpness=5.0, recording_id="rec1"
    )
    assert r.abstain and r.confidence == 0.0 and r.recording_id == "rec1"


def test_flat_peak_abstains_even_with_votes():
    # Enough votes but the peak doesn't beat the runner-up -> not trustworthy.
    r = fp.fp_result_to_alignment(
        9.0, votes=200, stretch=1.0, sharpness=1.05, recording_id="rec1"
    )
    assert r.abstain


def test_run_wires_loader_and_fp_offset(monkeypatch):
    # Stub the audio loader and fp_offset so no librosa / real audio is needed.
    monkeypatch.setattr(
        fp, "fp_offset", lambda mix_y, ref_y, stretches: (7.0, 300, 1.0, 4.0)
    )
    probe = fp.FingerprintProbe(loader=lambda path: [0.0])
    mix = MixContext(audio_path=Path("/tmp/mix.wav"))
    ref = RefContext(recording_id="recX", audio_path=Path("/tmp/ref.wav"))
    out = probe(mix, ref, CandidatePool())
    assert isinstance(out, AlignmentResult)
    assert out.recording_id == "recX" and out.offset_s == 7.0 and not out.abstain
    assert out.source == "fp"


def test_run_abstains_on_weak_match(monkeypatch):
    monkeypatch.setattr(
        fp, "fp_offset", lambda mix_y, ref_y, stretches: (0.0, 1, 1.0, 1.0)
    )
    probe = fp.FingerprintProbe(loader=lambda path: [0.0])
    out = probe(
        MixContext(audio_path=Path("/tmp/m.wav")),
        RefContext(recording_id="r", audio_path=Path("/tmp/r.wav")),
        CandidatePool(),
    )
    assert out.abstain
