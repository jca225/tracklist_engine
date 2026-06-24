"""PathDecodeProbe: piecewise segments land in AlignmentResult.segments; confidence
is mean per-frame path quality with abstention; satisfies the contract w/o audio."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import workspaces.alignment_prototype.harness.path_decode_probe as pdp
from workspaces.alignment_prototype.harness import (
    AlignmentResult,
    CandidatePool,
    MixContext,
    RefContext,
)


def test_score_to_confidence_is_mean_and_clamped():
    assert pdp.score_to_confidence(80.0, 100) == 0.8
    assert pdp.score_to_confidence(500.0, 100) == 1.0  # clamp high
    assert pdp.score_to_confidence(-5.0, 100) == 0.0  # clamp low
    assert pdp.score_to_confidence(5.0, 0) == 1.0  # n clamped to >=1


def test_single_segment_summarized():
    segs = [(0.0, 12.0, 24.0)]
    r = pdp.path_result_to_alignment(segs, 80.0, n_mix_frames=100, recording_id="rec1")
    assert not r.abstain
    assert r.offset_s == 12.0 and r.ref_end_s == 24.0 and len(r.segments) == 1


def test_loop_produces_multiple_segments():
    # A loop: the span maps to the same ref region twice -> two segments.
    segs = [(0.0, 4.0, 8.0), (4.0, 4.0, 8.0)]
    r = pdp.path_result_to_alignment(segs, 90.0, n_mix_frames=100, recording_id="rec1")
    assert len(r.segments) == 2
    assert r.offset_s == 4.0  # first segment ref_start
    assert r.ref_end_s == 8.0  # last segment ref_end
    assert r.segments[1].mix_start_s == 4.0


def test_no_segments_abstains():
    r = pdp.path_result_to_alignment([], 99.0, n_mix_frames=100, recording_id="rec1")
    assert r.abstain


def test_weak_path_abstains():
    r = pdp.path_result_to_alignment(
        [(0.0, 1.0, 2.0)], 10.0, n_mix_frames=100, recording_id="rec1"
    )  # mean 0.1 < floor 0.3
    assert r.abstain


def test_run_wires_features_and_decode(monkeypatch):
    monkeypatch.setattr(
        pdp, "decode_path", lambda M, R, stretches, lam: ([(0.0, 5.0, 9.0)], 70.0)
    )
    feats = SimpleNamespace(shape=(12, 100))  # D x Tm
    probe = pdp.PathDecodeProbe(
        mix_features=lambda m: feats, ref_features=lambda r: feats
    )
    out = probe(
        MixContext(audio_path=Path("/tmp/m.wav")),
        RefContext(recording_id="recZ", audio_path=Path("/tmp/r.wav")),
        CandidatePool(),
    )
    assert isinstance(out, AlignmentResult)
    assert out.recording_id == "recZ" and out.offset_s == 5.0 and out.ref_end_s == 9.0
    assert out.confidence == 0.7 and out.source == "path"
