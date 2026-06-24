"""The alignment harness contract: confidence is bounded, abstention is explicit,
and a Probe is a uniform callable. These invariants are what let heterogeneous
probes/drivers compose against one surface."""

from __future__ import annotations

from pathlib import Path

import pytest

from workspaces.alignment_prototype.harness import (
    AlignmentResult,
    CandidatePool,
    MixContext,
    Probe,
    RefContext,
    RefSegment,
)
from workspaces.alignment_prototype.records import SlotCandidate


def test_valid_result_constructs():
    r = AlignmentResult(
        recording_id="rec1", offset_s=12.5, confidence=0.8, source="chroma"
    )
    assert r.recording_id == "rec1" and r.offset_s == 12.5
    assert r.segments == () and r.abstain is False


@pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0, -5.0])
def test_confidence_must_be_in_unit_interval(bad):
    with pytest.raises(ValueError):
        AlignmentResult(recording_id="r", offset_s=0.0, confidence=bad, source="x")


@pytest.mark.parametrize("ok", [0.0, 0.5, 1.0])
def test_confidence_bounds_inclusive(ok):
    AlignmentResult(recording_id="r", offset_s=0.0, confidence=ok, source="x")


def test_abstained_constructor():
    r = AlignmentResult.abstained(source="fp", recording_id="rec2")
    assert r.abstain is True and r.confidence == 0.0 and r.source == "fp"


def test_refsegment_and_segments_carry_piecewise_map():
    segs = (RefSegment(0.0, 4.0, 8.0), RefSegment(8.0, 4.0, 8.0))  # a loop
    r = AlignmentResult(recording_id="r", offset_s=4.0, segments=segs, source="path")
    assert len(r.segments) == 2 and r.segments[0].ref_end_s == 8.0


def test_probe_is_abstract():
    with pytest.raises(TypeError):
        Probe()  # type: ignore[abstract]


def test_concrete_probe_call_delegates_to_run():
    class _Stub(Probe):
        name = "stub"

        def run(self, mix, ref, candidates):
            return AlignmentResult(
                recording_id=ref.recording_id,
                offset_s=1.0,
                confidence=0.9,
                source=self.name,
            )

    probe = _Stub()
    mix = MixContext(audio_path=Path("/tmp/mix.wav"), set_id="s")
    ref = RefContext(recording_id="rec1", audio_path=Path("/tmp/ref.wav"))
    pool = CandidatePool((SlotCandidate(recording_id="rec1", claimed_stem="regular"),))
    out = probe(mix, ref, pool)  # __call__ -> run
    assert out.recording_id == "rec1" and out.source == "stub" and out.confidence == 0.9
    assert len(pool) == 1 and list(pool)[0].recording_id == "rec1"
