"""DeterministicDriver: probes x candidates -> merge -> best candidate, with
abstention and skip-on-unresolved. Pure orchestration (stub probes, no audio)."""

from __future__ import annotations

from pathlib import Path

from workspaces.alignment_prototype.harness import (
    AlignmentResult,
    CandidatePool,
    DeterministicDriver,
    MixContext,
    Probe,
    RefContext,
)
from workspaces.alignment_prototype.records import SlotCandidate


class _FixedProbe(Probe):
    """Returns a preset (offset, confidence) per recording_id; abstains otherwise."""

    def __init__(self, name, table):
        self.name = name
        self._table = table  # recording_id -> (offset, confidence)

    def run(self, mix, ref, candidates):
        if ref.recording_id not in self._table:
            return AlignmentResult.abstained(
                source=self.name, recording_id=ref.recording_id
            )
        off, conf = self._table[ref.recording_id]
        return AlignmentResult(
            recording_id=ref.recording_id,
            offset_s=off,
            confidence=conf,
            source=self.name,
        )


def _resolver(c):
    return RefContext(
        recording_id=c.recording_id, audio_path=Path(f"/tmp/{c.recording_id}.wav")
    )


_MIX = MixContext(audio_path=Path("/tmp/mix.wav"))


def _pool(*recs):
    return CandidatePool(
        tuple(SlotCandidate(recording_id=r, claimed_stem="regular") for r in recs)
    )


def test_single_candidate_merges_probes():
    p1 = _FixedProbe("fp", {"recA": (10.0, 0.6)})
    p2 = _FixedProbe("chroma", {"recA": (10.5, 0.5)})  # agrees within tol -> boost
    out = DeterministicDriver([p1, p2], resolve_ref=_resolver).align(
        _MIX, _pool("recA")
    )
    assert out.recording_id == "recA" and out.offset_s == 10.0
    assert out.confidence == 0.7  # 0.6 + 0.1 corroboration


def test_picks_most_confident_candidate():
    p = _FixedProbe("fp", {"recA": (10.0, 0.5), "recB": (20.0, 0.9)})
    out = DeterministicDriver([p], resolve_ref=_resolver).align(
        _MIX, _pool("recA", "recB")
    )
    assert out.recording_id == "recB" and out.confidence == 0.9


def test_unresolvable_candidate_is_skipped():
    p = _FixedProbe("fp", {"recA": (10.0, 0.8), "recB": (20.0, 0.95)})
    # recB resolves to None -> skipped, recA wins despite lower confidence.
    resolve = lambda c: None if c.recording_id == "recB" else _resolver(c)
    out = DeterministicDriver([p], resolve_ref=resolve).align(
        _MIX, _pool("recA", "recB")
    )
    assert out.recording_id == "recA"


def test_all_abstain_driver_abstains():
    p = _FixedProbe("fp", {})  # abstains on every recording
    out = DeterministicDriver([p], resolve_ref=_resolver).align(
        _MIX, _pool("recA", "recB")
    )
    assert out.abstain and out.source == "driver"


def test_min_confidence_gate_abstains():
    p = _FixedProbe("fp", {"recA": (10.0, 0.2)})
    out = DeterministicDriver([p], resolve_ref=_resolver, min_confidence=0.5).align(
        _MIX, _pool("recA")
    )
    assert out.abstain
