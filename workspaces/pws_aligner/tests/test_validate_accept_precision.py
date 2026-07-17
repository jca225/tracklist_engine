"""TDD for the BB-GT → GtCandidateCase builder (ACCEPT-precision validation).

The seam's banding (`band_agreement`) and the real-probe scorer are already
tested elsewhere. The NEW logic under test here is `build_gt_cases`: turning a
hand `GroundTruthSet` into positive (GT-correct ref) + decoy (wrong ref) cases
so `evaluate_banding` can measure whether 2-channel agreement ever ACCEPTs a
wrong ref (the co-training poisoning risk).
"""

from __future__ import annotations

from pathlib import Path

from labeling.ground_truth.schema import GroundTruthSet, GroundTruthTrack
from workspaces.pws_aligner.validate_accept_precision import (
    build_gt_cases,
    gt_relative_offset,
)


def _track(
    track_id: str | None,
    stem: str,
    slot: str,
    s0: float,
    s1: float,
    ref0: float,
    *,
    label: str = "x",
    unalignable: bool = False,
) -> GroundTruthTrack:
    return GroundTruthTrack(
        label=label,
        track_id=track_id,
        claimed_stem=stem,
        set_start_s=s0,
        set_end_s=s1,
        ref_start_s=ref0,
        slot_label=slot,
        unalignable=unalignable,
    )


def _resolver(paths: dict[tuple[str, str], Path]):
    def r(rid: str, stem: str) -> Path | None:
        return paths.get((rid, stem))

    return r


def test_build_positive_case_per_alignable_track():
    gt = GroundTruthSet(
        "s1",
        (
            _track("rA", "acappella", "010", 100.0, 140.0, 12.0),
            _track("rB", "instrumental", "020", 200.0, 250.0, 5.0),
        ),
    )
    paths = {
        ("rA", "acappella"): Path("/x/rA_v.flac"),
        ("rB", "instrumental"): Path("/x/rB_i.flac"),
    }
    cases = build_gt_cases(gt, _resolver(paths), n_decoys=0)
    assert len(cases) == 2
    assert all(c.candidate_is_correct for c in cases)
    by_rid = {c.candidate.recording_id: c for c in cases}
    a = by_rid["rA"]
    assert a.span.set_id == "s1"
    assert a.span.slot_label == "010"
    assert a.span.set_start_s == 100.0
    assert a.span.span_dur_s == 40.0
    assert a.candidate.stem == "acappella"
    assert a.candidate.source_path == "/x/rA_v.flac"
    assert a.claim_axes is not None and a.claim_axes["stem"] == "acappella"


def test_skips_unalignable_missing_id_and_unresolvable():
    gt = GroundTruthSet(
        "s1",
        (
            _track("rA", "acappella", "010", 100, 140, 12),  # ok
            _track("rU", "acappella", "011", 100, 140, 12, unalignable=True),
            _track(None, "acappella", "012", 100, 140, 12),  # no track_id
            _track("rN", "acappella", "013", 100, 140, 12),  # resolver returns None
        ),
    )
    paths = {("rA", "acappella"): Path("/x/rA.flac")}
    cases = build_gt_cases(gt, _resolver(paths), n_decoys=0)
    assert [c.candidate.recording_id for c in cases] == ["rA"]


def test_decoys_are_wrong_distinct_and_share_the_span():
    gt = GroundTruthSet(
        "s1",
        (
            _track("rA", "acappella", "010", 100, 140, 12),
            _track("rB", "acappella", "020", 200, 250, 5),
        ),
    )
    paths = {
        ("rA", "acappella"): Path("/x/a.flac"),
        ("rB", "acappella"): Path("/x/b.flac"),
    }
    cases = build_gt_cases(gt, _resolver(paths), n_decoys=1, seed=1)
    pos = [c for c in cases if c.candidate_is_correct]
    neg = [c for c in cases if not c.candidate_is_correct]
    assert len(pos) == 2 and len(neg) == 2
    for c in neg:
        same_slot_pos = next(p for p in pos if p.span.slot_label == c.span.slot_label)
        # decoy is a DIFFERENT recording than the GT-correct one at this span
        assert c.candidate.recording_id != same_slot_pos.candidate.recording_id
        # but it targets the exact same mix span
        assert c.span.set_start_s == same_slot_pos.span.set_start_s
        assert c.span.span_dur_s == same_slot_pos.span.span_dur_s
        # and it carries real audio routed as the span's stem
        assert c.candidate.source_path is not None
        assert c.candidate.stem == same_slot_pos.candidate.stem


def test_decoy_prefers_same_stem_pool():
    gt = GroundTruthSet(
        "s1",
        (
            _track("rA", "acappella", "010", 100, 140, 12),
            _track(
                "rV", "acappella", "020", 200, 250, 5
            ),  # same-stem decoy (preferred)
            _track("rI", "instrumental", "030", 300, 340, 5),
        ),
    )
    paths = {
        ("rA", "acappella"): Path("/a"),
        ("rV", "acappella"): Path("/v"),
        ("rI", "instrumental"): Path("/i"),
        ("rV", "instrumental"): Path("/vi"),
        ("rI", "acappella"): Path("/ia"),
    }
    cases = build_gt_cases(gt, _resolver(paths), n_decoys=1, seed=3)
    a_decoys = [
        c for c in cases if not c.candidate_is_correct and c.span.slot_label == "010"
    ]
    assert len(a_decoys) == 1
    # rV (another acappella) is a fairer hard-negative than rI (instrumental)
    assert a_decoys[0].candidate.recording_id == "rV"
    assert a_decoys[0].candidate.stem == "acappella"
    assert a_decoys[0].candidate.source_path == "/v"


def test_deterministic_given_seed():
    gt = GroundTruthSet(
        "s1",
        tuple(
            _track(f"r{i}", "acappella", f"{i:03d}", i * 10.0, i * 10 + 5.0, 3.0)
            for i in range(6)
        ),
    )
    paths = {(f"r{i}", "acappella"): Path(f"/{i}") for i in range(6)}
    c1 = build_gt_cases(gt, _resolver(paths), n_decoys=2, seed=7)
    c2 = build_gt_cases(gt, _resolver(paths), n_decoys=2, seed=7)

    def key(cs):
        return [
            (c.span.slot_label, c.candidate.recording_id, c.candidate_is_correct)
            for c in cs
        ]

    assert key(c1) == key(c2)


def test_stem_filter_restricts_positives():
    gt = GroundTruthSet(
        "s1",
        (
            _track("rA", "acappella", "010", 100, 140, 12),
            _track("rI", "instrumental", "020", 200, 250, 5),
        ),
    )
    paths = {
        ("rA", "acappella"): Path("/a"),
        ("rI", "instrumental"): Path("/i"),
    }
    cases = build_gt_cases(gt, _resolver(paths), n_decoys=0, stem_filter="acappella")
    assert [c.candidate.recording_id for c in cases] == ["rA"]


def test_gt_relative_offset_is_ref_minus_set_start():
    t = _track("rA", "acappella", "010", 100.0, 140.0, 12.0)
    assert gt_relative_offset(t) == 12.0 - 100.0
