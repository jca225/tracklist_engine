"""Tests for the acquisition-case schema and the BB12 backfill.

The BB12 backfill is the schema's pressure test: if all 166 hand-resolved GT
outcomes can't round-trip cleanly through the case record, the schema is wrong.
"""

from __future__ import annotations

from pathlib import Path

from core.acquisition_case import (
    AcquisitionCase,
    Actor,
    Attempt,
    AttemptAction,
    AttemptVerdict,
    CaseClaim,
    CaseStatus,
    ProblemClass,
    append_attempt,
    case_from_dict,
    case_to_dict,
    load_cases,
    save_cases,
    with_problem_classes,
)

REPO = Path(__file__).resolve().parents[1]
GT = REPO / "labeling" / "fixtures" / "bb12_ground_truth.yaml"
AUDIT = REPO / "labeling" / "fixtures" / "bb12_path_audit.json"


def _sample_case() -> AcquisitionCase:
    return AcquisitionCase(
        set_id="testset",
        slot_label="013w1",
        layer_role="payload",
        claim=CaseClaim(
            recording_id="abc123",
            display_name="Artist - Title (Acappella)",
            stem="acappella",
        ),
        status=CaseStatus.RESOLVED,
        problem_classes=(ProblemClass.MISSING_ASSET,),
    )


# ── schema basics ─────────────────────────────────────────────────────────────


def test_case_id_is_derived():
    assert _sample_case().case_id == "testset:013w1:payload"


def test_round_trip_dict():
    case = _sample_case()
    again = case_from_dict(case_to_dict(case))
    assert again == case


def test_append_attempt_is_immutable():
    case = _sample_case()
    att = Attempt(action=AttemptAction.YT_SEARCH, verdict=AttemptVerdict.PROMOTE)
    grown = append_attempt(case, att)
    assert len(case.attempts) == 0  # original untouched
    assert len(grown.attempts) == 1
    assert grown.attempts[0].action is AttemptAction.YT_SEARCH


def test_with_problem_classes_dedupes():
    case = with_problem_classes(
        _sample_case(), ProblemClass.MISSING_ASSET, ProblemClass.WRONG_STEM
    )
    assert case.problem_classes == (ProblemClass.MISSING_ASSET, ProblemClass.WRONG_STEM)


def test_unknown_enum_falls_back():
    d = case_to_dict(_sample_case())
    d["status"] = "bogus_status"
    d["problem_classes"] = ["not_a_real_class"]
    case = case_from_dict(d)  # lenient: must not raise
    assert case.status is CaseStatus.OPEN


def test_jsonl_save_load_round_trip(tmp_path: Path):
    cases = [
        _sample_case(),
        append_attempt(_sample_case(), Attempt(AttemptAction.SEPARATE)),
    ]
    p = tmp_path / "testset.jsonl"
    save_cases(p, cases)
    assert load_cases(p) == cases


def test_load_missing_file_is_empty(tmp_path: Path):
    assert load_cases(tmp_path / "nope.jsonl") == []


# ── live-logging: record_attempt find-or-create ───────────────────────────────


def test_record_attempt_creates_then_appends(tmp_path: Path):
    from core.acquisition_case import Resolution, record_attempt

    kw = dict(set_id="bb11", slot_label="013w1", recording_id="rec1", root=tmp_path)
    c1 = record_attempt(attempt=Attempt(AttemptAction.YT_SEARCH), **kw)
    assert c1.case_id == "bb11:013w1:payload"  # layer_role derived from slot
    assert len(c1.attempts) == 1

    c2 = record_attempt(
        attempt=Attempt(AttemptAction.SEPARATE),
        resolution=Resolution(ref_source="online_candidate"),
        status=CaseStatus.RESOLVED,
        **kw,
    )
    # Same (slot, recording) → appended, not duplicated.
    assert len(load_cases(tmp_path / "bb11.jsonl")) == 1
    assert len(c2.attempts) == 2
    assert c2.status is CaseStatus.RESOLVED
    assert c2.resolution is not None and c2.resolution.ref_source == "online_candidate"


def test_record_attempt_distinct_recordings_are_separate_cases(tmp_path: Path):
    from core.acquisition_case import record_attempt

    record_attempt(
        set_id="bb11",
        slot_label="013",
        recording_id="recA",
        attempt=Attempt(AttemptAction.INGEST_URL),
        root=tmp_path,
    )
    record_attempt(
        set_id="bb11",
        slot_label="013",
        recording_id="recB",
        attempt=Attempt(AttemptAction.INGEST_URL),
        root=tmp_path,
    )
    assert len(load_cases(tmp_path / "bb11.jsonl")) == 2


# ── BB12 backfill: the real coverage check ────────────────────────────────────


def _bb12_cases() -> list[AcquisitionCase]:
    from scripts.backfill_bb12_cases import build_cases

    return build_cases(GT, AUDIT)


def test_bb12_backfill_covers_every_row():
    cases = _bb12_cases()
    # 166 GT rows collapse to 155 cases (11 re-entry / multi-source rows merged).
    assert len(cases) == 155
    # Every case has a recording identity and a claim.
    assert all(c.claim.recording_id for c in cases)
    # Every case has at least one attempt reconstructed.
    assert all(c.attempts for c in cases)


def test_bb12_status_partition():
    cases = _bb12_cases()
    by_status = {s: 0 for s in CaseStatus}
    for c in cases:
        by_status[c.status] += 1
    assert by_status[CaseStatus.UNRESOLVABLE] == 2  # the two mix-extract hosts
    assert by_status[CaseStatus.HUMAN_REVIEW] == 41  # unresolved_manifest beds
    assert by_status[CaseStatus.RESOLVED] == 112


def test_bb12_preference_pairs_are_online_over_demucs():
    cases = _bb12_cases()
    pref = [c for c in cases if c.training.preference_pairs]
    assert len(pref) == 4
    for c in pref:
        # winner is the online candidate; loser is the separated stem
        assert c.resolution is not None
        assert c.resolution.ref_source == "online_candidate"
        winner, loser = c.training.preference_pairs[0]
        assert "/candidates/" in winner
        assert "/candidates/" not in loser


def test_bb12_unalignable_are_phantom():
    cases = _bb12_cases()
    unres = [c for c in cases if c.status is CaseStatus.UNRESOLVABLE]
    assert len(unres) == 2
    for c in unres:
        assert ProblemClass.PHANTOM_TRACK in c.problem_classes
        assert c.notes  # carries the human source_note


def test_bb12_backfill_serializes_cleanly():
    # Every BB12 case must survive a dict round-trip (schema completeness).
    for c in _bb12_cases():
        assert case_from_dict(case_to_dict(c)) == c
