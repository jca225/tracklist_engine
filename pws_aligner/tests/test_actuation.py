from __future__ import annotations

from pws_aligner.actuation import (
    AbstentionEvent,
    ActuationKind,
    actuation_queue_dict,
    collect_actuations,
    route_abstention,
)
from pws_aligner.votes import AbstainReason


def _ev(reason, recording_id="rec1", lf="fp", detail=""):
    return AbstentionEvent(
        set_id="set1",
        slot_label="7",
        recording_id=recording_id,
        lf=lf,
        reason=reason,
        detail=detail,
    )


def test_no_data_routes_to_fetch_missing():
    req = route_abstention(_ev(AbstainReason.NO_DATA))
    assert req is not None
    assert req.kind == ActuationKind.FETCH_MISSING.value
    assert req.recording_id == "rec1"
    assert req.set_id == "set1"


def test_out_of_domain_routes_to_reacquire_wrong_version():
    req = route_abstention(_ev(AbstainReason.OUT_OF_DOMAIN, detail="46s preview clip"))
    assert req is not None
    assert req.kind == ActuationKind.REACQUIRE_WRONG_VERSION.value
    assert "preview" in req.reason


def test_low_margin_does_not_actuate():
    # genuine ambiguity is NOT an ingredient problem — never touch the corpus for it
    assert route_abstention(_ev(AbstainReason.LOW_MARGIN)) is None


def test_fired_vote_does_not_actuate():
    assert route_abstention(_ev(AbstainReason.NONE)) is None


def test_missing_recording_id_cannot_actuate():
    # NO_DATA but we don't know WHICH recording to fetch -> no actuation target
    assert route_abstention(_ev(AbstainReason.NO_DATA, recording_id=None)) is None


def test_budget_giveup():
    # already tried max_attempts times -> stop re-acquiring (no infinite redownload loop)
    ev = _ev(AbstainReason.NO_DATA)
    assert route_abstention(ev, attempt_count=2, max_attempts=2) is None
    assert route_abstention(ev, attempt_count=1, max_attempts=2) is not None


def test_collect_dedups_by_recording_id():
    # the same missing recording abstained on across many spans -> ONE re-acquire
    events = [
        _ev(AbstainReason.NO_DATA, recording_id="recX"),
        _ev(AbstainReason.NO_DATA, recording_id="recX"),
        _ev(AbstainReason.NO_DATA, recording_id="recY"),
    ]
    reqs = collect_actuations(events)
    rids = sorted(r.recording_id for r in reqs)
    assert rids == ["recX", "recY"]


def test_collect_respects_prior_attempts():
    events = [_ev(AbstainReason.NO_DATA, recording_id="recX")]
    # recX already attempted twice -> budget exhausted -> dropped
    reqs = collect_actuations(events, attempts={"recX": 2}, max_attempts=2)
    assert reqs == []


def test_collect_later_no_data_survives_earlier_low_margin_same_rid():
    # first event for recZ is LOW_MARGIN (no request); a later NO_DATA for recZ must still produce one
    events = [
        _ev(AbstainReason.LOW_MARGIN, recording_id="recZ"),
        _ev(AbstainReason.NO_DATA, recording_id="recZ"),
    ]
    reqs = collect_actuations(events)
    assert [r.recording_id for r in reqs] == ["recZ"]


def test_queue_dict_is_serializable():
    reqs = collect_actuations([_ev(AbstainReason.NO_DATA, recording_id="recX")])
    rows = actuation_queue_dict(reqs)
    assert rows[0]["recording_id"] == "recX"
    assert rows[0]["kind"] == ActuationKind.FETCH_MISSING.value
    assert set(rows[0]) >= {
        "recording_id",
        "set_id",
        "slot_label",
        "kind",
        "reason",
        "lf",
        "attempt",
    }
