"""Honest identity on an incomplete GT.

The de-poisoned GT abstains on 34-43% of appearances (``track_id=None``): sound
but incomplete. Prediction-centric identity (per span, does it match a GT
track_id) counts a correct prediction on an abstained appearance as a MISS ->
deflation. The honest axis is appearance-recall over the ADJUDICABLE (bound)
appearances only, with the abstained fraction reported as unmeasurable, never
scored.
"""

from __future__ import annotations

from workspaces.alignment_prototype.never_matched import identity_recall


def _gt(tid, ss, se, **kw):
    d = {"track_id": tid, "set_start_s": ss, "set_end_s": se}
    d.update(kw)
    return d


def _span(rid, ss, se):
    return {"recording_id": rid, "set_start_s": ss, "set_end_s": se}


def test_bound_appearance_found_by_matching_span_is_a_hit():
    gt = [_gt("r1", 100.0, 140.0)]
    spans = [_span("r1", 101.0, 139.0)]
    r = identity_recall(gt, spans)
    assert r["bound"] == 1 and r["hits"] == 1 and r["abstain"] == 0
    assert r["recall"] == 1.0


def test_bound_appearance_with_no_matching_span_is_a_miss():
    gt = [_gt("r1", 100.0, 140.0)]
    spans = [_span("OTHER", 101.0, 139.0)]
    r = identity_recall(gt, spans)
    assert r["bound"] == 1 and r["hits"] == 0 and r["recall"] == 0.0


def test_abstained_appearance_is_unadjudicable_not_a_miss():
    gt = [_gt("r1", 100.0, 140.0), _gt(None, 200.0, 240.0, claimed_stem="acappella")]
    spans = [_span("r1", 101.0, 139.0)]  # nothing at the abstained appearance
    r = identity_recall(gt, spans)
    # abstained row is excluded from the denominator entirely
    assert r["bound"] == 1 and r["abstain"] == 1
    assert r["hits"] == 1 and r["recall"] == 1.0  # 1/1, not 1/2


def test_matching_id_at_wrong_time_does_not_credit_the_appearance():
    gt = [_gt("r1", 100.0, 140.0)]
    spans = [_span("r1", 900.0, 940.0)]  # right id, far-away time
    r = identity_recall(gt, spans)
    assert r["hits"] == 0 and r["recall"] == 0.0


def test_unalignable_bound_rows_excluded():
    gt = [_gt("r1", 100.0, 140.0), _gt("r2", 200.0, 240.0, unalignable=True)]
    spans = [_span("r1", 101.0, 139.0)]
    r = identity_recall(gt, spans)
    assert r["bound"] == 1  # r2 excluded as unalignable
    assert r["recall"] == 1.0


def test_coverage_fields_report_the_adjudicable_fraction():
    gt = [_gt("a", 10, 20), _gt("b", 30, 40), _gt(None, 50, 60)]
    spans = [_span("a", 11, 19)]
    r = identity_recall(gt, spans)
    assert r["bound"] == 2 and r["abstain"] == 1
    assert abs(r["adjudicable_frac"] - 2 / 3) < 1e-9
