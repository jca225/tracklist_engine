"""Injective per-recording assignment of predicted spans to GT forms.

RT1 form-centric re-measure: each span is assigned to at most one GT form and
each form to at most one span (greedy nearest, no reuse), so a reprise scores
each span against its OWN form and a recording's seconds are never double-counted.
Forms no span is assigned to are the honest recall loss.
"""

from __future__ import annotations

from workspaces.alignment_prototype.never_matched import assign_spans_to_forms


def _row(tid, ss):
    return {
        "track_id": tid,
        "set_start_s": ss,
        "set_end_s": ss + 30,
        "claimed_stem": "regular",
    }


def _span(rid, ss):
    return {"recording_id": rid, "set_start_s": ss}


def test_reprise_each_span_gets_its_own_form():
    forms = [_row("r", 100.0), _row("r", 500.0)]
    spans = [_span("r", 105.0), _span("r", 490.0)]
    assigned, unmatched = assign_spans_to_forms(forms, spans)
    assert assigned[id(spans[0])] is forms[0]
    assert assigned[id(spans[1])] is forms[1]
    assert unmatched == []


def test_two_forms_one_span_leaves_farther_form_unmatched():
    forms = [_row("r", 100.0), _row("r", 500.0)]
    spans = [_span("r", 105.0)]
    assigned, unmatched = assign_spans_to_forms(forms, spans)
    assert assigned[id(spans[0])] is forms[0]
    assert unmatched == [forms[1]]


def test_two_spans_one_form_one_span_unassigned_no_unmatched_forms():
    forms = [_row("r", 100.0)]
    spans = [_span("r", 105.0), _span("r", 108.0)]
    assigned, unmatched = assign_spans_to_forms(forms, spans)
    # nearest span wins the single form; the other span is left unassigned
    assert assigned[id(spans[0])] is forms[0]
    assert id(spans[1]) not in assigned
    assert unmatched == []


def test_greedy_is_globally_sensible_not_first_come():
    # s2 is a very tight match to A; s1 should then take B
    forms = [_row("r", 100.0), _row("r", 110.0)]  # A, B
    spans = [_span("r", 108.0), _span("r", 101.0)]  # s1, s2
    assigned, unmatched = assign_spans_to_forms(forms, spans)
    assert assigned[id(spans[1])] is forms[0]  # s2 -> A (dist 1)
    assert assigned[id(spans[0])] is forms[1]  # s1 -> B (dist 2)
    assert unmatched == []


def test_trackless_forms_ignored_and_unmatched_recording_all_forms_returned():
    forms = [
        _row("gone", 300.0),
        {"track_id": None, "set_start_s": 9.0, "set_end_s": 9.0},
    ]
    spans = [_span("other", 10.0)]
    assigned, unmatched = assign_spans_to_forms(forms, spans)
    assert assigned == {}
    assert unmatched == [forms[0]]  # trackless row skipped entirely
