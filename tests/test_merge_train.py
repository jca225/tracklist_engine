"""Unit tests for the merge-train ordering logic (pure, no git/braid)."""

from __future__ import annotations

from scripts.merge_train import Candidate, format_plan, plan_order


def _c(branch, ready=True, ahead=1, conflict_files=0, behind=0, note=""):
    return Candidate(branch, ready, ahead, conflict_files, behind, note)


def test_ready_before_blocked():
    blocked = _c("blocked", ready=False, conflict_files=1)
    ready = _c("ready", ready=True, conflict_files=9)
    assert [c.branch for c in plan_order([blocked, ready])] == ["ready", "blocked"]


def test_fewest_conflicts_first_among_ready():
    a = _c("a", conflict_files=5)
    b = _c("b", conflict_files=0)
    c = _c("c", conflict_files=2)
    assert [x.branch for x in plan_order([a, b, c])] == ["b", "c", "a"]


def test_ahead_breaks_conflict_ties():
    big = _c("big", conflict_files=0, ahead=62)
    small = _c("small", conflict_files=0, ahead=2)
    assert [x.branch for x in plan_order([big, small])] == ["small", "big"]


def test_first_forces_to_top_even_over_readier():
    forced = _c("dep", ready=True, conflict_files=99, ahead=50)
    other = _c("other", ready=True, conflict_files=0, ahead=1)
    assert [x.branch for x in plan_order([other, forced], first="dep")][0] == "dep"


def test_name_is_stable_final_tiebreak():
    x = _c("x", conflict_files=0, ahead=1)
    a = _c("a", conflict_files=0, ahead=1)
    assert [c.branch for c in plan_order([x, a])] == ["a", "x"]


def test_plan_marks_first_ready_as_next():
    out = format_plan(plan_order([_c("only", ready=True, conflict_files=0)]))
    assert "-> only" in out
    assert "next to land: only" in out
    assert "gh pr merge --squash --auto" in out


def test_plan_reports_nothing_landable():
    out = format_plan(plan_order([_c("stuck", ready=False, conflict_files=3)]))
    assert "NOTHING landable" in out


def test_plan_handles_empty_fleet():
    assert "no candidate branches" in format_plan([])


def test_plan_header_reflects_gated_mode():
    ordered = plan_order([_c("only", ready=True)])
    assert "plan (read-only)" in format_plan(ordered, read_only=True)
    assert "land-next (gated)" in format_plan(ordered, read_only=False)
