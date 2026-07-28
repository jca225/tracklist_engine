"""Branch-GC classification rules (scripts.gc_branches.classify).

`classify` is the whole safety argument for `make gc-branches-apply`: only the
MERGED bucket is ever deleted, and a branch reaches MERGED only when it has zero
commits that aren't already in `main`. We test the pure decision so the rules are
verified without a git fixture.
"""

from __future__ import annotations

from scripts.gc_branches import (
    ACTIVE,
    HELD,
    MERGED,
    PROTECTED_STATE,
    STALE,
    BranchState,
    classify,
)


def _b(name="feat/x", ahead=0, checked_out=False, days_idle=0):
    return BranchState(
        name=name, ahead=ahead, checked_out=checked_out, days_idle=days_idle
    )


def test_fully_merged_branch_is_deletable():
    assert classify(_b(ahead=0)) == MERGED


def test_merged_but_checked_out_is_held_not_deleted():
    # git would refuse the delete anyway; classifying it separately is what tells
    # the operator to remove the worktree first.
    assert classify(_b(ahead=0, checked_out=True)) == HELD


def test_unlanded_work_is_never_merged():
    # the load-bearing case: one unlanded commit must keep it out of the delete set
    assert classify(_b(ahead=1)) != MERGED
    assert classify(_b(ahead=1, days_idle=99)) != MERGED
    assert classify(_b(ahead=99, checked_out=True)) != MERGED


def test_main_is_protected_even_when_zero_ahead():
    assert classify(_b(name="main", ahead=0)) == PROTECTED_STATE
    assert classify(_b(name="master", ahead=0)) == PROTECTED_STATE


def test_idle_unlanded_branch_is_a_park_candidate():
    assert classify(_b(ahead=3, days_idle=3)) == STALE
    assert classify(_b(ahead=3, days_idle=30)) == STALE


def test_recent_unlanded_branch_is_left_alone():
    assert classify(_b(ahead=3, days_idle=0)) == ACTIVE
    assert classify(_b(ahead=3, days_idle=2)) == ACTIVE


def test_stale_threshold_is_configurable():
    assert classify(_b(ahead=1, days_idle=5), stale_days=7) == ACTIVE
    assert classify(_b(ahead=1, days_idle=7), stale_days=7) == STALE
