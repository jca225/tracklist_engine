"""WIP-limit fence threshold logic (scripts.guardrails.over_wip_limit).

The fence blocks a branch that has run more than WIP_COMMIT_LIMIT commits ahead
of origin/main, with two escape hatches (an `epic` branch or the
GUARDRAILS_ALLOW_BIG env override). We test the *pure* decision function so the
threshold logic is verified without shelling out to git.
"""

from __future__ import annotations

from scripts.guardrails import WIP_COMMIT_LIMIT, over_wip_limit


def test_under_limit_is_allowed():
    assert over_wip_limit(0, "feature/x", allow_big=False) is False
    assert over_wip_limit(WIP_COMMIT_LIMIT, "feature/x", allow_big=False) is False


def test_at_limit_boundary_is_allowed():
    # exactly the limit is fine; strictly greater trips it
    assert over_wip_limit(WIP_COMMIT_LIMIT, "feature/x", allow_big=False) is False
    assert over_wip_limit(WIP_COMMIT_LIMIT + 1, "feature/x", allow_big=False) is True


def test_over_limit_trips_the_fence():
    assert over_wip_limit(WIP_COMMIT_LIMIT + 1, "feature/x", allow_big=False) is True
    assert over_wip_limit(999, "feature/x", allow_big=False) is True


def test_allow_big_escape_hatch():
    assert over_wip_limit(999, "feature/x", allow_big=True) is False


def test_epic_branch_escape_hatch():
    assert over_wip_limit(999, "epic/big-refactor", allow_big=False) is False
    # substring match, case-insensitive
    assert over_wip_limit(999, "feature/EPIC-migration", allow_big=False) is False
    assert over_wip_limit(999, "my-Epic-thing", allow_big=False) is False


def test_non_epic_name_is_not_falsely_exempted():
    # a plain branch that merely contains none of the escape tokens is fenced
    assert (
        over_wip_limit(WIP_COMMIT_LIMIT + 5, "chore/cleanup", allow_big=False) is True
    )
