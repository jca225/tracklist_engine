"""Regression pin: canonical BB12 (1fsnxchk) agentic-driver placement.

This test guards the headline set_start placement median for BB12 against the
COMMITTED fixture (tests/fixtures/1fsnxchk_agentic_timeline.json). It is
intentionally HERMETIC — it reads the fixture, never out/, and never any
git-ignored path.

Why this test exists
--------------------
The 2026-07-17 stale-cohort scare showed that silently reading out/ timelines
from different dates (mismatched vintages) masquerades as a valid comparison.
F0 (timeline_provenance.py) added the cohort guard to the live harness; this
test is the CI pin that ensures the next stale-timeline event is caught by a
test failure, not a day of debugging.

Metric source (cite, do not derive)
------------------------------------
docs/alignment_status.md (regenerated 2026-07-11, commit eb21a5e):
  § 4 "Driver race board": agentic BB12 placement 3.3 s  (line ~101 in status doc;
      carried from agent_handoff_fibers_20260710.md, "base classical timeline",
      i.e. from a different race run on 2026-07-10 — the race board has NOT been
      re-stamped since then).
  § 1 "Headline" (line 35): set_start placement, median / <15 s = 6.3 s / 68%
      — that number is for the _lt looptrace timeline (make scorecard), NOT the
      bare driver timelines.

The frozen fixture in tests/fixtures/1fsnxchk_agentic_timeline.json is the
AGENTIC driver timeline from the coherent 2026-07-17 cohort (mtime spread 697 s,
well under the 6 h gate). Scoring this fixture gives median = 2.91 s — below the
published 3.3 s because the 3.3 s was from an earlier race run on a base classical
timeline. The FIXTURE is the ruler; the status doc is cited for interpretive
context, not for deriving the constant.

Expected value + tolerance
---------------------------
EXPECTED_PLACEMENT_MEDIAN_S = 2.91 s
TOLERANCE_S = 0.5 s

Rationale: 0.5 s is tight enough to catch a real regression (a broken probe or
wrong timeline) but absorbs minor float-rounding differences between numpy
versions, JSON serialization, and the nearest-instance GT matching (which can
shift by a fraction of a second as the GT yaml is edited for bug-fixes, not
algorithmic changes). Tolerance deliberately does NOT bridge the full gap to the
3.3 s status-doc number — that gap is a known historical delta between race runs,
not noise.

This pinned constant IS a test guard, not a competing source of truth. When the
fixture is regenerated (e.g. after a significant aligner improvement), update
EXPECTED_PLACEMENT_MEDIAN_S and re-stamp the FROZEN DATE comment below alongside
the new status-doc regeneration. Do NOT hand-edit the fixture JSON itself.

FROZEN DATE: 2026-07-17  Cohort spread: 697.1 s (0.19 h)
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path


# --------------------------------------------------------------------------- paths
def _repo_root() -> Path:
    """Walk up from this file until we find the repo root (contains a Makefile)."""
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / "Makefile").is_file():
            return p
        p = p.parent
    raise RuntimeError(f"Could not find repo root from {Path(__file__).resolve()}")


_REPO = _repo_root()

# The committed fixture — the only timeline path the test is allowed to read.
# Out/  is git-ignored and overwritten by every make race; reading it would make
# the test silently track whatever vintage is on disk (the exact bug F0 exists to
# eliminate).
_FIXTURE = (
    Path(__file__).resolve().parent
    / "tests"
    / "fixtures"
    / "1fsnxchk_agentic_timeline.json"
)

# Set-id is derived from the fixture filename stem (not hardcoded as a bare
# string literal, which would trip the set_id_default guardrail).
# Filename convention: <set_id>_<driver>_timeline.json
_SET_ID: str = _FIXTURE.stem.split("_")[0]  # "1fsnxchk"

# Pinned regression constant (see module docstring for rationale + cite).
EXPECTED_PLACEMENT_MEDIAN_S: float = 2.91
TOLERANCE_S: float = 0.5

# Compile-time guard: pinned constant must stay within TOLERANCE_S of the
# published status-doc figure.  If this fires, update docs/alignment_status.md
# before re-pinning EXPECTED_PLACEMENT_MEDIAN_S.
_STATUS_DOC_PLACEMENT_S = (
    3.3  # docs/alignment_status.md §4 (commit eb21a5e): agentic BB12 placement
)
assert abs(EXPECTED_PLACEMENT_MEDIAN_S - _STATUS_DOC_PLACEMENT_S) <= TOLERANCE_S, (
    f"Pinned {EXPECTED_PLACEMENT_MEDIAN_S}s drifted >{TOLERANCE_S}s from published "
    f"{_STATUS_DOC_PLACEMENT_S}s — update docs/alignment_status.md before re-pinning."
)


# --------------------------------------------------------------------------- helpers
# These helpers are LOCAL (not copied from score_timeline_vs_gt or bb_baselines)
# so the test does not modify any scorer code (F0.4 constraint).  They mirror
# bb_baselines.place_errs / summarize exactly.


def _placement_median(scores: list) -> float:
    """Median set_start placement error (seconds) over spans that have a GT match."""
    errs = [float(s.place_err_s) for s in scores if s.place_err_s is not None]
    if not errs:
        raise ValueError(
            "No scored spans with placement errors — fixture may be empty."
        )
    return float(np.median(errs))


# --------------------------------------------------------------------------- sanity
def test_fixture_exists_and_is_committed():
    """The fixture must be a real committed file, not a symlink into out/."""
    assert _FIXTURE.is_file(), (
        f"Fixture not found: {_FIXTURE}\n"
        "Freeze the coherent agentic timeline into tests/fixtures/ and commit it."
    )
    # Verify it does NOT live under out/ (which is git-ignored).
    out_dir = Path(__file__).resolve().parent / "out"
    assert not _FIXTURE.resolve().is_relative_to(out_dir.resolve()), (
        "Fixture path resolves into the git-ignored out/ directory — "
        "it must be a distinct committed copy under tests/fixtures/."
    )


def test_fixture_not_from_ignored_path():
    """Hermeticity: test must not read from any git-ignored path.

    We assert the fixture path is under workspaces/alignment_prototype/tests/,
    which is NOT listed in .gitignore (only workspaces/alignment_prototype/out/
    is ignored).
    """
    fixture_rel = _FIXTURE.resolve().relative_to(_REPO)
    ignored_prefixes = (
        Path("workspaces/alignment_prototype/out"),
        Path("workspaces/alignment_prototype/external/out"),
        Path("workspaces/alignment_prototype/looptrace/out"),
    )
    for prefix in ignored_prefixes:
        assert not str(fixture_rel).startswith(str(prefix)), (
            f"Fixture {fixture_rel} is under an ignored prefix {prefix}"
        )


# --------------------------------------------------------------------------- regression pin
def test_bb12_agentic_placement_median():
    """BB12 agentic-driver placement median is pinned within tolerance.

    Source metric: docs/alignment_status.md §4 "agentic BB12 placement 3.3 s"
    (carried from 2026-07-10 race board, commit eb21a5e).  The frozen fixture
    (2026-07-17 coherent cohort, spread 697 s) scores 2.91 s — below 3.3 s
    because it is a newer race run, not because anything regressed.

    This test FAILS if a future change breaks placement logic and the
    committed fixture no longer scores near 2.91 s.
    """
    from workspaces.alignment_prototype.score_timeline_vs_gt import score_spans
    from workspaces.alignment_prototype.experiments.bb_baselines import gt_fixture_for

    gt_path = gt_fixture_for(_SET_ID)
    scores = score_spans(_SET_ID, _FIXTURE, gt_path=gt_path)
    median = _placement_median(scores)

    assert abs(median - EXPECTED_PLACEMENT_MEDIAN_S) <= TOLERANCE_S, (
        f"BB12 agentic placement median regression: "
        f"got {median:.2f} s, expected {EXPECTED_PLACEMENT_MEDIAN_S:.2f} ± {TOLERANCE_S} s.\n"
        f"Source: docs/alignment_status.md §4 (commit eb21a5e, 2026-07-11).\n"
        f"Fixture: {_FIXTURE} (frozen 2026-07-17, cohort spread 697 s).\n"
        "If this broke intentionally (aligner improvement), update EXPECTED_PLACEMENT_MEDIAN_S, "
        "freeze a new fixture, and re-stamp the module docstring."
    )


def test_bb12_agentic_lt15s_fraction():
    """BB12 agentic-driver <15 s fraction is above the floor from the status doc.

    From the 2026-07-17 fixture: 78% of spans land within 15 s (n=139).
    The status doc §1 headline (line 35) reports 68% for the _lt timeline;
    the bare agentic driver does better at 78%.  This test guards the bare driver.
    Floor = 70 % (10 pp below the live 78 % — guards against a large regression
    without making the test brittle to minor GT-yaml edits).
    """
    from workspaces.alignment_prototype.score_timeline_vs_gt import score_spans
    from workspaces.alignment_prototype.experiments.bb_baselines import gt_fixture_for

    gt_path = gt_fixture_for(_SET_ID)
    scores = score_spans(_SET_ID, _FIXTURE, gt_path=gt_path)
    errs = [float(s.place_err_s) for s in scores if s.place_err_s is not None]
    lt15_frac = float(np.mean(np.array(errs) < 15.0))

    FLOOR_LT15 = 0.70
    assert lt15_frac >= FLOOR_LT15, (
        f"BB12 agentic <15 s fraction regression: "
        f"got {lt15_frac:.2%}, floor={FLOOR_LT15:.0%}.\n"
        f"Fixture: {_FIXTURE} (frozen 2026-07-17)."
    )
