"""Tests for the shared timeline-provenance module.

These pin the exact semantics of cohort_spread_s (which the bb_baselines
regression tests also cover — imported through bb_baselines so the redirect
chain is exercised).  The tests here import directly from the NEW module.
"""

from __future__ import annotations

from workspaces.alignment_prototype.timeline_provenance import (
    _git_sha,
    cohort_spread_s,
    driver_provenance,
)

# ------------------------------------------------------------------ cohort_spread_s


def test_cohort_spread_flags_incoherent_6h():
    """A cohort whose mtimes span > 6 h is flagged — the scorer must warn."""
    prov = {
        "classical": {"exists": True, "mtime_epoch": 1_000_000.0},
        "agentic": {"exists": True, "mtime_epoch": 1_000_000.0 + 7 * 3600},
    }
    spread = cohort_spread_s(prov)
    assert spread is not None
    assert spread > 21600  # 6 h threshold


def test_cohort_spread_mirrors_bb11_bug():
    """Mirrors the exact 2026-07-17 bug: classical 07-11, agentic 07-14 -> ~3-day spread."""
    prov = {
        "classical": {"exists": True, "mtime_epoch": 1_000_000.0},
        "agentic": {"exists": True, "mtime_epoch": 1_000_000.0 + 3 * 86400},
    }
    spread = cohort_spread_s(prov)
    assert spread == 3 * 86400
    assert spread > 6 * 3600  # would trip the default gate


def test_cohort_spread_none_when_single_or_absent():
    """Mirrors the none-when-<2 case from test_bb_baselines."""
    assert cohort_spread_s({"classical": {"exists": True, "mtime_epoch": 1.0}}) is None
    assert cohort_spread_s({"agentic": {"exists": False}}) is None


def test_cohort_spread_small_coherent_cohort():
    """A within-run cohort (60 s spread) returns the exact spread, well under gate."""
    prov = {
        "classical": {"exists": True, "mtime_epoch": 500.0},
        "agentic": {"exists": True, "mtime_epoch": 560.0},
    }
    assert cohort_spread_s(prov) == 60.0


# ------------------------------------------------------------------ _git_sha


def test_git_sha_returns_string():
    """_git_sha() always returns a non-empty string (may be 'unknown' in CI)."""
    sha = _git_sha()
    assert isinstance(sha, str)
    assert len(sha) > 0
