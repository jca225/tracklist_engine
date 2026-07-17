"""Unit tests for the BB baseline placement harness (pure logic only — no audio)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from core.contracts import load_timeline
from workspaces.alignment_prototype.experiments.bb_baselines import (
    PILEUP_DENSITY,
    _stats,
    baseline_timeline,
    cohort_spread_s,
    summarize,
)


def test_cohort_spread_flags_mismatched_vintages():
    # the exact 2026-07-17 bug: classical 07-11, agentic 07-14 -> ~3-day spread
    prov = {
        "classical": {"exists": True, "mtime_epoch": 1_000_000.0},
        "agentic": {"exists": True, "mtime_epoch": 1_000_000.0 + 3 * 86400},
    }
    spread = cohort_spread_s(prov)
    assert spread == 3 * 86400
    assert spread > 6 * 3600  # would trip the default gate


def test_cohort_spread_none_when_single_or_absent():
    assert cohort_spread_s({"classical": {"exists": True, "mtime_epoch": 1.0}}) is None
    assert cohort_spread_s({"agentic": {"exists": False}}) is None
    # a coherent same-run cohort is a tiny spread, well under the gate
    prov = {
        "classical": {"exists": True, "mtime_epoch": 500.0},
        "agentic": {"exists": True, "mtime_epoch": 560.0},
    }
    assert cohort_spread_s(prov) == 60.0


def test_stats_basic():
    s = _stats([0.0, 1.0, 2.0, 20.0])  # 3/4 under 15s
    assert s["n"] == 4
    assert s["median"] == 1.5
    assert s["lt15_pct"] == 75.0
    assert s["mae"] == 5.75


def test_stats_empty():
    s = _stats([])
    assert s["n"] == 0 and s["median"] is None and s["p90"] is None


def test_summarize_pileup_exclusion():
    # a big-error span sitting in a medley pileup (density>=4) must NOT count
    # against the noPile placement axis, but DOES count in strict.
    pairs = [(1.0, 1), (2.0, 2), (300.0, PILEUP_DENSITY + 1)]
    out = summarize(pairs)
    assert out["strict"]["n"] == 3
    assert out["strict"]["median"] == 2.0  # median of [1, 2, 300]
    assert out["noPile"]["n"] == 2  # pileup span dropped
    assert out["noPile"]["median"] == 1.5  # median of [1, 2]
    assert out["noPile"]["p90"] < out["strict"]["p90"]  # tail no longer penalized


def test_baseline_timeline_is_load_timeline_valid():
    rows = [
        {
            "slot_label": "1",
            "track_id": "recA",
            "set_end_s": 60.0,
            "claimed_stem": "regular",
        },
        {
            "slot_label": "2w1",
            "track_id": "recB",
            "set_end_s": 90.0,
            "claimed_stem": "acappella",
        },
        {
            "slot_label": "3",
            "track_id": "recC",
            "set_end_s": 30.0,
        },  # no pred -> dropped
    ]
    preds = {"1": 25.3, "2w1": 40.5}
    tl = baseline_timeline("unittestset", rows, preds)
    assert tl["set_id"] == "unittestset"
    assert {s["slot_label"] for s in tl["spans"]} == {"1", "2w1"}  # slot 3 dropped
    # score_spans' trajectory helper reads s["ref_start_s"] directly (KeyError
    # otherwise) — every baseline span must carry it, not just be load_timeline-valid.
    assert all("ref_start_s" in s for s in tl["spans"])

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(tl, fh)
        p = Path(fh.name)
    try:
        rec = load_timeline(p)  # raises on schema violation
        assert rec.set_id == "unittestset"
        assert len(rec.spans) == 2
        by_slot = {s.slot_label: s for s in rec.spans}
        assert by_slot["1"].set_start_s == 25.3
        assert by_slot["1"].recording_id == "recA"
    finally:
        p.unlink(missing_ok=True)
