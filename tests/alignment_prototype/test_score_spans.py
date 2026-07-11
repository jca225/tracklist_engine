"""Tests for score_spans() — the per-span scoring loop extracted from score_timeline_vs_gt."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from workspaces.alignment_prototype.score_timeline_vs_gt import SpanScore, score_spans

REPO = Path(__file__).resolve().parents[2]
TL = REPO / "workspaces/alignment_prototype/out/1fsnxchk_predicted_timeline_lt.json"


@pytest.mark.skipif(not TL.exists(), reason="needs local _lt timeline")
def test_score_spans_returns_row_per_span():
    rows = score_spans("1fsnxchk", TL)  # fibers=False → no audio needed
    assert rows and all(isinstance(r, SpanScore) for r in rows)
    n_spans = len(json.loads(TL.read_text())["spans"])
    assert len(rows) == n_spans  # one row per timeline span
    # strict is populated for spans with a same-recording GT row
    assert any(r.strict is not None for r in rows)
    # fiber == strict when fibers=False (trajectory_acc contract)
    assert all(r.fiber == r.strict for r in rows if r.strict is not None)
