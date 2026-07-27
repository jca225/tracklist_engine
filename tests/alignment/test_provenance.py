"""Provenance stamp + freshness check: the timeline staleness detector.

Guards the fix for the "is this July-6 file stale?" fog (state-of-record #18):
a stamp must detect drift in each input it covers, and must NOT false-flag when
nothing changed.
"""

from __future__ import annotations

import json
from pathlib import Path

from alignment import provenance

ROWS = (
    {"slot_label": "1", "recording_id": "aaa", "claimed_stem": "regular"},
    {"slot_label": "2", "recording_id": "bbb", "claimed_stem": "acappella"},
)


def _gt(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "gt.yaml"
    p.write_text(text)
    return p


def test_stamp_roundtrips_fresh(tmp_path):
    gt = _gt(tmp_path, "set_id: x\n")
    tl = {"provenance": provenance.stamp("1fsnxchk", ROWS, gt_paths=[gt])}
    fresh, drift = provenance.check(tl, "1fsnxchk", rows=ROWS, gt_paths=[gt])
    assert fresh and drift == [], drift


def test_unstamped_is_flagged():
    fresh, drift = provenance.check({"spans": []}, "1fsnxchk")
    assert not fresh and drift == ["unstamped"]


def test_detects_spine_drift(tmp_path):
    gt = _gt(tmp_path, "set_id: x\n")
    tl = {"provenance": provenance.stamp("1fsnxchk", ROWS, gt_paths=[gt])}
    mutated = (
        {"slot_label": "1", "recording_id": "DIFFERENT", "claimed_stem": "regular"},
        ROWS[1],
    )
    fresh, drift = provenance.check(tl, "1fsnxchk", rows=mutated, gt_paths=[gt])
    assert not fresh
    assert any("spine" in d for d in drift), drift


def test_detects_gt_drift(tmp_path):
    gt = _gt(tmp_path, "set_id: x\n")
    tl = {"provenance": provenance.stamp("1fsnxchk", ROWS, gt_paths=[gt])}
    gt.write_text("set_id: x\nchanged: true\n")
    fresh, drift = provenance.check(tl, "1fsnxchk", rows=ROWS, gt_paths=[gt])
    assert not fresh
    assert any("GT" in d for d in drift), drift


def test_spine_skipped_when_rows_absent(tmp_path):
    # pi unreachable -> rows not supplied -> spine must NOT be false-flagged
    gt = _gt(tmp_path, "set_id: x\n")
    tl = {"provenance": provenance.stamp("1fsnxchk", ROWS, gt_paths=[gt])}
    fresh, drift = provenance.check(tl, "1fsnxchk", gt_paths=[gt])
    assert fresh and drift == [], drift


def test_spine_hash_is_order_stable():
    assert provenance.spine_hash(ROWS) == provenance.spine_hash(tuple(reversed(ROWS)))
