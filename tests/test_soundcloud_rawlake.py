# tests/test_soundcloud_rawlake.py
"""Tests for soundcloud.rawlake append-only snapshots."""

from __future__ import annotations

import json

from soundcloud import rawlake


def test_write_snapshot_returns_relref_and_writes_jsonl(tmp_path):
    ref = rawlake.write_snapshot(
        tmp_path, "likes", 7, "2026-07-18T00:00:00Z", [{"id": 1}, {"id": 2}]
    )
    assert ref == "likes/7/2026-07-18T00-00-00Z.jsonl"
    lines = (tmp_path / ref).read_text().splitlines()
    assert [json.loads(x)["id"] for x in lines] == [1, 2]


def test_empty_records_still_writes_file(tmp_path):
    ref = rawlake.write_snapshot(tmp_path, "user", 7, "2026-07-18T00:00:00Z", [])
    assert (tmp_path / ref).exists()
    assert (tmp_path / ref).read_text() == ""
