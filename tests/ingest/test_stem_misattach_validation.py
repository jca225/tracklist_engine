from __future__ import annotations

from pathlib import Path

from scripts.audit_stem_recording_links import parse_acquired_song, scan_ledger_tsv

FIXTURE = Path("tests/fixtures/ingest/correction_ledger_snapshot_20260709.tsv")


def test_parse_acquired_song_from_file_reason():
    # ledger reasons for stem/add rows embed the acquired file/song
    got = parse_acquired_song(
        "file: 148__Come On Over Baby (Acapella).wav ; auto-attached"
    )
    assert got is not None
    assert "come on over baby" in got.lower()


def test_scan_returns_stem_add_rows():
    rows = scan_ledger_tsv(FIXTURE)
    assert len(rows) > 0
    # every returned row is a stem-axis add with a parsed acquired song
    assert all(r["stem_value"] in ("acappella", "instrumental") for r in rows)
