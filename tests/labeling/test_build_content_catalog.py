from __future__ import annotations

import sqlite3
from pathlib import Path

from labeling.build_content_catalog import build_catalog


def _db(tmp_path: Path) -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.executescript(
        """
        CREATE TABLE set_track_slots(set_id TEXT, row_index INTEGER,
            recording_id TEXT, track_id TEXT);
        CREATE TABLE track_audio(track_audio_id INTEGER PRIMARY KEY,
            recording_id TEXT, stem TEXT, sha256 TEXT, path TEXT);
        CREATE TABLE track_stems(track_audio_id INTEGER, stem_name TEXT, path TEXT);
        """
    )
    return c


def test_build_catalog_covers_track_audio_and_stems(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    vocals = tmp_path / "vocals.flac"
    vocals.write_bytes(b"VOCALS-STEM-BYTES" * 100)
    conn.executemany(
        "INSERT INTO set_track_slots VALUES(?,?,?,?)",
        [("s", 0, "recA", "recA"), ("s", 1, "recB", "recB")],
    )
    conn.executemany(
        "INSERT INTO track_audio VALUES(?,?,?,?,?)",
        [
            (1, "recA", "regular", "shaA", "/x/a.m4a"),
            (2, "recB", "acappella", "shaB", "/x/b.m4a"),
        ],
    )
    conn.execute("INSERT INTO track_stems VALUES(1, 'vocals', ?)", (str(vocals),))

    out = build_catalog(
        conn,
        "s",
        file_sha256=lambda p: "STEMHASH" if p == str(vocals) else "?",
        mdat_sha256=lambda p: None,  # skip real mp4 parsing in unit test
    )
    got = {(e["recording_id"], e["stem"], e["content_sha256"]) for e in out["entries"]}
    assert ("recA", "regular", "shaA") in got
    assert ("recB", "acappella", "shaB") in got
    assert ("recA", "acappella", "STEMHASH") in got  # demucs vocals -> acappella
    assert out["set_id"] == "s"
