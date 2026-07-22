"""Task A3 — catalog stem-derivation correctness: parent filter + `kind`.

Covers:
- P14: a separated stem (track_stems) is only a valid acappella/instrumental
  catalog entry when its parent track_audio row is the regular master. A
  separated stem hung off an acappella (or other non-regular) parent must
  NOT be emitted.
- P15: raw component-stem names (drums/bass/other) must never be passed
  through as a catalog `stem` value — they must be excluded entirely.
- `kind`: every emitted entry carries kind='master' (from track_audio) or
  kind='separated' (from track_stems).
"""

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
            recording_id TEXT, stem TEXT, sha256 TEXT, path TEXT, variant TEXT);
        CREATE TABLE track_stems(track_audio_id INTEGER, stem_name TEXT, path TEXT);
        """
    )
    return c


def test_separated_stem_under_acappella_parent_is_excluded(tmp_path: Path) -> None:
    """P14: parent track_audio.stem must be 'regular' for a separated stem to count."""
    conn = _db(tmp_path)

    reg_vocals = tmp_path / "reg_vocals.flac"
    reg_vocals.write_bytes(b"REG-VOCALS" * 100)
    reg_instr = tmp_path / "reg_instrumental.flac"
    reg_instr.write_bytes(b"REG-INSTR" * 100)
    bad_instr = tmp_path / "bad_instrumental.flac"
    bad_instr.write_bytes(b"BAD-INSTR-FROM-ACAP-PARENT" * 100)

    conn.executemany(
        "INSERT INTO set_track_slots VALUES(?,?,?,?)",
        [("s", 0, "recA", "recA"), ("s", 1, "recB", "recB")],
    )
    conn.executemany(
        "INSERT INTO track_audio VALUES(?,?,?,?,?,?)",
        [
            (1, "recA", "regular", "shaA", "/x/a.m4a", "regular"),
            (2, "recB", "acappella", "shaB", "/x/b.m4a", "regular"),
        ],
    )
    conn.executemany(
        "INSERT INTO track_stems VALUES(?,?,?)",
        [
            (1, "vocals", str(reg_vocals)),
            (1, "instrumental", str(reg_instr)),
            # separated stem hung off an acappella-master parent (recB) —
            # its "instrumental" is a phase-cancel residual, not the real one.
            (2, "instrumental", str(bad_instr)),
        ],
    )

    def fake_sha(p: str) -> str:
        return {
            str(reg_vocals): "REGVOCALSHASH",
            str(reg_instr): "REGINSTRHASH",
            str(bad_instr): "BADINSTRHASH",
        }[p]

    out = build_catalog(
        conn,
        "s",
        file_sha256=fake_sha,
        mdat_sha256=lambda p: None,
    )

    hashes = {e["content_sha256"] for e in out["entries"]}
    assert "BADINSTRHASH" not in hashes, (
        "separated stem under non-regular parent must be excluded"
    )

    got = {(e["recording_id"], e["stem"], e["content_sha256"]) for e in out["entries"]}
    # positive control: regular parent's separated vocals/instrumental DO appear
    assert ("recA", "acappella", "REGVOCALSHASH") in got
    assert ("recA", "instrumental", "REGINSTRHASH") in got


def test_component_stem_names_are_excluded_not_passed_through(tmp_path: Path) -> None:
    """P15: drums/bass/other must never leak into the catalog as a raw stem value."""
    conn = _db(tmp_path)

    drums = tmp_path / "drums.flac"
    drums.write_bytes(b"DRUMS-STEM" * 100)

    conn.execute(
        "INSERT INTO set_track_slots VALUES(?,?,?,?)", ("s", 0, "recA", "recA")
    )
    conn.execute(
        "INSERT INTO track_audio VALUES(?,?,?,?,?,?)",
        (1, "recA", "regular", "shaA", "/x/a.m4a", "regular"),
    )
    conn.execute("INSERT INTO track_stems VALUES(1, 'drums', ?)", (str(drums),))

    out = build_catalog(
        conn,
        "s",
        file_sha256=lambda p: "DRUMSHASH" if p == str(drums) else "?",
        mdat_sha256=lambda p: None,
    )

    stems = {e["stem"] for e in out["entries"]}
    assert "drums" not in stems
    hashes = {e["content_sha256"] for e in out["entries"]}
    assert "DRUMSHASH" not in hashes


def test_every_entry_carries_kind(tmp_path: Path) -> None:
    conn = _db(tmp_path)

    vocals = tmp_path / "vocals.flac"
    vocals.write_bytes(b"VOCALS-STEM" * 100)

    conn.execute(
        "INSERT INTO set_track_slots VALUES(?,?,?,?)", ("s", 0, "recA", "recA")
    )
    conn.execute(
        "INSERT INTO track_audio VALUES(?,?,?,?,?,?)",
        (1, "recA", "regular", "shaA", "/x/a.m4a", "regular"),
    )
    conn.execute("INSERT INTO track_stems VALUES(1, 'vocals', ?)", (str(vocals),))

    out = build_catalog(
        conn,
        "s",
        file_sha256=lambda p: "STEMHASH" if p == str(vocals) else "?",
        mdat_sha256=lambda p: None,
    )

    assert out["entries"], "expected entries"
    for e in out["entries"]:
        assert e["kind"] in {"master", "separated"}

    master_entries = [e for e in out["entries"] if e["content_sha256"] == "shaA"]
    assert master_entries and all(e["kind"] == "master" for e in master_entries)

    separated_entries = [e for e in out["entries"] if e["content_sha256"] == "STEMHASH"]
    assert separated_entries and all(
        e["kind"] == "separated" for e in separated_entries
    )
