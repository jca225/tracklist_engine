from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from labeling.identity.build_content_catalog import build_catalog


def _db(tmp_path: Path) -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.executescript(
        """
        CREATE TABLE set_track_slots(set_id TEXT, row_index INTEGER,
            recording_id TEXT, track_id TEXT);
        CREATE TABLE track_audio(track_audio_id INTEGER PRIMARY KEY,
            recording_id TEXT, stem TEXT, sha256 TEXT, path TEXT, variant TEXT,
            track_id TEXT);
        CREATE TABLE track_stems(track_audio_id INTEGER, stem_name TEXT, path TEXT);
        """
    )
    return c


def test_build_catalog_covers_track_audio_and_stems(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    vocals = tmp_path / "vocals.flac"
    vocals.write_bytes(b"VOCALS-STEM-BYTES" * 100)
    instr = tmp_path / "instrumental.flac"
    instr.write_bytes(b"INSTRUMENTAL-STEM-BYTES" * 100)
    conn.executemany(
        "INSERT INTO set_track_slots VALUES(?,?,?,?)",
        [("s", 0, "recA", "recA"), ("s", 1, "recB", "recB"), ("s", 2, "recC", "recC")],
    )
    conn.executemany(
        "INSERT INTO track_audio VALUES(?,?,?,?,?,?,?)",
        [
            (1, "recA", "regular", "shaA", "/x/a.m4a", "extended", "recA"),
            (2, "recB", "acappella", "shaB", "/x/b.m4a", None, "recB"),
            (3, "recC", "regular", "shaC", "/x/c.m4a", "regular", "recC"),
        ],
    )
    conn.execute("INSERT INTO track_stems VALUES(1, 'vocals', ?)", (str(vocals),))
    conn.execute("INSERT INTO track_stems VALUES(3, 'instrumental', ?)", (str(instr),))

    out = build_catalog(
        conn,
        "s",
        file_sha256=lambda p: {str(vocals): "STEMHASH", str(instr): "INSTRHASH"}.get(
            p, "?"
        ),
        mdat_sha256=lambda p: None,  # skip real mp4 parsing in unit test
    )
    got = {
        (e["recording_id"], e["stem"], e["variant"], e["content_sha256"], e["kind"])
        for e in out["entries"]
    }
    assert ("recA", "regular", "extended", "shaA", "master") in got
    assert (
        "recB",
        "acappella",
        "regular",
        "shaB",
        "master",
    ) in got  # NULL variant -> regular
    # demucs vocals of recA's regular parent -> acappella, kind='separated'.
    # A separation preserves the parent master's length, so a stem of an
    # extended parent is itself extended — the stem must inherit variant,
    # not hardcode "regular".
    assert ("recA", "acappella", "extended", "STEMHASH", "separated") in got
    # Positive control: a stem of a variant='regular' parent still emits
    # 'regular' (not just always echoing whatever the last parent was).
    assert ("recC", "instrumental", "regular", "INSTRHASH", "separated") in got
    assert out["set_id"] == "s"


def test_flac_payload_key_wired_for_masters_and_stems(tmp_path: Path) -> None:
    """B3/P11: a FLAC master and a FLAC separated stem now carry a
    payload_sha256 (decoded-PCM MD5) — previously the stem population always
    emitted payload None, needlessly abstaining on re-encoded/re-containered
    FLACs. mdat stays for mp4; the dispatcher picks by extension."""
    conn = _db(tmp_path)
    conn.execute(
        "INSERT INTO set_track_slots VALUES(?,?,?,?)", ("s", 0, "recF", "recF")
    )
    conn.execute(
        "INSERT INTO track_audio VALUES(?,?,?,?,?,?,?)",
        (1, "recF", "regular", "shaF", "/x/master.flac", "regular", "recF"),
    )
    stem = tmp_path / "vocals.flac"
    stem.write_bytes(b"x")
    conn.execute("INSERT INTO track_stems VALUES(1, 'vocals', ?)", (str(stem),))

    payloads = {"/x/master.flac": "PCM_MASTER", str(stem): "PCM_VOX"}
    out = build_catalog(
        conn,
        "s",
        file_sha256=lambda p: "STEMSHA",
        mdat_sha256=lambda p: pytest.fail("mdat must not be called for .flac"),
        flac_pcm_md5=lambda p: payloads.get(str(p)),
    )
    by_stem = {e["stem"]: e for e in out["entries"]}
    assert by_stem["regular"]["payload_sha256"] == "PCM_MASTER"  # FLAC master
    assert by_stem["acappella"]["payload_sha256"] == "PCM_VOX"  # FLAC separated stem


def test_legacy_slot_null_recording_id_track_id_keyed_audio_is_included(
    tmp_path: Path,
) -> None:
    """Hard case (P10): a set_track_slots row with recording_id=NULL,
    track_id='legZ' (the Rvmor-gap shape), whose track_audio row is keyed by
    track_id only (recording_id also NULL), must still appear in the catalog —
    this is exactly the row the pull's dual-key `wanted` CTE recovers. Prior to
    the A4 fix this row is invisible (single-key recording_id-only match)."""
    conn = _db(tmp_path)
    conn.execute("INSERT INTO set_track_slots VALUES(?,?,?,?)", ("s", 0, None, "legZ"))
    conn.execute(
        "INSERT INTO track_audio VALUES(?,?,?,?,?,?,?)",
        (3, None, "regular", "shaZ", "/x/z.m4a", "regular", "legZ"),
    )

    out = build_catalog(
        conn,
        "s",
        file_sha256=lambda p: "?",
        mdat_sha256=lambda p: None,
    )

    got = {(e["recording_id"], e["content_sha256"], e["kind"]) for e in out["entries"]}
    assert ("legZ", "shaZ", "master") in got, (
        "legacy track_id-keyed slot/audio must be included with COALESCEd identity"
    )


def test_legacy_slot_null_recording_id_track_id_keyed_stems_is_included(
    tmp_path: Path,
) -> None:
    """Legacy-shape coverage for the SECOND (track_stems) loop's dual-key
    match: a set_track_slots row with recording_id=NULL, track_id='legS', a
    regular track_audio parent keyed the same way (recording_id=NULL,
    track_id='legS'), and a demucs vocals stem under it. All of this
    module's other tests use rows where recording_id==track_id, so a
    wrong-column/param-order regression in the track_stems loop's
    `ta.track_id IN (...) OR ta.recording_id IN (...)` clause would slip
    past them undetected."""
    conn = _db(tmp_path)
    vocals = tmp_path / "legacy_vocals.flac"
    vocals.write_bytes(b"LEGACY-VOCALS-BYTES" * 100)
    conn.execute("INSERT INTO set_track_slots VALUES(?,?,?,?)", ("s", 0, None, "legS"))
    conn.execute(
        "INSERT INTO track_audio VALUES(?,?,?,?,?,?,?)",
        (4, None, "regular", "shaS", "/x/s.m4a", "regular", "legS"),
    )
    conn.execute("INSERT INTO track_stems VALUES(4, 'vocals', ?)", (str(vocals),))

    out = build_catalog(
        conn,
        "s",
        file_sha256=lambda p: "LEGACYSTEMHASH" if p == str(vocals) else "?",
        mdat_sha256=lambda p: None,
    )

    got = {
        (e["recording_id"], e["stem"], e["content_sha256"], e["kind"])
        for e in out["entries"]
    }
    assert ("legS", "acappella", "LEGACYSTEMHASH", "separated") in got


def test_track_audio_for_recording_not_in_this_sets_slots_is_excluded(
    tmp_path: Path,
) -> None:
    """No-over-inclusion guard: widening the WHERE to a dual-key match must not
    accidentally pull in track_audio rows for recordings outside this set's
    slots."""
    conn = _db(tmp_path)
    conn.execute(
        "INSERT INTO set_track_slots VALUES(?,?,?,?)", ("s", 0, "recA", "recA")
    )
    conn.executemany(
        "INSERT INTO track_audio VALUES(?,?,?,?,?,?,?)",
        [
            (1, "recA", "regular", "shaA", "/x/a.m4a", "regular", "recA"),
            # recC is not referenced by any slot in set "s" — must NOT appear.
            (2, "recC", "regular", "shaC", "/x/c.m4a", "regular", "recC"),
        ],
    )

    out = build_catalog(
        conn,
        "s",
        file_sha256=lambda p: "?",
        mdat_sha256=lambda p: None,
    )

    rids = {e["recording_id"] for e in out["entries"]}
    assert rids == {"recA"}
