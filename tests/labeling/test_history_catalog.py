"""C2/C3 — certificate-gated historical catalog emit + sound id_source strata."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ingest.content_history import SCHEMA, Generation, append_generation_on
from labeling.extract._shared import _load_content_catalog
from labeling.extract.export_als_to_gt import id_coverage
from labeling.identity.build_content_catalog import build_catalog


def _db_with_history(tmp_path: Path) -> sqlite3.Connection:
    c = sqlite3.connect(tmp_path / "t.db")
    c.row_factory = sqlite3.Row
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
    c.executescript(SCHEMA)
    return c


def test_derivation_cert_emits_historical_separated(tmp_path: Path) -> None:
    """Old stem hash + parent_content matching current master → historical emit."""
    conn = _db_with_history(tmp_path)
    conn.execute(
        "INSERT INTO set_track_slots VALUES(?,?,?,?)", ("s", 0, "recA", "recA")
    )
    conn.execute(
        "INSERT INTO track_audio VALUES(?,?,?,?,?,?,?)",
        (1, "recA", "regular", "MASTER_NOW", "/x/a.m4a", "regular", "recA"),
    )
    append_generation_on(
        conn,
        Generation(
            recording_id="recA",
            stem="acappella",
            variant="regular",
            kind="separated",
            content_sha256="OLD_STEM_SHA",
            parent_content_sha256="MASTER_NOW",
            op="re-separate",
            source="test",
        ),
    )
    conn.commit()

    out = build_catalog(conn, "s", mdat_sha256=lambda p: None)
    hist = [
        e
        for e in out["entries"]
        if e.get("id_source") == "historical-content"
        and e["content_sha256"] == "OLD_STEM_SHA"
    ]
    assert len(hist) == 1
    assert hist[0]["cert"] == "derivation"
    assert hist[0]["stem"] == "acappella"


def test_uncertified_historical_not_emitted(tmp_path: Path) -> None:
    """Separated gen whose parent hash no longer matches current master → skip."""
    conn = _db_with_history(tmp_path)
    conn.execute(
        "INSERT INTO set_track_slots VALUES(?,?,?,?)", ("s", 0, "recA", "recA")
    )
    conn.execute(
        "INSERT INTO track_audio VALUES(?,?,?,?,?,?,?)",
        (1, "recA", "regular", "MASTER_NOW", "/x/a.m4a", "regular", "recA"),
    )
    append_generation_on(
        conn,
        Generation(
            recording_id="recA",
            stem="acappella",
            variant="regular",
            kind="separated",
            content_sha256="ORPHAN_STEM",
            parent_content_sha256="MASTER_OLD",  # parent was replaced
            op="re-separate",
        ),
    )
    conn.commit()
    out = build_catalog(conn, "s", mdat_sha256=lambda p: None)
    assert not any(e["content_sha256"] == "ORPHAN_STEM" for e in out["entries"])


def test_tombstoned_historical_not_emitted(tmp_path: Path) -> None:
    conn = _db_with_history(tmp_path)
    conn.execute(
        "INSERT INTO set_track_slots VALUES(?,?,?,?)", ("s", 0, "recA", "recA")
    )
    conn.execute(
        "INSERT INTO track_audio VALUES(?,?,?,?,?,?,?)",
        (1, "recA", "regular", "MASTER_NOW", "/x/a.m4a", "regular", "recA"),
    )
    append_generation_on(
        conn,
        Generation(
            recording_id="recA",
            stem="acappella",
            variant="regular",
            kind="separated",
            content_sha256="DEAD_STEM",
            parent_content_sha256="MASTER_NOW",
            valid=0,
        ),
    )
    conn.commit()
    out = build_catalog(conn, "s", mdat_sha256=lambda p: None)
    assert not any(e["content_sha256"] == "DEAD_STEM" for e in out["entries"])


def test_load_catalog_preserves_historical_id_source(tmp_path: Path) -> None:
    set_dir = tmp_path / "set"
    set_dir.mkdir()
    (set_dir / "content_catalog.json").write_text(
        json.dumps(
            {
                "set_id": "s",
                "entries": [
                    {
                        "content_sha256": "HIST",
                        "payload_sha256": None,
                        "recording_id": "recA",
                        "track_audio_id": "9",
                        "stem": "acappella",
                        "variant": "regular",
                        "kind": "separated",
                        "id_source": "historical-content",
                        "cert": "derivation",
                    }
                ],
            }
        )
    )
    cat = _load_content_catalog(set_dir)
    assert cat is not None
    entry = cat.by_head_hash["HIST"]
    assert entry.id_source == "historical-content"
    assert entry.cert == "derivation"


def test_id_coverage_counts_historical_content() -> None:
    tracks = [
        type("T", (), {"id_source": "content"})(),
        type("T", (), {"id_source": "historical-content"})(),
        type("T", (), {"id_source": "abstain"})(),
    ]
    resolved, total, frac = id_coverage(tracks)
    assert (resolved, total) == (2, 3)
    assert abs(frac - 2 / 3) < 1e-9
