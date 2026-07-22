"""Self-test for the corpus data-integrity checker (scripts/corpus_integrity.py).

A checker that can silently rot is worse than no checker: if the schema drifts
and an invariant's SQL stops matching, `make check-corpus` would report all-clear
on a broken corpus. So we assert the checker fires on a DELIBERATELY broken DB and
stays silent on a clean one.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from scripts import corpus_integrity as ci

_SCHEMA = (
    Path(__file__).resolve().parent.parent / "web_crawler" / "database" / "schema.sql"
)


def _fresh_conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.executescript(_SCHEMA.read_text())
    # These tests exercise the invariant SQL, not referential integrity; insert
    # audio rows directly without seeding parent work/recording rows.
    c.execute("PRAGMA foreign_keys=OFF")
    return c


@pytest.fixture
def conn() -> sqlite3.Connection:
    return _fresh_conn()


def _counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {inv.key: n for inv, n in ci.evaluate(conn)}


def _add_audio(
    conn: sqlite3.Connection,
    recording_id: str,
    *,
    taid: int,
    stem: str = "regular",
    variant: str = "regular",
    is_reference: int = 0,
    platform: str = "youtube",
    path: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO track_audio (track_audio_id, recording_id, track_id, platform, "
        "source_url, player_id, path, stem, variant, is_reference) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            taid,
            recording_id,
            recording_id,
            platform,
            "u",
            f"p{taid}",
            f"/o/{recording_id}/{taid}.m4a" if path is None else path,
            stem,
            variant,
            is_reference,
        ),
    )
    conn.commit()


def test_clean_db_has_zero_errors(conn: sqlite3.Connection) -> None:
    _add_audio(conn, "R1", taid=1, is_reference=1)
    counts = _counts(conn)
    errors = [inv.key for inv, n in ci.evaluate(conn) if inv.severity == "ERROR" and n]
    assert errors == [], f"clean DB should have no ERROR violations, got {errors}"
    assert counts["dup_reference"] == 0


def test_dup_reference_fires(conn: sqlite3.Connection) -> None:
    _add_audio(conn, "R1", taid=1, is_reference=1)
    _add_audio(conn, "R1", taid=2, is_reference=1, platform="soundcloud")
    assert _counts(conn)["dup_reference"] == 1


def test_reference_on_stem_when_regular_exists_fires(conn: sqlite3.Connection) -> None:
    _add_audio(conn, "R1", taid=1, stem="regular", is_reference=0)
    _add_audio(conn, "R1", taid=2, stem="acappella", is_reference=1, platform="sc")
    assert _counts(conn)["reference_on_stem_when_regular_exists"] == 1


def test_null_path_fires(conn: sqlite3.Connection) -> None:
    _add_audio(conn, "R1", taid=1, path="")
    assert _counts(conn)["null_path"] == 1


def test_regular_claim_served_by_stem_fires(conn: sqlite3.Connection) -> None:
    # a set plays R1 as regular, but its only reference is an acappella stem
    _add_audio(conn, "R1", taid=1, stem="acappella", is_reference=1)
    conn.execute(
        "INSERT INTO set_track_slots (set_id, row_index, track_id, recording_id, "
        "claimed_stem) VALUES ('s', 0, 'R1', 'R1', 'regular')"
    )
    conn.commit()
    assert _counts(conn)["regular_claim_served_by_stem"] == 1


def test_same_platform_dup_regular_fires(conn: sqlite3.Connection) -> None:
    _add_audio(conn, "R1", taid=1, platform="youtube")
    # UNIQUE(recording_id, platform, player_id) allows a 2nd youtube row via
    # a different player_id — the true-duplicate case the invariant catches
    conn.execute(
        "INSERT INTO track_audio (track_audio_id, recording_id, track_id, platform, "
        "source_url, player_id, path, stem, variant) "
        "VALUES (2,'R1','R1','youtube','u','p2','/o/R1/2.m4a','regular','regular')"
    )
    conn.commit()
    assert _counts(conn)["same_platform_dup_regular"] == 1


def test_mojibake_path_fires_and_clean_is_silent(conn: sqlite3.Connection) -> None:
    # A correctly-stored non-ASCII path is clean; its double-encoded form fires.
    correct = "/o/R1/Sign Of The Times – Acapella.m4a"  # real en-dash U+2013
    mojibake = correct.encode("utf-8").decode("latin-1")  # â€" — issue #74
    _add_audio(conn, "R1", taid=1, path=correct)
    assert _counts(conn)["mojibake_path"] == 0
    _add_audio(conn, "R2", taid=2, path=mojibake)
    assert _counts(conn)["mojibake_path"] == 1


def test_mojibake_path_is_error_severity() -> None:
    inv = next(i for i in ci.INVARIANTS if i.key == "mojibake_path")
    assert inv.severity == "ERROR"


def test_every_invariant_key_is_covered() -> None:
    # guards against an invariant being added without a schema-valid count query
    c = _fresh_conn()
    _add_audio(c, "R1", taid=1, is_reference=1)
    # evaluate() must not raise on any invariant SQL (schema-drift canary)
    results = ci.evaluate(c)
    assert len(results) == len(ci.INVARIANTS)
    assert all(isinstance(n, int) for _, n in results)
