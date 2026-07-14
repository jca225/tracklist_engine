from __future__ import annotations

import sqlite3
from pathlib import Path

from lab.appleseed.appleseed_librarian import _best_hit, _safe_stem, _claim_one


def test_safe_stem_sanitizes() -> None:
    assert _safe_stem("AC/DC - Back in Black!! ") == "AC_DC - Back in Black"


def test_best_hit_prefers_validated_then_shortest_reasonable(monkeypatch) -> None:
    # _best_hit takes a list of hit-like objects (title, url, duration_s) and
    # returns the one to download, or None. A too-short (<60s) preview loses.
    class H:
        def __init__(self, title, url, dur):
            self.title, self.url, self.duration_s = title, url, dur

    hits = [
        H("Song (Preview)", "u1", 30),
        H("Song", "u2", 210),
        H("Song (Live)", "u3", 240),
    ]
    assert _best_hit(hits).url == "u2"  # full studio-length, not preview, not live
    assert _best_hit([H("x", "u", 20)]) is None  # only a preview → no confident pick


def test_claim_one_is_atomic(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE song_requests(id INTEGER PRIMARY KEY, query TEXT, kind TEXT,"
        " status TEXT, error TEXT, created_by TEXT, created_at REAL)"
    )
    con.execute(
        "INSERT INTO song_requests(id,query,kind,status,created_at) VALUES"
        "(1,'a','name','searching',0),(2,'b','name','done',0)"
    )
    con.commit()
    con.close()
    first = _claim_one(db)
    assert first is not None and first["id"] == 1
    # once claimed it flips to 'searching'->in-progress marker so a 2nd poller skips it
    second = _claim_one(db)
    assert second is None
