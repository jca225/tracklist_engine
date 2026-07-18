"""SoundCloud data-lake SQLite store: connection, schema init, parse + upsert."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=60.0)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 60000")
        yield conn
    finally:
        conn.close()


def init_db(db_path: Path) -> None:
    sql = SCHEMA_PATH.read_text()
    with connect(db_path) as conn:
        conn.executescript(sql)
        conn.commit()
