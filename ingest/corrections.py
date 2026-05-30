"""Append-only correction ledger (ingest stage).

Records one row each time a track's downloaded audio is replaced or a variant
added because the auto-acquired version was the wrong *identity* along one of
the three axes (version / variant / stem). See core/identity.py.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

from core.db import _connect
from core.errors import DbError
from core.identity import normalize_stem
from core.result import Err, Ok, Result

AXES = ("version", "variant", "stem")
ACTIONS = ("replace", "add")


@dataclass(frozen=True)
class Correction:
    track_id: str
    axis: str
    action: str
    set_id: str | None = None
    position: str | None = None
    old_track_audio_id: int | None = None
    old_platform: str | None = None
    old_player_id: str | None = None
    old_url: str | None = None
    new_track_audio_id: int | None = None
    new_platform: str | None = None
    new_player_id: str | None = None
    new_url: str | None = None
    stem_value: str | None = None     # regular | acappella | instrumental
    reason: str | None = None
    source: str | None = None


def log_correction(db_path: Path, c: Correction) -> Result[int, DbError]:
    if c.axis not in AXES:
        return Err(DbError(kind="bad_axis", detail=f"{c.axis!r} not in {AXES}"))
    if c.action not in ACTIONS:
        return Err(DbError(kind="bad_action", detail=f"{c.action!r} not in {ACTIONS}"))
    try:
        with _connect(db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO track_audio_correction
                  (set_id, position, track_id, axis, action,
                   old_track_audio_id, old_platform, old_player_id, old_url,
                   new_track_audio_id, new_platform, new_player_id, new_url,
                   stem_value, reason, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    c.set_id, c.position, c.track_id, c.axis, c.action,
                    c.old_track_audio_id, c.old_platform, c.old_player_id, c.old_url,
                    c.new_track_audio_id, c.new_platform, c.new_player_id, c.new_url,
                    c.stem_value, c.reason, c.source,
                ),
            )
            conn.commit()
            return Ok(int(cur.lastrowid))
    except sqlite3.Error as e:
        return Err(DbError(kind="integrity", detail=str(e)))


def snapshot_row(db_path: Path, track_audio_id: int) -> dict | None:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        r = conn.execute(
            "SELECT platform, player_id, source_url, stem "
            "FROM track_audio WHERE track_audio_id = ?",
            (track_audio_id,),
        ).fetchone()
    return dict(r) if r else None


def latest_row(db_path: Path, track_id: str, stem: str | None = None) -> dict | None:
    q = (
        "SELECT track_audio_id, platform, player_id, source_url, stem "
        "FROM track_audio WHERE recording_id = ?"
    )
    params: list[object] = [track_id]
    if stem is not None:
        q += " AND stem = ?"
        params.append(normalize_stem(stem))
    q += " ORDER BY track_audio_id DESC LIMIT 1"
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        r = conn.execute(q, tuple(params)).fetchone()
    return dict(r) if r else None


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Append a correction-ledger row (standalone / manual path).",
    )
    p.add_argument("--db", type=Path,
                   default=Path(os.environ.get("TRACKLIST_DB",
                                               "/mnt/storage/data/db/music_database.db")))
    p.add_argument("--track-id", required=True)
    p.add_argument("--axis", required=True, choices=AXES)
    p.add_argument("--action", required=True, choices=ACTIONS)
    p.add_argument("--set-id", default=None)
    p.add_argument("--position", default=None)
    p.add_argument("--old-taid", type=int, default=None)
    p.add_argument("--new-taid", type=int, default=None)
    p.add_argument("--stem", default=None, help="Stem axis value for ledger row")
    p.add_argument("--reason", default=None)
    p.add_argument("--source", default="manual")
    a = p.parse_args(argv)

    old = snapshot_row(a.db, a.old_taid) if a.old_taid is not None else None
    new = snapshot_row(a.db, a.new_taid) if a.new_taid is not None else None
    c = Correction(
        track_id=a.track_id, axis=a.axis, action=a.action,
        set_id=a.set_id, position=a.position,
        old_track_audio_id=a.old_taid,
        old_platform=(old or {}).get("platform"),
        old_player_id=(old or {}).get("player_id"),
        old_url=(old or {}).get("source_url"),
        new_track_audio_id=a.new_taid,
        new_platform=(new or {}).get("platform"),
        new_player_id=(new or {}).get("player_id"),
        new_url=(new or {}).get("source_url"),
        stem_value=a.stem or (new or {}).get("stem"),
        reason=a.reason, source=a.source,
    )
    r = log_correction(a.db, c)
    match r:
        case Ok(cid):
            print(f"logged correction_id={cid} ({a.axis}/{a.action}) for {a.track_id}")
            return 0
        case Err(e):
            print(f"log_correction failed: {e.kind} — {e.detail}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    sys.exit(_main())
