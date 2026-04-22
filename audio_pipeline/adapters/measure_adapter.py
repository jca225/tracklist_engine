"""Measure-grid persistence adapter.

Populates the first-class `track_measures` and `set_measures` tables from
the JSON blobs already produced by `audio_pipeline.analysis` and stored in
`track_analysis.measure_times_json` / `set_analysis.measure_times_json`.

Why bother, when the JSON is already there?

Measure-level alignment (Stage-3 revamp) reads the measure grid hundreds
of times per set, across many workers, and wants to join against per-track
BPM / key data. Normalised tables make those queries index-driven
(O(log n) per measure lookup) and let downstream code `JOIN` measures
against `measure_alignment` without re-parsing JSON on every read.

This adapter is idempotent: it wipes and re-writes the measure rows for
each (track_audio_id | set_audio_id) it's asked about, so running it on
the same analysis output twice is a no-op on the final table contents.

Pure batch work — no per-row domain logic — so the catch-block simply
lifts sqlite exceptions into `DbError` at the library boundary.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from ..errors import DbError
from ..result import Err, Ok, Result


@contextmanager
def _connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        yield conn
    finally:
        conn.close()


def _measure_rows(measure_times: list[float]) -> list[tuple[int, float, float]]:
    """Turn a flat list of measure-start timestamps into (idx, start, end)
    triples. The last measure's `end_s` is synthesised by doubling its
    own duration — it's the best we can do without the audio duration in
    hand, and downstream code should treat the final measure as
    approximate-end anyway.
    """
    if len(measure_times) < 2:
        return []
    rows: list[tuple[int, float, float]] = []
    for i in range(len(measure_times) - 1):
        rows.append((i, float(measure_times[i]), float(measure_times[i + 1])))
    # Final measure: extrapolate one measure-length forward.
    last_i = len(measure_times) - 1
    last_start = float(measure_times[last_i])
    prev_dur = last_start - float(measure_times[last_i - 1])
    rows.append((last_i, last_start, last_start + prev_dur))
    return rows


def _bpm_for_measures(rows: list[tuple[int, float, float]]) -> list[float]:
    """Per-measure BPM from measure duration, assuming 4 beats/measure.

    `bpm = 4 / (end - start) * 60`. This matches what beat_this's measure
    grid encodes: the grid is downbeat-synced, so every measure spans
    exactly one bar.
    """
    bpms: list[float] = []
    for _, start, end in rows:
        dur = end - start
        bpms.append(240.0 / dur if dur > 1e-6 else 0.0)
    return bpms


def persist_track_measures(
    db_path: Path, track_audio_id: int,
) -> Result[int, DbError]:
    """Read `track_analysis.measure_times_json` for this audio asset and
    replace the corresponding `track_measures` rows. Returns count of
    rows written."""
    try:
        with _connect(db_path) as conn:
            row = conn.execute(
                """
                SELECT tan.measure_times_json,
                       (SELECT taf.key_pc   FROM track_audio_features taf
                          WHERE taf.track_audio_id = tan.track_audio_id
                          ORDER BY taf.analyzed_at DESC LIMIT 1) AS key_pc,
                       (SELECT taf.key_mode FROM track_audio_features taf
                          WHERE taf.track_audio_id = tan.track_audio_id
                          ORDER BY taf.analyzed_at DESC LIMIT 1) AS key_mode
                FROM track_analysis tan
                WHERE tan.track_audio_id = ?
                """,
                (track_audio_id,),
            ).fetchone()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="query_failed", detail=str(e)))

    if row is None or row["measure_times_json"] is None:
        return Err(DbError(
            kind="not_found",
            detail=f"no track_analysis for track_audio_id={track_audio_id}",
        ))

    try:
        measure_times = list(json.loads(row["measure_times_json"]))
    except (json.JSONDecodeError, TypeError) as e:
        return Err(DbError(kind="query_failed", detail=f"parse: {e}"))

    rows = _measure_rows(measure_times)
    bpms = _bpm_for_measures(rows)
    key_pc = row["key_pc"]
    key_mode = row["key_mode"]

    try:
        with _connect(db_path) as conn:
            conn.execute("BEGIN")
            conn.execute(
                "DELETE FROM track_measures WHERE track_audio_id = ?",
                (track_audio_id,),
            )
            conn.executemany(
                """
                INSERT INTO track_measures
                  (track_audio_id, measure_idx, start_s, end_s, bpm, key_pc, key_mode)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (track_audio_id, idx, start, end, bpm, key_pc, key_mode)
                    for (idx, start, end), bpm in zip(rows, bpms)
                ],
            )
            conn.commit()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="integrity", detail=str(e)))

    return Ok(len(rows))


def persist_set_measures(
    db_path: Path, set_audio_id: int,
) -> Result[int, DbError]:
    """Read `set_analysis.measure_times_json` for this mix and replace the
    corresponding `set_measures` rows."""
    try:
        with _connect(db_path) as conn:
            row = conn.execute(
                "SELECT measure_times_json FROM set_analysis WHERE set_audio_id = ?",
                (set_audio_id,),
            ).fetchone()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="query_failed", detail=str(e)))

    if row is None or row["measure_times_json"] is None:
        return Err(DbError(
            kind="not_found",
            detail=f"no set_analysis for set_audio_id={set_audio_id}",
        ))

    try:
        measure_times = list(json.loads(row["measure_times_json"]))
    except (json.JSONDecodeError, TypeError) as e:
        return Err(DbError(kind="query_failed", detail=f"parse: {e}"))

    rows = _measure_rows(measure_times)
    bpms = _bpm_for_measures(rows)

    try:
        with _connect(db_path) as conn:
            conn.execute("BEGIN")
            conn.execute(
                "DELETE FROM set_measures WHERE set_audio_id = ?",
                (set_audio_id,),
            )
            conn.executemany(
                """
                INSERT INTO set_measures
                  (set_audio_id, measure_idx, start_s, end_s, bpm)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (set_audio_id, idx, start, end, bpm)
                    for (idx, start, end), bpm in zip(rows, bpms)
                ],
            )
            conn.commit()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="integrity", detail=str(e)))

    return Ok(len(rows))


def persist_track_sections(
    db_path: Path, track_audio_id: int,
) -> Result[int, DbError]:
    """Mirror `track_mert_sections` (idx, start_s, end_s) into the
    boundary-only `track_sections` table. Drops the BLOB embedding —
    callers who want the embedding should keep using track_mert_sections.
    """
    try:
        with _connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT section_idx, start_s, end_s
                FROM track_mert_sections
                WHERE track_audio_id = ?
                ORDER BY section_idx
                """,
                (track_audio_id,),
            ).fetchall()
            conn.execute("BEGIN")
            conn.execute(
                "DELETE FROM track_sections WHERE track_audio_id = ?",
                (track_audio_id,),
            )
            conn.executemany(
                """
                INSERT INTO track_sections
                  (track_audio_id, section_idx, start_s, end_s, kind)
                VALUES (?, ?, ?, ?, NULL)
                """,
                [
                    (track_audio_id, r["section_idx"], r["start_s"], r["end_s"])
                    for r in rows
                ],
            )
            conn.commit()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="integrity", detail=str(e)))
    return Ok(len(rows))


def backfill_all(db_path: Path) -> Result[dict[str, int], DbError]:
    """One-shot backfill for every track + set that has analysis data.

    Useful after schema migration: walks `track_analysis` + `set_analysis`
    and populates the new measure tables for everything already analysed.
    Returns counts per table for reporting.
    """
    try:
        with _connect(db_path) as conn:
            track_ids = [r[0] for r in conn.execute(
                "SELECT track_audio_id FROM track_analysis "
                "WHERE measure_times_json IS NOT NULL"
            ).fetchall()]
            set_ids = [r[0] for r in conn.execute(
                "SELECT set_audio_id FROM set_analysis "
                "WHERE measure_times_json IS NOT NULL"
            ).fetchall()]
            mert_track_ids = [r[0] for r in conn.execute(
                "SELECT DISTINCT track_audio_id FROM track_mert_sections"
            ).fetchall()]
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="query_failed", detail=str(e)))

    counts = {"track_measures": 0, "set_measures": 0, "track_sections": 0}
    for tid in track_ids:
        r = persist_track_measures(db_path, tid)
        if r.is_ok():
            counts["track_measures"] += r.value
    for sid in set_ids:
        r = persist_set_measures(db_path, sid)
        if r.is_ok():
            counts["set_measures"] += r.value
    for tid in mert_track_ids:
        r = persist_track_sections(db_path, tid)
        if r.is_ok():
            counts["track_sections"] += r.value
    return Ok(counts)
