"""One-shot backfill: sum existing {drums, bass, other} stems into a
pre-summed `instrumental.wav` + register it in `track_stems`/`set_stems`.

Exists because the 3-hypothesis tournament expects a stem named
`instrumental`, but tracks analysed before the demucs adapter learned
to emit derived stems only have the four raw stems on disk. Rather
than re-running demucs (expensive), we sum the three on the fly.

Safe to run repeatedly: skips any (audio_id) that already has an
`instrumental` row in the stems table.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from ..errors import DbError
from ..result import Err, Ok, Result


INSTR_RECIPE: tuple[str, ...] = ("drums", "bass", "other")


@contextmanager
def _connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        yield conn
    finally:
        conn.close()


def _sum_stems(paths: tuple[Path, ...]) -> tuple[np.ndarray, int] | None:
    """Load three stem WAVs, sum sample-wise, return (mono-or-stereo, sr).

    Handles mixed mono/stereo inputs by coercing to the max channel count
    seen (demucs output is usually stereo so this is a rare fallback).
    Returns None on any load failure — caller logs and skips.
    """
    import soundfile as sf
    arrays: list[np.ndarray] = []
    sr: int | None = None
    for p in paths:
        try:
            y, file_sr = sf.read(str(p), always_2d=True)
        except (FileNotFoundError, OSError, RuntimeError):
            return None
        if sr is None:
            sr = file_sr
        elif sr != file_sr:
            return None
        arrays.append(y.astype(np.float32))
    if not arrays or sr is None:
        return None
    # Match channel counts by broadcasting mono up; truncate to shortest
    # sample count so trivial frame-count drifts don't error.
    max_ch = max(a.shape[1] for a in arrays)
    min_n = min(a.shape[0] for a in arrays)
    padded = []
    for a in arrays:
        a = a[:min_n]
        if a.shape[1] < max_ch:
            a = np.tile(a, (1, max_ch // a.shape[1]))
        padded.append(a)
    return np.sum(padded, axis=0).astype(np.float32), sr


def _write_wav(out_path: Path, samples: np.ndarray, sr: int) -> bool:
    import soundfile as sf
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), samples, sr, subtype="PCM_16")
    except (OSError, RuntimeError):
        return False
    return True


@dataclass(frozen=True)
class BackfillReport:
    scanned: int
    created: int
    skipped: int
    failed: int


def backfill_track_instrumentals(db_path: Path) -> Result[BackfillReport, DbError]:
    """Backfill `track_stems` instrumental rows for tracks with all three
    component stems present on disk.
    """
    try:
        with _connect(db_path) as conn:
            # Candidate track_audio_ids: have all three component stems
            # but no 'instrumental' row.
            rows = conn.execute(
                """
                SELECT ta.track_audio_id, ta.path,
                       (SELECT path FROM track_stems WHERE track_audio_id = ta.track_audio_id AND stem_name = 'drums')  AS drums,
                       (SELECT path FROM track_stems WHERE track_audio_id = ta.track_audio_id AND stem_name = 'bass')   AS bass,
                       (SELECT path FROM track_stems WHERE track_audio_id = ta.track_audio_id AND stem_name = 'other')  AS other,
                       (SELECT 1    FROM track_stems WHERE track_audio_id = ta.track_audio_id AND stem_name = 'instrumental') AS has_instr
                FROM track_audio ta
                """
            ).fetchall()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="query_failed", detail=str(e)))

    scanned = created = skipped = failed = 0
    for r in rows:
        scanned += 1
        if r["has_instr"]:
            skipped += 1
            continue
        if not (r["drums"] and r["bass"] and r["other"]):
            skipped += 1
            continue
        summed = _sum_stems((Path(r["drums"]), Path(r["bass"]), Path(r["other"])))
        if summed is None:
            failed += 1
            continue
        samples, sr = summed
        # Place the instrumental alongside the other stem files.
        drums_path = Path(r["drums"])
        out_path = drums_path.with_name("instrumental.wav")
        if not _write_wav(out_path, samples, sr):
            failed += 1
            continue
        try:
            with _connect(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO track_stems (track_audio_id, stem_name, path, codec)
                    VALUES (?, 'instrumental', ?, 'wav')
                    ON CONFLICT(track_audio_id, stem_name) DO UPDATE SET
                        path  = excluded.path,
                        codec = excluded.codec
                    """,
                    (r["track_audio_id"], str(out_path)),
                )
                conn.commit()
        except sqlite3.DatabaseError:
            failed += 1
            continue
        created += 1
    return Ok(BackfillReport(scanned=scanned, created=created, skipped=skipped, failed=failed))


def backfill_set_instrumentals(db_path: Path) -> Result[BackfillReport, DbError]:
    """Same backfill for set-mix stems (`set_stems`)."""
    try:
        with _connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT sa.set_audio_id,
                       (SELECT path FROM set_stems WHERE set_audio_id = sa.set_audio_id AND stem_name = 'drums')  AS drums,
                       (SELECT path FROM set_stems WHERE set_audio_id = sa.set_audio_id AND stem_name = 'bass')   AS bass,
                       (SELECT path FROM set_stems WHERE set_audio_id = sa.set_audio_id AND stem_name = 'other')  AS other,
                       (SELECT 1    FROM set_stems WHERE set_audio_id = sa.set_audio_id AND stem_name = 'instrumental') AS has_instr
                FROM set_audio sa
                """
            ).fetchall()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="query_failed", detail=str(e)))

    scanned = created = skipped = failed = 0
    for r in rows:
        scanned += 1
        if r["has_instr"]:
            skipped += 1
            continue
        if not (r["drums"] and r["bass"] and r["other"]):
            skipped += 1
            continue
        summed = _sum_stems((Path(r["drums"]), Path(r["bass"]), Path(r["other"])))
        if summed is None:
            failed += 1
            continue
        samples, sr = summed
        drums_path = Path(r["drums"])
        out_path = drums_path.with_name("instrumental.wav")
        if not _write_wav(out_path, samples, sr):
            failed += 1
            continue
        try:
            with _connect(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO set_stems (set_audio_id, stem_name, path, codec)
                    VALUES (?, 'instrumental', ?, 'wav')
                    ON CONFLICT(set_audio_id, stem_name) DO UPDATE SET
                        path  = excluded.path,
                        codec = excluded.codec
                    """,
                    (r["set_audio_id"], str(out_path)),
                )
                conn.commit()
        except sqlite3.DatabaseError:
            failed += 1
            continue
        created += 1
    return Ok(BackfillReport(scanned=scanned, created=created, skipped=skipped, failed=failed))
