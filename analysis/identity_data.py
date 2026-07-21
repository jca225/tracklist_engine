"""Shared data loading helpers for the learned identity verifier.

This module decouples DB I/O from model training and evaluation so both scripts
(and future lab experiments) can load the same ``RecordingFeatures`` view of the
canonical database without duplication.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from analysis.identity_learned import EssentiaFeatures, RecordingFeatures


def _decode_mert(blob: bytes, dim: int) -> np.ndarray | None:
    """Decode a float16 MERT embedding BLOB."""
    if not blob or dim <= 0:
        return None
    try:
        return np.frombuffer(blob, dtype=np.float16).astype(np.float32)
    except ValueError:
        return None


def _decode_fingerprint(blob: bytes) -> np.ndarray | None:
    """Decode a chromaprint fingerprint BLOB (uint32 raw sub-fingerprints)."""
    if not blob:
        return None
    try:
        return np.frombuffer(blob, dtype=np.uint32)
    except ValueError:
        return None


def _mean_pool_mert(rows: list[tuple[bytes, int]]) -> np.ndarray | None:
    """Mean-pool a list of (mert_blob, dim) measure embeddings."""
    vectors: list[np.ndarray] = []
    for blob, dim in rows:
        v = _decode_mert(blob, dim)
        if v is None or not np.isfinite(v).all():
            continue
        vectors.append(v)
    if not vectors:
        return None
    # Guard against mixed dimensions within a track (should not happen for a
    # single track, but the DB is heterogeneous across tracks).
    target_shape = vectors[0].shape
    same_shape = [v for v in vectors if v.shape == target_shape]
    if not same_shape:
        return None
    return np.mean(same_shape, axis=0)


def _track_metadata_join_column(cur: sqlite3.Cursor) -> str:
    """Return the column name in track_metadata that maps to recording_id.

    During the identity-axes migration, track_metadata may still use the legacy
    ``track_id`` column while recording.recording_id is the new key. This helper
    lets the loader work against both schemas.
    """
    cur.execute("PRAGMA table_info(track_metadata)")
    columns = {row["name"] for row in cur.fetchall()}
    if "recording_id" in columns:
        return "recording_id"
    if "track_id" in columns:
        return "track_id"
    return "recording_id"  # fallback to new schema and let the join fail loudly


def _load_recordings(
    db_path: Path,
    recording_ids: set[str] | None = None,
    mert_stride: int = 1,
) -> dict[str, RecordingFeatures]:
    """Load one RecordingFeatures per recording_id from the DB.

    Picks the reference track_audio row if available, otherwise the most recent
    downloaded row. MERT embeddings are mean-pooled across measures.

    Args:
        recording_ids: Optional subset of recording IDs to load. When provided,
            only these recordings (and their features) are queried, which keeps
            memory bounded for lab/sandbox evals that do not need the full corpus.
        mert_stride: If > 1, load only every Nth MERT measure for each
            track_audio_id. This reduces DB I/O on hosts with slow storage at the
            cost of a coarser mean-pooled embedding. Default 1 loads all measures.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    tm_join_col = _track_metadata_join_column(cur)

    # Optional subset filter. Use a temp table so we don't blow past SQLite's
    # default 999 placeholder limit for large IN clauses.
    filter_join = ""
    if recording_ids is not None:
        cur.execute(
            "CREATE TEMP TABLE _load_recording_ids (recording_id TEXT PRIMARY KEY)"
        )
        cur.executemany(
            "INSERT INTO _load_recording_ids (recording_id) VALUES (?)",
            [(rid,) for rid in recording_ids],
        )
        filter_join = (
            "INNER JOIN _load_recording_ids lf ON lf.recording_id = r.recording_id"
        )

    # 1. recording metadata + best track_audio row
    cur.execute(
        f"""
        SELECT
            r.recording_id,
            r.work_id,
            r.version,
            r.stem,
            r.variant,
            COALESCE(tm.full_name, r.full_name) AS full_name,
            tm.artists_json,
            tm.title,
            ta.track_audio_id,
            ta.platform,
            ta.is_reference
        FROM recording r
        {filter_join}
        LEFT JOIN track_metadata tm ON tm.{tm_join_col} = r.recording_id
        LEFT JOIN track_audio ta ON ta.recording_id = r.recording_id
        ORDER BY r.recording_id, ta.is_reference DESC, ta.downloaded_at DESC
        """
    )

    rows_by_recording: dict[str, sqlite3.Row] = {}
    for row in cur.fetchall():
        rid = row["recording_id"]
        if rid in rows_by_recording:
            continue
        rows_by_recording[rid] = row

    if not rows_by_recording:
        conn.close()
        return {}

    # Collect track_audio_ids for bulk feature loading.
    taids = {
        r["track_audio_id"] for r in rows_by_recording.values() if r["track_audio_id"]
    }
    taid_placeholders = ",".join("?" * len(taids)) if taids else "NULL"

    # 2. Essentia features
    essentia_by_taid: dict[int, EssentiaFeatures] = {}
    if taids:
        cur.execute(
            f"""
            SELECT * FROM track_audio_features
            WHERE track_audio_id IN ({taid_placeholders})
            """,
            tuple(taids),
        )
        for row in cur.fetchall():
            essentia_by_taid[row["track_audio_id"]] = EssentiaFeatures(
                bpm=row["bpm"],
                key_pc=row["key_pc"],
                key_mode=row["key_mode"],
                energy=row["energy"],
                danceability=row["danceability"],
                instrumentalness=row["instrumentalness"],
                lufs=row["lufs"],
            )

    # 3. MERT mean-pooled embeddings per track_audio_id
    mert_by_taid: dict[int, np.ndarray] = {}
    if taids:
        mert_params = list(taids)
        mert_where = f"track_audio_id IN ({taid_placeholders})"
        if mert_stride > 1:
            mert_where += " AND measure_idx % ? = 0"
            mert_params.append(mert_stride)
        cur.execute(
            f"""
            SELECT track_audio_id, dim, embedding
            FROM track_mert_measures
            WHERE {mert_where}
            """,
            tuple(mert_params),
        )
        by_taid: dict[int, list[tuple[bytes, int]]] = {}
        for row in cur.fetchall():
            by_taid.setdefault(row["track_audio_id"], []).append(
                (row["embedding"], row["dim"])
            )
        for taid, rows in by_taid.items():
            pooled = _mean_pool_mert(rows)
            if pooled is not None:
                mert_by_taid[taid] = pooled

    # 4. Fingerprints per recording_id (prefer 'regular' stem)
    fp_by_recording: dict[str, np.ndarray] = {}
    fp_filter = ""
    if recording_ids is not None:
        fp_filter = "INNER JOIN _load_recording_ids lf ON lf.recording_id = track_fingerprints.recording_id"
    cur.execute(
        f"""
        SELECT track_fingerprints.recording_id, track_fingerprints.stem, track_fingerprints.fingerprint
        FROM track_fingerprints
        {fp_filter}
        """
    )
    for row in cur.fetchall():
        rid = row["recording_id"]
        stem = row["stem"]
        fp = _decode_fingerprint(row["fingerprint"])
        if fp is None:
            continue
        if rid not in fp_by_recording or stem == "regular":
            fp_by_recording[rid] = fp

    conn.close()

    # 5. Build RecordingFeatures
    out: dict[str, RecordingFeatures] = {}
    for rid, row in rows_by_recording.items():
        taid = row["track_audio_id"]
        artists: list[str] = []
        try:
            artists = json.loads(row["artists_json"] or "[]") or []
        except json.JSONDecodeError:
            pass
        artist = artists[0] if artists else ""
        title = row["title"] or ""
        full_name = row["full_name"] or f"{artist} - {title}".strip(" -")

        # Try to parse remixer from title (e.g. "Title (Artist Remix)")
        remixer = ""
        if "(" in full_name:
            parens = full_name[full_name.rfind("(") + 1 : full_name.rfind(")")]
            if "remix" in parens.lower():
                remixer = parens.replace("Remix", "").replace("remix", "").strip()

        out[rid] = RecordingFeatures(
            recording_id=rid,
            work_id=row["work_id"],
            artist=artist,
            title=title,
            version=row["version"] or "original",
            stem=row["stem"] or "regular",
            variant=row["variant"] or "regular",
            remixer=remixer,
            essentia=essentia_by_taid.get(taid),
            mert_mean=mert_by_taid.get(taid),
            fingerprint=fp_by_recording.get(rid),
        )

    return out
