"""Analysis-result persistence — writes TrackAnalysisResult / EssentiaFeatures /
SetAnalysisResult to the canonical DB.

Split out of the generic core.db adapter so core stays free of analysis-domain
types: these writers are the only DB functions that depend on analysis.models,
so they live in the analysis stage and import core.db for the connection
primitive.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from core.db import connect
from core.errors import DbError
from core.result import Err, Ok, Result

from .models import EssentiaFeatures, MeasureEmbedding, TrackAnalysisResult

if TYPE_CHECKING:
    from .set_analysis import SetAnalysisResult

log = logging.getLogger(__name__)

# --- forward-routing dual-write (cutover Decision 2, step 2) -----------------
#
# When enabled, a just-persisted feature is ALSO registered into the canonical
# content-addressed provenance store — so new analysis is canonical by
# construction. Three properties make this safe to land (and deploy) before it is
# ever turned on in production:
#   * ADDITIVE  — runs only AFTER the legacy write has committed; the legacy path
#                 above is byte-for-byte unchanged.
#   * FLAGGED   — off unless BOTH env vars are set (PROVENANCE_DUAL_WRITE truthy
#                 AND PROVENANCE_STORE_ROOT a real path). Unset ⇒ this returns
#                 before importing anything, so it cannot slow the legacy path.
#   * DEFENSIVE — the whole body is wrapped; ANY failure is logged and swallowed.
#                 A provenance-store problem can never fail or slow-fatally an
#                 analysis run.
# Turning it on in production is a separate, human step (set both env vars on the
# analysis workers, deploy, restart) — not done here.

_DUAL_WRITE_FLAG = "PROVENANCE_DUAL_WRITE"
_STORE_ROOT_ENV = "PROVENANCE_STORE_ROOT"
_MERT_MODEL_REFS = ("m-a-p/MERT-v1-95M", "layer=6")


def _dual_write_enabled() -> bool:
    return os.environ.get(_DUAL_WRITE_FLAG, "").strip().lower() in {"1", "true", "yes"}


def _serialize_measures(measures: "list[MeasureEmbedding]") -> bytes:
    """Deterministic, content-addressable blob of a track's MERT measure stack:
    one sorted-key JSON header line, then the raw per-measure embedding bytes in
    measure-index order (byte-stable for content addressing)."""
    ms = sorted(measures, key=lambda m: m.measure_idx)
    header = {
        "v": 1,
        "kind": "mert_features",
        "n_measures": len(ms),
        "measures": [
            {
                "idx": m.measure_idx,
                "start_s": m.start_s,
                "end_s": m.end_s,
                "dim": m.dim,
                "dtype": m.dtype,
            }
            for m in ms
        ],
    }
    body = b"".join(m.embedding_bytes for m in ms)
    return (
        json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
        + body
    )


def _maybe_dual_write_provenance(result: TrackAnalysisResult) -> None:
    """Additively register the just-persisted MERT features into the canonical
    provenance store. Fully guarded: returns silently unless explicitly enabled,
    and never raises (a provenance failure must not affect the analysis run)."""
    if not _dual_write_enabled() or not result.measures:
        return
    root = os.environ.get(_STORE_ROOT_ENV, "").strip()
    if not root:
        log.warning(
            "%s set but %s unset — skipping provenance dual-write",
            _DUAL_WRITE_FLAG,
            _STORE_ROOT_ENV,
        )
        return
    try:
        # Lazy imports: nothing here touches the legacy path when disabled.
        from core.provenance import (
            ArtifactStore,
            ProvenanceRepository,
            connect as prov_connect,
            register_computed_feature,
        )

        db_root = Path(root)
        db_root.mkdir(parents=True, exist_ok=True)
        conn = prov_connect(db_root)
        store = ArtifactStore(db_root, conn)
        repo = ProvenanceRepository(conn)
        try:
            commit = (
                subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
                .decode()
                .strip()
            )
        except Exception:
            commit = "unknown"

        art = register_computed_feature(
            "mert_features",
            _serialize_measures(list(result.measures)),
            media_type="application/octet-stream",
            analyzer_name="analysis.mert",
            analyzer_version=str(result.analyzer_versions.get("mert", "1")),
            store=store,
            repo=repo,
            model_refs=_MERT_MODEL_REFS,
            params={"source": "analysis.persist_analysis"},
            code_commit=commit,
            metadata={"track_audio_id": result.track_audio_id},
        )
        log.info(
            "provenance dual-write: track_audio_id=%s mert_features=%s",
            result.track_audio_id,
            art.content_sha256[:12],
        )
    except Exception as exc:  # never let provenance break analysis
        log.warning("provenance dual-write failed (ignored): %r", exc)


def persist_set_analysis(
    db_path: Path, result: "SetAnalysisResult"
) -> Result[None, DbError]:
    """Write `SetAnalysisResult` atomically to set_analysis + set_stems.

    Both tables are keyed by set_audio_id; the stems UNIQUE constraint is
    per (set_audio_id, stem_name) so re-running analysis overwrites the
    previous rows instead of leaving stale ones around.
    """
    sid = result.set_audio_id
    try:
        with connect(db_path) as conn:
            conn.execute("BEGIN")

            for stem in result.stems.stems:
                conn.execute(
                    """
                    INSERT INTO set_stems (set_audio_id, stem_name, path, codec)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(set_audio_id, stem_name) DO UPDATE SET
                        path  = excluded.path,
                        codec = excluded.codec
                    """,
                    (sid, stem.stem_name, stem.path, stem.codec),
                )

            conn.execute(
                """
                INSERT INTO set_analysis
                  (set_audio_id, beat_times_json, downbeat_times_json,
                   measure_times_json, analyzer_versions_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(set_audio_id) DO UPDATE SET
                    beat_times_json        = excluded.beat_times_json,
                    downbeat_times_json    = excluded.downbeat_times_json,
                    measure_times_json     = excluded.measure_times_json,
                    analyzer_versions_json = excluded.analyzer_versions_json,
                    analyzed_at            = CURRENT_TIMESTAMP
                """,
                (
                    sid,
                    json.dumps(list(result.beats.beat_times)),
                    json.dumps(list(result.beats.downbeat_times)),
                    json.dumps(list(result.beats.measure_times)),
                    json.dumps(result.analyzer_versions),
                ),
            )
            conn.commit()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="integrity", detail=str(e)))
    return Ok(None)


def persist_analysis(
    db_path: Path, result: TrackAnalysisResult
) -> Result[None, DbError]:
    """Write a `TrackAnalysisResult` atomically across 4 analysis tables.

    Upserts stems, track_analysis, and a bundled `audio_pipeline_v1` row in
    track_audio_features (bpm + lufs). Replaces all MERT section rows for
    the track_audio_id so re-running analysis leaves a consistent snapshot.
    """
    tid = result.track_audio_id
    try:
        with connect(db_path) as conn:
            conn.execute("BEGIN")

            for stem in result.stems.stems:
                conn.execute(
                    """
                    INSERT INTO track_stems (track_audio_id, stem_name, path, codec)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(track_audio_id, stem_name) DO UPDATE SET
                        path  = excluded.path,
                        codec = excluded.codec
                    """,
                    (tid, stem.stem_name, stem.path, stem.codec),
                )

            conn.execute(
                """
                INSERT INTO track_analysis
                  (track_audio_id, beat_times_json, downbeat_times_json,
                   measure_times_json, cue_points_json, analyzer_versions_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(track_audio_id) DO UPDATE SET
                    beat_times_json        = excluded.beat_times_json,
                    downbeat_times_json    = excluded.downbeat_times_json,
                    measure_times_json     = excluded.measure_times_json,
                    cue_points_json        = excluded.cue_points_json,
                    analyzer_versions_json = excluded.analyzer_versions_json,
                    analyzed_at            = CURRENT_TIMESTAMP
                """,
                (
                    tid,
                    json.dumps(list(result.beats.beat_times)),
                    json.dumps(list(result.beats.downbeat_times)),
                    json.dumps(list(result.beats.measure_times)),
                    json.dumps(list(result.cues.cue_times)),
                    json.dumps(result.analyzer_versions),
                ),
            )

            conn.execute(
                """
                INSERT INTO track_audio_features
                  (track_audio_id, source, bpm, lufs)
                VALUES (?, 'audio_pipeline_v1', ?, ?)
                ON CONFLICT(track_audio_id, source) DO UPDATE SET
                    bpm         = excluded.bpm,
                    lufs        = excluded.lufs,
                    analyzed_at = CURRENT_TIMESTAMP
                """,
                (tid, result.beats.bpm, result.loudness.integrated_lufs),
            )

            conn.execute(
                "DELETE FROM track_mert_measures WHERE track_audio_id = ?",
                (tid,),
            )
            for m in result.measures:
                conn.execute(
                    """
                    INSERT INTO track_mert_measures
                      (track_audio_id, measure_idx, start_s, end_s,
                       dim, dtype, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        m.track_audio_id,
                        m.measure_idx,
                        m.start_s,
                        m.end_s,
                        m.dim,
                        m.dtype,
                        m.embedding_bytes,
                    ),
                )

            if result.essentia is not None:
                _write_essentia_row(conn, result.essentia)

            conn.commit()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="integrity", detail=str(e)))
    # Additive forward-routing: only reached after the legacy write committed;
    # fully guarded + off by default (see _maybe_dual_write_provenance).
    _maybe_dual_write_provenance(result)
    return Ok(None)


# Map Essentia's KeyExtractor tonic strings to pitch class 0..11 (C=0).
_PITCH_CLASS: dict[str, int] = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}


def _key_pc(tonic: str) -> int | None:
    return _PITCH_CLASS.get(tonic)


def _write_essentia_row(conn: sqlite3.Connection, feat: EssentiaFeatures) -> None:
    """INSERT/UPDATE one essentia_v2 row using the caller's transaction."""
    instrumentalness = 1.0 - feat.voice_prob if feat.voice_prob is not None else None
    confidence: dict[str, object] = {
        "models_present": list(feat.models_present),
        "key_strength": feat.key_strength,
        "key_profile": feat.key_profile,
        "danceability_sp": feat.danceability_sp,
        "danceability_tf": feat.danceability_tf,
        "mood_happy": feat.mood_happy,
        "mood_aggressive": feat.mood_aggressive,
        "valence_raw_emomusic_1_9": feat.valence_raw,
        "arousal_raw_emomusic_1_9": feat.arousal_raw,
        "voice_prob": feat.voice_prob,
        "yamnet_raw": feat.yamnet_raw,
        "time_sig_assumption": "4/4 default, not measured",
    }
    conn.execute(
        """
        INSERT INTO track_audio_features
          (track_audio_id, source, key_pc, key_mode, key_strength, bpm,
           time_sig_num, time_sig_den,
           danceability, energy, valence,
           acousticness, instrumentalness, speechiness, liveness,
           confidence_json)
        VALUES (?, 'essentia_v2', ?, ?, ?, ?, 4, 4, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(track_audio_id, source) DO UPDATE SET
            key_pc           = excluded.key_pc,
            key_mode         = excluded.key_mode,
            key_strength     = excluded.key_strength,
            bpm              = excluded.bpm,
            danceability     = excluded.danceability,
            energy           = excluded.energy,
            valence          = excluded.valence,
            acousticness     = excluded.acousticness,
            instrumentalness = excluded.instrumentalness,
            speechiness      = excluded.speechiness,
            liveness         = excluded.liveness,
            confidence_json  = excluded.confidence_json,
            analyzed_at      = CURRENT_TIMESTAMP
        """,
        (
            feat.track_audio_id,
            _key_pc(feat.key_tonic),
            feat.key_mode,
            feat.key_strength,
            feat.bpm,
            feat.danceability_tf
            if feat.danceability_tf is not None
            else feat.danceability_sp / 3.0,
            feat.mood_aggressive,
            feat.valence,
            feat.mood_acoustic,
            instrumentalness,
            feat.speechiness,
            feat.liveness,
            json.dumps(confidence),
        ),
    )


def persist_mert_measures(
    db_path: Path,
    track_audio_id: int,
    measures: tuple[MeasureEmbedding, ...],
    mert_version: str,
) -> Result[None, DbError]:
    """Replace per-measure MERT rows and patch the `mert` key in analyzer_versions.

    Does not touch stems, beats, cues, or Essentia — for re-embedding after a
    model upgrade when `track_analysis` already exists.
    """
    try:
        with connect(db_path) as conn:
            conn.execute("BEGIN")
            conn.execute(
                "DELETE FROM track_mert_measures WHERE track_audio_id = ?",
                (track_audio_id,),
            )
            for m in measures:
                conn.execute(
                    """
                    INSERT INTO track_mert_measures
                      (track_audio_id, measure_idx, start_s, end_s,
                       dim, dtype, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        m.track_audio_id,
                        m.measure_idx,
                        m.start_s,
                        m.end_s,
                        m.dim,
                        m.dtype,
                        m.embedding_bytes,
                    ),
                )
            row = conn.execute(
                "SELECT analyzer_versions_json FROM track_analysis "
                "WHERE track_audio_id = ?",
                (track_audio_id,),
            ).fetchone()
            if row is not None:
                versions = json.loads(row[0]) if row[0] else {}
                versions["mert"] = mert_version
                conn.execute(
                    """
                    UPDATE track_analysis
                    SET analyzer_versions_json = ?, analyzed_at = CURRENT_TIMESTAMP
                    WHERE track_audio_id = ?
                    """,
                    (json.dumps(versions), track_audio_id),
                )
            conn.commit()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="integrity", detail=str(e)))
    return Ok(None)


def persist_set_mert_measures(
    db_path: Path,
    set_audio_id: int,
    measures: tuple[MeasureEmbedding, ...],
    mert_version: str,
) -> Result[None, DbError]:
    """Replace per-measure MERT rows on the mix side (P4 / 6b).

    Patches the ``mert`` key in ``set_analysis.analyzer_versions_json``.
    Requires ``scripts/migrations/migrate_set_mert_measures.sql`` on the target DB.
    """
    try:
        with connect(db_path) as conn:
            conn.execute("BEGIN")
            conn.execute(
                "DELETE FROM set_mert_measures WHERE set_audio_id = ?",
                (set_audio_id,),
            )
            for m in measures:
                conn.execute(
                    """
                    INSERT INTO set_mert_measures
                      (set_audio_id, measure_idx, start_s, end_s,
                       dim, dtype, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        set_audio_id,
                        m.measure_idx,
                        m.start_s,
                        m.end_s,
                        m.dim,
                        m.dtype,
                        m.embedding_bytes,
                    ),
                )
            row = conn.execute(
                "SELECT analyzer_versions_json FROM set_analysis "
                "WHERE set_audio_id = ?",
                (set_audio_id,),
            ).fetchone()
            if row is not None:
                versions = json.loads(row[0]) if row[0] else {}
                versions["mert"] = mert_version
                conn.execute(
                    """
                    UPDATE set_analysis
                    SET analyzer_versions_json = ?, analyzed_at = CURRENT_TIMESTAMP
                    WHERE set_audio_id = ?
                    """,
                    (json.dumps(versions), set_audio_id),
                )
            conn.commit()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="integrity", detail=str(e)))
    return Ok(None)


def persist_essentia_features(
    db_path: Path,
    feat: EssentiaFeatures,
) -> Result[None, DbError]:
    """Standalone writer for the `essentia_v2` row.

    `persist_analysis` writes this row inline as part of its own
    transaction; this function is the one-shot entry point for callers
    that already have an EssentiaFeatures and want only that row updated
    (e.g. backfilling Essentia values without re-running the full MIR
    pipeline).
    """
    try:
        with connect(db_path) as conn:
            _write_essentia_row(conn, feat)
            conn.commit()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="integrity", detail=str(e)))
    return Ok(None)
