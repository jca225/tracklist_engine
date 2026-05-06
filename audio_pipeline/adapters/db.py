"""DB adapter — converts sqlite exceptions into DbError Results.

Domain code never imports sqlite3 directly; it calls these functions.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from ..analysis.models import EssentiaFeatures, MeasureEmbedding, TrackAnalysisResult
from ..errors import DbError

# `set_analysis` transitively imports torch/demucs/MERT via pipeline.py, which
# we don't want to drag into the downloader-only path on pi-storage. Defer to
# the call site (persist_set_analysis below).
if TYPE_CHECKING:
    from ..analysis.set_analysis import SetAnalysisResult
from ..models import (
    AudioAsset, MediaSource, SetAudioAsset, SetMediaLink, SetTimeline,
    TimelineSegment, Track,
    normalize_set_media_url, soundcloud_api_url, spotify_track_url, youtube_url,
)
from ..result import Err, Ok, Result


@contextmanager
def _connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        # Default: enforce FKs (matches canonical DB invariants). Scratch DB
        # callers (e.g. Vast worker writing to /workspace/scratch.db before
        # shipping rows back to canonical) set TRACKLIST_DISABLE_FK=1 because
        # the scratch DB has no track_audio rows to satisfy the FK; the
        # actual integrity check happens when those rows hit canonical.
        if not os.environ.get("TRACKLIST_DISABLE_FK"):
            conn.execute("PRAGMA foreign_keys = ON;")
        yield conn
    finally:
        conn.close()


def _source_for(platform: str, player_id: str) -> MediaSource | None:
    if not player_id:
        return None
    match platform:
        case "youtube":
            return MediaSource(platform, player_id, youtube_url(player_id))
        case "soundcloud":
            return MediaSource(platform, player_id, soundcloud_api_url(player_id))
        case "spotify":
            return MediaSource(platform, player_id, spotify_track_url(player_id))
        case _:
            return None


def load_set_tracks(db_path: Path, set_id: str) -> Result[tuple[Track, ...], DbError]:
    """Load canonical tracks for one set_id, with all known media sources."""
    try:
        with _connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT track_id, tlp_id, platform, player_id
                FROM dj_set_track_media_links
                WHERE set_id = ?
                  AND track_id IS NOT NULL AND track_id != ''
                  AND player_id IS NOT NULL AND player_id != ''
                """,
                (set_id,),
            ).fetchall()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="query_failed", detail=str(e)))

    by_track: dict[str, dict[str, object]] = {}
    for r in rows:
        tid = r["track_id"]
        bucket = by_track.setdefault(tid, {"tlp_ids": set(), "sources": {}})
        if r["tlp_id"]:
            bucket["tlp_ids"].add(r["tlp_id"])  # type: ignore[union-attr]
        src = _source_for(r["platform"], r["player_id"])
        if src is not None:
            bucket["sources"][(src.platform, src.player_id)] = src  # type: ignore[index]

    tracks = tuple(
        Track(
            track_id=tid,
            tlp_ids=tuple(sorted(b["tlp_ids"])),  # type: ignore[arg-type]
            sources=tuple(b["sources"].values()),  # type: ignore[union-attr]
        )
        for tid, b in by_track.items()
    )
    return Ok(tracks)


def already_downloaded(
    db_path: Path, track_id: str, platform: str, player_id: str
) -> Result[bool, DbError]:
    try:
        with _connect(db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM track_audio WHERE track_id=? AND platform=? AND player_id=?",
                (track_id, platform, player_id),
            ).fetchone()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="query_failed", detail=str(e)))
    return Ok(row is not None)


def has_any_audio(db_path: Path, track_id: str) -> Result[bool, DbError]:
    """True if any track_audio row exists for this canonical track_id.

    Used by the fallback-chain downloader to skip a track entirely when
    we already have audio from any platform — the chain shouldn't attempt
    SoundCloud just because the YouTube row uses a different player_id.
    """
    try:
        with _connect(db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM track_audio WHERE track_id=? LIMIT 1",
                (track_id,),
            ).fetchone()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="query_failed", detail=str(e)))
    return Ok(row is not None)


def load_set_media_links(db_path: Path, set_id: str) -> Result[tuple[SetMediaLink, ...], DbError]:
    """Load all full-mix URLs posted for this set, normalized for yt-dlp.

    SoundCloud widget URLs are unwrapped into `api.soundcloud.com/tracks/<id>`.
    Mixcloud/hearthis/other platforms pass through as-is.
    """
    try:
        with _connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT platform, url
                FROM dj_set_media_links
                WHERE set_id = ? AND url IS NOT NULL AND url != ''
                """,
                (set_id,),
            ).fetchall()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="query_failed", detail=str(e)))
    seen: set[tuple[str, str]] = set()
    out: list[SetMediaLink] = []
    for r in rows:
        url = normalize_set_media_url(r["url"])
        key = (r["platform"], url)
        if key in seen:
            continue
        seen.add(key)
        out.append(SetMediaLink(set_id=set_id, platform=r["platform"], url=url))
    return Ok(tuple(out))


def already_downloaded_set(
    db_path: Path, set_id: str, platform: str, source_url: str
) -> Result[bool, DbError]:
    try:
        with _connect(db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM set_audio WHERE set_id=? AND platform=? AND source_url=?",
                (set_id, platform, source_url),
            ).fetchone()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="query_failed", detail=str(e)))
    return Ok(row is not None)


def insert_set_audio(db_path: Path, asset: SetAudioAsset) -> Result[int, DbError]:
    try:
        with _connect(db_path) as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO set_audio
                (set_id, platform, source_url, path, sha256,
                 duration_s, sample_rate, codec, bitrate_kbps, is_reference)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    asset.set_id, asset.platform, asset.source_url, asset.path,
                    asset.sha256, asset.duration_s, asset.sample_rate, asset.codec,
                    asset.bitrate_kbps,
                ),
            )
            conn.commit()
            if cur.lastrowid is None or cur.rowcount == 0:
                existing = conn.execute(
                    "SELECT set_audio_id FROM set_audio WHERE set_id=? AND platform=? AND source_url=?",
                    (asset.set_id, asset.platform, asset.source_url),
                ).fetchone()
                if existing is None:
                    return Err(DbError(kind="integrity", detail="set_audio insert returned no row"))
                return Ok(int(existing[0]))
            return Ok(int(cur.lastrowid))
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="integrity", detail=str(e)))


def upsert_timeline(db_path: Path, set_id: str, set_audio_id: int | None, payload_json: str) -> Result[None, DbError]:
    try:
        with _connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO set_timeline (set_id, set_audio_id, payload_json)
                VALUES (?, ?, ?)
                ON CONFLICT(set_id) DO UPDATE SET
                    set_audio_id = excluded.set_audio_id,
                    payload_json = excluded.payload_json,
                    built_at     = CURRENT_TIMESTAMP
                """,
                (set_id, set_audio_id, payload_json),
            )
            conn.commit()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="integrity", detail=str(e)))
    return Ok(None)


def insert_track_media_link(
    db_path: Path, *, set_id: str, track_id: str, platform: str,
    player_id: str, url: str | None = None, tlp_id: str | None = None,
) -> Result[None, DbError]:
    """Insert a manually-provided URL into dj_set_track_media_links.

    The scraper normally fills this table from AJAX responses; this function
    lets the UI add links for tracks the scraper couldn't resolve. The url
    column is informational — yt-dlp is driven by `_source_for` which
    reconstructs the canonical URL from platform + player_id.
    """
    try:
        with _connect(db_path) as conn:
            existing = conn.execute(
                """
                SELECT 1 FROM dj_set_track_media_links
                WHERE set_id = ? AND track_id = ? AND platform = ? AND player_id = ?
                """,
                (set_id, track_id, platform, player_id),
            ).fetchone()
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO dj_set_track_media_links
                      (set_id, tlp_id, track_id, platform, player_id, view_source)
                    VALUES (?, ?, ?, ?, ?, 'manual')
                    """,
                    (set_id, tlp_id, track_id, platform, player_id),
                )
                conn.commit()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="integrity", detail=str(e)))
    return Ok(None)


def load_set_timeline(db_path: Path, set_id: str) -> Result[SetTimeline, DbError]:
    """Read back the persisted timeline sidecar for a set."""
    try:
        with _connect(db_path) as conn:
            row = conn.execute(
                "SELECT set_audio_id, payload_json FROM set_timeline WHERE set_id = ?",
                (set_id,),
            ).fetchone()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="query_failed", detail=str(e)))
    if row is None:
        return Err(DbError(kind="not_found", detail=f"no set_timeline for {set_id!r}"))
    try:
        payload = json.loads(row["payload_json"])
    except (json.JSONDecodeError, TypeError) as e:
        return Err(DbError(kind="query_failed", detail=f"payload parse: {e}"))
    segments = tuple(
        TimelineSegment(
            row_index=int(s["row_index"]),
            track_id=s.get("track_id"),
            tlp_id=s.get("tlp_id"),
            title=s.get("title"),
            artists=tuple(s.get("artists") or ()),
            cue_seconds_section=s.get("cue_seconds_section"),
            is_ided=bool(s.get("is_ided")),
            is_concurrent=bool(s.get("is_concurrent")),
            is_remixish=bool(s.get("is_remixish")),
            has_yt=bool(s.get("has_yt")),
            has_sc=bool(s.get("has_sc")),
            has_sp=bool(s.get("has_sp")),
        )
        for s in payload.get("segments", ())
    )
    return Ok(SetTimeline(set_id=set_id, set_audio_id=row["set_audio_id"], segments=segments))


def load_set_measure_grid(db_path: Path, set_id: str) -> Result[list[float] | None, DbError]:
    """Return the set-mix downbeat-derived measure grid (seconds), or None
    if this set hasn't had `analyze_set` run on it yet."""
    try:
        with _connect(db_path) as conn:
            row = conn.execute(
                """
                SELECT san.measure_times_json
                FROM set_analysis san
                JOIN set_audio sa ON sa.set_audio_id = san.set_audio_id
                WHERE sa.set_id = ?
                ORDER BY sa.is_reference DESC, sa.downloaded_at DESC LIMIT 1
                """,
                (set_id,),
            ).fetchone()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="query_failed", detail=str(e)))
    if row is None or row["measure_times_json"] is None:
        return Ok(None)
    try:
        return Ok(list(json.loads(row["measure_times_json"])))
    except (json.JSONDecodeError, TypeError) as e:
        return Err(DbError(kind="query_failed", detail=f"measure_times parse: {e}"))


def load_ref_section_grids(
    db_path: Path, track_ids: tuple[str, ...]
) -> Result[dict[str, tuple[tuple[int, float, float], ...]], DbError]:
    """Load (section_idx, start_s, end_s) tuples per canonical track_id.

    Used by the CCC aligner's section-labelling step: given a matched
    ref-side span, find which cue-detr/MERT section(s) it falls inside.
    Tracks with no `track_mert_sections` rows are absent from the
    output dict (caller falls back to no section label).
    """
    if not track_ids:
        return Ok({})
    placeholders = ",".join("?" for _ in track_ids)
    try:
        with _connect(db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT ta.track_id, tms.section_idx, tms.start_s, tms.end_s
                FROM track_mert_sections tms
                JOIN track_audio ta ON ta.track_audio_id = tms.track_audio_id
                WHERE ta.track_id IN ({placeholders})
                ORDER BY ta.track_id, tms.section_idx
                """,
                track_ids,
            ).fetchall()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="query_failed", detail=str(e)))
    out: dict[str, list[tuple[int, float, float]]] = {}
    for r in rows:
        tid = r["track_id"]
        out.setdefault(tid, []).append(
            (int(r["section_idx"]), float(r["start_s"]), float(r["end_s"]))
        )
    return Ok({k: tuple(v) for k, v in out.items()})


def load_ref_measure_grids(db_path: Path, track_ids: tuple[str, ...]) -> Result[dict[str, list[float]], DbError]:
    """Map each canonical track_id to its measure grid (seconds). Tracks that
    haven't been analyzed are simply absent from the returned dict."""
    if not track_ids:
        return Ok({})
    placeholders = ",".join("?" for _ in track_ids)
    try:
        with _connect(db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT ta.track_id, tan.measure_times_json
                FROM track_analysis tan
                JOIN track_audio ta ON ta.track_audio_id = tan.track_audio_id
                WHERE ta.track_id IN ({placeholders})
                  AND tan.measure_times_json IS NOT NULL
                ORDER BY ta.is_reference DESC, ta.downloaded_at DESC
                """,
                track_ids,
            ).fetchall()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="query_failed", detail=str(e)))
    out: dict[str, list[float]] = {}
    for r in rows:
        if r["track_id"] in out:
            continue        # first (preferred) row per track wins
        try:
            out[r["track_id"]] = list(json.loads(r["measure_times_json"]))
        except (json.JSONDecodeError, TypeError):
            continue
    return Ok(out)


def load_set_audio_path(db_path: Path, set_id: str) -> Result[Path, DbError]:
    """Return the on-disk path of the preferred set_audio rip (is_reference > newest)."""
    try:
        with _connect(db_path) as conn:
            row = conn.execute(
                """
                SELECT path FROM set_audio WHERE set_id = ?
                ORDER BY is_reference DESC, downloaded_at DESC LIMIT 1
                """,
                (set_id,),
            ).fetchone()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="query_failed", detail=str(e)))
    if row is None:
        return Err(DbError(kind="not_found", detail=f"no set_audio for {set_id!r}"))
    return Ok(Path(row["path"]))


def load_ref_stem_paths(
    db_path: Path, track_ids: tuple[str, ...]
) -> Result[dict[str, dict[str, Path]], DbError]:
    """For each track_id, return {stem_name → on-disk stem path}. Tracks
    without demucs output are simply absent from the returned dict."""
    if not track_ids:
        return Ok({})
    placeholders = ",".join("?" for _ in track_ids)
    try:
        with _connect(db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT ta.track_id, ts.stem_name, ts.path
                FROM track_stems ts
                JOIN track_audio ta ON ta.track_audio_id = ts.track_audio_id
                WHERE ta.track_id IN ({placeholders})
                ORDER BY ta.is_reference DESC, ta.downloaded_at DESC
                """,
                track_ids,
            ).fetchall()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="query_failed", detail=str(e)))
    out: dict[str, dict[str, Path]] = {}
    seen: set[tuple[str, str]] = set()
    for r in rows:
        key = (r["track_id"], r["stem_name"])
        if key in seen:
            continue
        seen.add(key)
        out.setdefault(r["track_id"], {})[r["stem_name"]] = Path(r["path"])
    return Ok(out)


def load_set_stem_paths(
    db_path: Path, set_id: str
) -> Result[dict[str, Path] | None, DbError]:
    """Return {stem_name → on-disk stem path} for the set mix, or None if
    `analyze_set` hasn't run on this set yet."""
    try:
        with _connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT ss.stem_name, ss.path
                FROM set_stems ss
                JOIN set_audio sa ON sa.set_audio_id = ss.set_audio_id
                WHERE sa.set_id = ?
                ORDER BY sa.is_reference DESC, sa.downloaded_at DESC
                """,
                (set_id,),
            ).fetchall()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="query_failed", detail=str(e)))
    if not rows:
        return Ok(None)
    out: dict[str, Path] = {}
    for r in rows:
        out.setdefault(r["stem_name"], Path(r["path"]))
    return Ok(out)


def load_ref_audio_paths(db_path: Path, track_ids: tuple[str, ...]) -> Result[dict[str, Path], DbError]:
    """Map each canonical track_id to its preferred downloaded audio path."""
    if not track_ids:
        return Ok({})
    placeholders = ",".join("?" for _ in track_ids)
    try:
        with _connect(db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT track_id, path
                FROM track_audio
                WHERE track_id IN ({placeholders})
                ORDER BY is_reference DESC, downloaded_at DESC
                """,
                track_ids,
            ).fetchall()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="query_failed", detail=str(e)))
    out: dict[str, Path] = {}
    for r in rows:
        out.setdefault(r["track_id"], Path(r["path"]))    # first row per track wins
    return Ok(out)


def upsert_section_alignment(
    db_path: Path,
    *,
    set_id: str,
    section_idx: int,
    set_start_s: float,
    set_end_s: float,
    ref_track_id: str | None,
    transposition_semitones: int | None,
    bpm_ratio: float | None,
    cutup_plan_json: str | None,
    confidence: float | None,
    stem_match_rates_json: str | None = None,
    ref_start_s: float | None = None,
    ref_end_s: float | None = None,
    ref_section_idx: int | None = None,
) -> Result[None, DbError]:
    """Insert or replace one set_section_alignment row.

    `ref_start_s` / `ref_end_s` / `ref_section_idx` are CCC-aligner
    outputs (which bars/section of the reference were played). DTW
    leaves them NULL — its warping-path output doesn't answer the
    "which part of the ref" question cleanly enough to persist.
    """
    try:
        with _connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO set_section_alignment
                  (set_id, section_idx, set_start_s, set_end_s, ref_track_id,
                   transposition_semitones, bpm_ratio, cutup_plan_json, confidence,
                   stem_match_rates_json,
                   ref_start_s, ref_end_s, ref_section_idx)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(set_id, section_idx) DO UPDATE SET
                    set_start_s             = excluded.set_start_s,
                    set_end_s               = excluded.set_end_s,
                    ref_track_id            = excluded.ref_track_id,
                    transposition_semitones = excluded.transposition_semitones,
                    bpm_ratio               = excluded.bpm_ratio,
                    cutup_plan_json         = excluded.cutup_plan_json,
                    confidence              = excluded.confidence,
                    stem_match_rates_json   = excluded.stem_match_rates_json,
                    ref_start_s             = excluded.ref_start_s,
                    ref_end_s               = excluded.ref_end_s,
                    ref_section_idx         = excluded.ref_section_idx,
                    aligned_at              = CURRENT_TIMESTAMP
                """,
                (
                    set_id, section_idx, set_start_s, set_end_s, ref_track_id,
                    transposition_semitones, bpm_ratio, cutup_plan_json, confidence,
                    stem_match_rates_json,
                    ref_start_s, ref_end_s, ref_section_idx,
                ),
            )
            conn.commit()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="integrity", detail=str(e)))
    return Ok(None)


def persist_set_analysis(db_path: Path, result: "SetAnalysisResult") -> Result[None, DbError]:
    """Write `SetAnalysisResult` atomically to set_analysis + set_stems.

    Both tables are keyed by set_audio_id; the stems UNIQUE constraint is
    per (set_audio_id, stem_name) so re-running analysis overwrites the
    previous rows instead of leaving stale ones around.
    """
    sid = result.set_audio_id
    try:
        with _connect(db_path) as conn:
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


def persist_analysis(db_path: Path, result: TrackAnalysisResult) -> Result[None, DbError]:
    """Write a `TrackAnalysisResult` atomically across 4 analysis tables.

    Upserts stems, track_analysis, and a bundled `audio_pipeline_v1` row in
    track_audio_features (bpm + lufs). Replaces all MERT section rows for
    the track_audio_id so re-running analysis leaves a consistent snapshot.
    """
    tid = result.track_audio_id
    try:
        with _connect(db_path) as conn:
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
                        m.track_audio_id, m.measure_idx, m.start_s, m.end_s,
                        m.dim, m.dtype, m.embedding_bytes,
                    ),
                )

            if result.essentia is not None:
                _write_essentia_row(conn, result.essentia)

            conn.commit()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="integrity", detail=str(e)))
    return Ok(None)


# Map Essentia's KeyExtractor tonic strings to pitch class 0..11 (C=0).
_PITCH_CLASS: dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
    "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10, "B": 11,
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
          (track_audio_id, source, key_pc, key_mode, bpm,
           time_sig_num, time_sig_den,
           danceability, energy, valence,
           acousticness, instrumentalness, speechiness, liveness,
           confidence_json)
        VALUES (?, 'essentia_v2', ?, ?, ?, 4, 4, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(track_audio_id, source) DO UPDATE SET
            key_pc           = excluded.key_pc,
            key_mode         = excluded.key_mode,
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
            feat.bpm,
            feat.danceability_tf if feat.danceability_tf is not None
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


def persist_essentia_features(
    db_path: Path, feat: EssentiaFeatures,
) -> Result[None, DbError]:
    """Standalone writer for the `essentia_v2` row.

    `persist_analysis` writes this row inline as part of its own
    transaction; this function is the one-shot entry point for callers
    that already have an EssentiaFeatures and want only that row updated
    (e.g. backfilling Essentia values without re-running the full MIR
    pipeline).
    """
    try:
        with _connect(db_path) as conn:
            _write_essentia_row(conn, feat)
            conn.commit()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="integrity", detail=str(e)))
    return Ok(None)


def load_set_fingerprint_hits(
    db_path: Path, set_id: str, *, min_score: float = 0.6,
) -> Result[dict[str, list[tuple[float, float, float]]], DbError]:
    """Load per-track fingerprint hits for one set: {track_id → [(start, end, score), ...]}.

    Used by Stage-3's Viterbi window-narrowing to replace the
    cue-anchored heuristic window with a data-driven "where chroma
    fingerprint says this track actually plays" window. The threshold
    `min_score` gates out low-confidence hits up-front so the caller
    doesn't have to re-filter.
    """
    try:
        with _connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT matched_track_id, mix_start_s, mix_end_s, score
                FROM set_fingerprint_hits
                WHERE set_id = ? AND score >= ?
                ORDER BY matched_track_id, mix_start_s
                """,
                (set_id, float(min_score)),
            ).fetchall()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="query_failed", detail=str(e)))
    out: dict[str, list[tuple[float, float, float]]] = {}
    for r in rows:
        out.setdefault(r["matched_track_id"], []).append(
            (float(r["mix_start_s"]), float(r["mix_end_s"]), float(r["score"])),
        )
    return Ok(out)


def upsert_measure_alignment_rows(
    db_path: Path,
    set_id: str,
    rows: list[dict],
) -> Result[int, DbError]:
    """Bulk-upsert measure-level alignment decisions.

    Each `rows` entry is a dict with keys:
      set_measure_idx, ref_track_id, ref_measure_idx,
      pitch_shift_semi, tempo_ratio, stem_mask (list[str]),
      gain_db (optional), confidence (optional).

    Replaces the whole per-set set of rows; callers re-compute the
    alignment for a set end-to-end and write once.
    """
    import json as _json
    try:
        with _connect(db_path) as conn:
            conn.execute("BEGIN")
            conn.execute(
                "DELETE FROM measure_alignment WHERE set_id = ?",
                (set_id,),
            )
            conn.executemany(
                """
                INSERT INTO measure_alignment
                  (set_id, set_measure_idx, ref_track_id, ref_measure_idx,
                   pitch_shift_semi, tempo_ratio, stem_mask_json,
                   gain_db, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(set_id, set_measure_idx, ref_track_id) DO UPDATE SET
                    ref_measure_idx   = excluded.ref_measure_idx,
                    pitch_shift_semi  = excluded.pitch_shift_semi,
                    tempo_ratio       = excluded.tempo_ratio,
                    stem_mask_json    = excluded.stem_mask_json,
                    gain_db           = excluded.gain_db,
                    confidence        = excluded.confidence,
                    aligned_at        = CURRENT_TIMESTAMP
                """,
                [
                    (
                        set_id,
                        int(r["set_measure_idx"]),
                        r["ref_track_id"],
                        int(r["ref_measure_idx"]),
                        int(r.get("pitch_shift_semi", 0)),
                        float(r.get("tempo_ratio", 1.0)),
                        _json.dumps(list(r.get("stem_mask", ()))),
                        r.get("gain_db"),
                        r.get("confidence"),
                    )
                    for r in rows
                ],
            )
            conn.commit()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="integrity", detail=str(e)))
    return Ok(len(rows))


def load_track_measure_grid(
    db_path: Path, track_id: str,
) -> Result[list[tuple[int, float, float, float | None]], DbError]:
    """Return (measure_idx, start_s, end_s, bpm) for the preferred audio
    asset of `track_id`. Empty list if no measures persisted."""
    try:
        with _connect(db_path) as conn:
            row = conn.execute(
                """
                SELECT track_audio_id FROM track_audio
                WHERE track_id = ?
                ORDER BY is_reference DESC, downloaded_at DESC LIMIT 1
                """,
                (track_id,),
            ).fetchone()
            if row is None:
                return Ok([])
            measures = conn.execute(
                """
                SELECT measure_idx, start_s, end_s, bpm
                FROM track_measures
                WHERE track_audio_id = ?
                ORDER BY measure_idx
                """,
                (int(row[0]),),
            ).fetchall()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="query_failed", detail=str(e)))
    return Ok([
        (int(r["measure_idx"]), float(r["start_s"]), float(r["end_s"]),
         None if r["bpm"] is None else float(r["bpm"]))
        for r in measures
    ])


def load_set_measure_grid_full(
    db_path: Path, set_id: str,
) -> Result[list[tuple[int, float, float, float | None]], DbError]:
    try:
        with _connect(db_path) as conn:
            row = conn.execute(
                """
                SELECT set_audio_id FROM set_audio WHERE set_id = ?
                ORDER BY is_reference DESC, downloaded_at DESC LIMIT 1
                """,
                (set_id,),
            ).fetchone()
            if row is None:
                return Ok([])
            measures = conn.execute(
                """
                SELECT measure_idx, start_s, end_s, bpm
                FROM set_measures
                WHERE set_audio_id = ?
                ORDER BY measure_idx
                """,
                (int(row[0]),),
            ).fetchall()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="query_failed", detail=str(e)))
    return Ok([
        (int(r["measure_idx"]), float(r["start_s"]), float(r["end_s"]),
         None if r["bpm"] is None else float(r["bpm"]))
        for r in measures
    ])


def load_track_section_starts(
    db_path: Path, track_id: str,
) -> Result[frozenset[int], DbError]:
    """Return the set of ref-measure indices at which known sections
    begin for `track_id`. Used as structural priors in measure-DTW."""
    try:
        with _connect(db_path) as conn:
            row = conn.execute(
                """
                SELECT track_audio_id FROM track_audio
                WHERE track_id = ?
                ORDER BY is_reference DESC, downloaded_at DESC LIMIT 1
                """,
                (track_id,),
            ).fetchone()
            if row is None:
                return Ok(frozenset())
            sections = conn.execute(
                """
                SELECT start_s FROM track_sections
                WHERE track_audio_id = ?
                ORDER BY section_idx
                """,
                (int(row[0]),),
            ).fetchall()
            if not sections:
                return Ok(frozenset())
            measures = conn.execute(
                """
                SELECT measure_idx, start_s FROM track_measures
                WHERE track_audio_id = ?
                ORDER BY measure_idx
                """,
                (int(row[0]),),
            ).fetchall()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="query_failed", detail=str(e)))

    if not measures:
        return Ok(frozenset())

    # For each section start time, find the nearest measure start.
    import bisect
    m_starts = [float(m["start_s"]) for m in measures]
    m_idxs = [int(m["measure_idx"]) for m in measures]
    starts: set[int] = set()
    for s in sections:
        t = float(s["start_s"])
        pos = bisect.bisect_left(m_starts, t)
        if pos >= len(m_starts):
            pos = len(m_starts) - 1
        elif pos > 0 and abs(m_starts[pos] - t) > abs(m_starts[pos - 1] - t):
            pos = pos - 1
        starts.add(m_idxs[pos])
    return Ok(frozenset(starts))


def insert_audio(db_path: Path, asset: AudioAsset) -> Result[int, DbError]:
    try:
        with _connect(db_path) as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO track_audio
                (track_id, platform, source_url, player_id, path, sha256,
                 duration_s, sample_rate, codec, bitrate_kbps, is_reference)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    asset.track_id, asset.platform, asset.source_url, asset.player_id,
                    asset.path, asset.sha256, asset.duration_s, asset.sample_rate,
                    asset.codec, asset.bitrate_kbps,
                ),
            )
            conn.commit()
            if cur.lastrowid is None or cur.rowcount == 0:
                existing = conn.execute(
                    "SELECT track_audio_id FROM track_audio WHERE track_id=? AND platform=? AND player_id=?",
                    (asset.track_id, asset.platform, asset.player_id),
                ).fetchone()
                if existing is None:
                    return Err(DbError(kind="integrity", detail="insert returned no row and lookup failed"))
                return Ok(int(existing[0]))
            return Ok(int(cur.lastrowid))
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="integrity", detail=str(e)))
