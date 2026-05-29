"""DB adapter — converts sqlite exceptions into DbError Results.

Domain code never imports sqlite3 directly; it calls these functions.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from core.errors import DbError
from core.models import (
    AudioAsset, MediaSource, SetAudioAsset, SetMediaLink, Track,
    normalize_set_media_url, soundcloud_api_url, spotify_track_url, youtube_url,
)
from core.result import Err, Ok, Result


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
                 duration_s, sample_rate, codec, bitrate_kbps, is_reference,
                 variant_tag)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                """,
                (
                    asset.track_id, asset.platform, asset.source_url, asset.player_id,
                    asset.path, asset.sha256, asset.duration_s, asset.sample_rate,
                    asset.codec, asset.bitrate_kbps, asset.variant_tag,
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
