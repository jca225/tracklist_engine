"""Targeted Spotify-only retry pass over tracks the main downloader couldn't fetch.

The main downloader (`audio_pipeline.main`) walks YouTube → SoundCloud per track.
spotdl was previously the second link in that chain but was removed (commit
d64dc96) after a 14h corpus run produced 0 successes and 174× 300s timeouts on
anonymous Spotify queries.

This entrypoint is the surgical inverse: only attempt spotdl, only on tracks
where (a) we have no `track_audio` row at all and (b) the scraper found a
`dj_set_track_media_links` row with `platform='spotify'`. Tighter 60s timeout
keeps the dead-link tax bounded — spotdl 4.x ships with built-in default
Spotify API credentials, so no credential plumbing is needed here.

Usage (pi-storage, runs in parallel with main download loop):
    venvs/spotdl/bin/python -m audio_pipeline.main_retry \\
        --db /mnt/storage/data/db/music_database.db \\
        --audio-root /mnt/storage \\
        --timeout 60

Smoke run on Big Bootie 10-15 only:
    venvs/spotdl/bin/python -m audio_pipeline.main_retry \\
        --db /mnt/storage/data/db/music_database.db \\
        --audio-root /mnt/storage \\
        --bb-only --dry-run
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

from .adapters import db as db_adapter
from .adapters import spotdl_adapter
from .adapters.downloader import DownloadConfig
from .errors import DbError, DownloadError
from .models import AudioAsset, MediaSource, spotify_track_url
from .result import Err, Ok, Result

_log = logging.getLogger("audio_pipeline.main_retry")

# Big Bootie 10-15 set IDs — mirrors `audio_pipeline.main`.
_BIG_BOOTIE_10_15: frozenset[str] = frozenset((
    "w1mgcjt", "2nvzlh2k", "1fsnxchk", "qj4v0wt", "1yl70ql1", "237tdqmk",
))


@dataclass(frozen=True)
class RetryCandidate:
    """One track that has a scraped Spotify URL but no track_audio row."""
    track_id: str
    player_id: str             # Spotify track id
    set_ids: tuple[str, ...]   # which sets this track appears in (for logging)


@dataclass(frozen=True)
class RunStats:
    candidates: int = 0
    downloaded: int = 0
    failed_timeout: int = 0
    failed_unavailable: int = 0
    failed_other: int = 0
    db_failed: int = 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--db", type=Path,
                   default=Path(os.environ.get("TRACKLIST_DB",
                                               "/mnt/storage/data/db/music_database.db")))
    p.add_argument("--audio-root", type=Path,
                   default=Path(os.environ.get("TRACKLIST_AUDIO_ROOT", "/mnt/storage")))
    p.add_argument("--bb-only", action="store_true",
                   help="Restrict to Big Bootie 10-15 set_ids only")
    p.add_argument("--set-ids", default=None,
                   help="Comma-separated set_ids to restrict the retry to")
    p.add_argument("--timeout", type=float, default=60.0,
                   help="Per-track spotdl timeout (default 60s)")
    p.add_argument("--max-tracks", type=int, default=None,
                   help="Stop after attempting N tracks (smoke testing)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print candidates only; do not invoke spotdl")
    p.add_argument("--audio-format", default="m4a")
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p.parse_args(argv)


def _load_candidates(
    db_path: Path,
    set_filter: frozenset[str] | None,
) -> Result[tuple[RetryCandidate, ...], DbError]:
    """Find tracks with a Spotify URL but no track_audio row.

    Joins:
      - dj_set_track_media_links (scraper output) restricted to platform='spotify'
      - LEFT JOIN track_audio so we can filter to NULL (no audio yet)
    Aggregates per (track_id, player_id) so we don't redundantly attempt the
    same Spotify URL across multiple sets.
    """
    where_set = ""
    params: tuple[object, ...] = ()
    if set_filter is not None:
        placeholders = ",".join("?" * len(set_filter))
        where_set = f"AND m.set_id IN ({placeholders})"
        params = tuple(set_filter)

    query = f"""
        SELECT
            m.track_id           AS track_id,
            m.player_id          AS player_id,
            GROUP_CONCAT(DISTINCT m.set_id) AS set_ids
        FROM dj_set_track_media_links m
        LEFT JOIN track_audio ta
               ON ta.track_id = m.track_id
        WHERE m.platform = 'spotify'
          AND m.track_id IS NOT NULL AND m.track_id != ''
          AND m.player_id IS NOT NULL AND m.player_id != ''
          AND ta.track_audio_id IS NULL
          {where_set}
        GROUP BY m.track_id, m.player_id
        ORDER BY m.track_id
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
    except sqlite3.DatabaseError as e:
        return Err(DbError(kind="query_failed", detail=str(e)))

    return Ok(tuple(
        RetryCandidate(
            track_id=r["track_id"],
            player_id=r["player_id"],
            set_ids=tuple((r["set_ids"] or "").split(",")),
        )
        for r in rows
    ))


def _attempt(
    candidate: RetryCandidate,
    db_path: Path,
    objects_root: Path,
    audio_format: str,
    timeout_s: float,
) -> tuple[str, str | None]:
    """Returns (status, detail). Status:
      'downloaded' | 'unavailable' | 'timeout' | 'other_fail' | 'db_failed'
    """
    out_dir = objects_root / candidate.track_id
    cfg = DownloadConfig(
        out_dir=out_dir, audio_format=audio_format, retries=1, cookies_path=None,
    )
    source = MediaSource(
        platform="spotify",
        player_id=candidate.player_id,
        url=spotify_track_url(candidate.player_id),
    )

    dl_r = spotdl_adapter.download_one(
        candidate.track_id, source, cfg, timeout_s=timeout_s,
    )
    match dl_r:
        case Err(err):
            if err.kind == "network" and "timeout" in (err.detail or "").lower():
                return ("timeout", err.detail)
            if err.kind == "unavailable":
                return ("unavailable", err.detail)
            return ("other_fail", f"{err.kind}: {(err.detail or '')[:200]}")
        case Ok(asset):
            ins_r = db_adapter.insert_audio(db_path, asset)
            match ins_r:
                case Err(e):
                    return ("db_failed", f"insert_audio: {e.detail}")
                case Ok(_):
                    return ("downloaded", asset.path)


def _run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    set_filter: frozenset[str] | None = None
    if args.bb_only and args.set_ids:
        _log.error("--bb-only and --set-ids are mutually exclusive")
        return 2
    if args.bb_only:
        set_filter = _BIG_BOOTIE_10_15
    elif args.set_ids:
        set_filter = frozenset(s.strip() for s in args.set_ids.split(",") if s.strip())

    cand_r = _load_candidates(args.db, set_filter)
    match cand_r:
        case Err(err):
            _log.error("load candidates failed: %s", err.detail)
            return 1
        case Ok(candidates):
            pass

    if not candidates:
        _log.info("no retry candidates (every spotify-linked track already has audio)")
        return 0

    _log.info(
        "candidates=%d, db=%s, audio_root=%s, timeout=%.0fs, dry_run=%s, set_filter=%s",
        len(candidates), args.db, args.audio_root, args.timeout, args.dry_run,
        ("bb-10-15" if args.bb_only else (args.set_ids or "all")),
    )

    objects_root = args.audio_root / "objects"
    stats = RunStats(candidates=len(candidates))
    t0 = time.monotonic()

    for i, c in enumerate(candidates, 1):
        if args.max_tracks is not None and i > args.max_tracks:
            _log.info("hit --max-tracks=%d, stopping", args.max_tracks)
            break

        if args.dry_run:
            _log.info("[%d/%d] DRY %s spotify:%s sets=%s",
                      i, len(candidates), c.track_id, c.player_id,
                      ",".join(c.set_ids[:3]))
            continue

        status, detail = _attempt(
            c, args.db, objects_root, args.audio_format, args.timeout,
        )
        if status == "downloaded":
            stats = replace(stats, downloaded=stats.downloaded + 1)
            _log.info("[%d/%d] OK    %s -> %s", i, len(candidates), c.track_id, detail)
        elif status == "timeout":
            stats = replace(stats, failed_timeout=stats.failed_timeout + 1)
            _log.warning("[%d/%d] TIMEOUT %s", i, len(candidates), c.track_id)
        elif status == "unavailable":
            stats = replace(stats, failed_unavailable=stats.failed_unavailable + 1)
            _log.info("[%d/%d] GONE  %s (no YT match for spotify track)",
                      i, len(candidates), c.track_id)
        elif status == "other_fail":
            stats = replace(stats, failed_other=stats.failed_other + 1)
            _log.warning("[%d/%d] FAIL  %s %s", i, len(candidates), c.track_id, detail)
        elif status == "db_failed":
            stats = replace(stats, db_failed=stats.db_failed + 1)
            _log.error("[%d/%d] DB    %s %s", i, len(candidates), c.track_id, detail)

    elapsed = time.monotonic() - t0
    _log.info(
        "DONE | candidates=%d downloaded=%d timeout=%d gone=%d other=%d db=%d in %.0fs",
        stats.candidates, stats.downloaded, stats.failed_timeout,
        stats.failed_unavailable, stats.failed_other, stats.db_failed, elapsed,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    return _run(_parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
