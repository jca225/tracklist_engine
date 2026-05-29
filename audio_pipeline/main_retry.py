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
from core.result import Err, Ok, Result

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


# spotdl bundles a single default Spotify app and writes it into
# ~/.config/spotdl/config.json on first run. That client_id is shared by every
# spotdl install on the planet, so it's perpetually rate-limited.
_SPOTDL_BUNDLED_DEFAULT_CLIENT_ID = "5f573c9620494bae87890c0f08a60293"


def _detect_creds_source(args: argparse.Namespace) -> str:
    """Best-effort: figure out which credentials spotdl will end up using.

    Returns one of: 'cli', 'env', 'config', 'spotdl_default', 'unknown'.
    Used only for logging / startup warning — does not affect behavior.
    """
    if args.client_id and args.client_secret:
        # CLI flags or env-var fallback already consolidated by argparse.
        return "cli_or_env"
    config_path = Path.home() / ".config" / "spotdl" / "config.json"
    if config_path.is_file():
        try:
            import json
            cfg = json.loads(config_path.read_text())
            cid = cfg.get("client_id")
            if cid and cid != _SPOTDL_BUNDLED_DEFAULT_CLIENT_ID:
                return "config"
            if cid == _SPOTDL_BUNDLED_DEFAULT_CLIENT_ID:
                return "spotdl_default"
        except (OSError, ValueError):
            return "unknown"
    return "spotdl_default"


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
    p.add_argument("--batch-size", type=int, default=1,
                   help="Number of URLs to send to one spotdl invocation. "
                        "Default 1 = sequential per-track adapter (back-compat). "
                        ">1 enables pooled mode: spotdl runs N URLs together "
                        "with internal --threads, amortizing Python startup + "
                        "spotipy auth across the batch. Right size for tier1 "
                        "corpus pass: 10-20.")
    p.add_argument("--threads", type=int, default=4,
                   help="spotdl --threads, only used when --batch-size > 1. "
                        "Default 4 matches spotdl's own default. Higher can "
                        "saturate yt-music search rate-limits.")
    p.add_argument("--client-id", default=os.environ.get("SPOTIFY_CLIENT_ID"),
                   help="Spotify Web API client ID. Falls back to "
                        "$SPOTIFY_CLIENT_ID. Required at production volume — "
                        "spotdl's built-in default creds are globally rate-"
                        "limited (24h backoff once tripped).")
    p.add_argument("--client-secret", default=os.environ.get("SPOTIFY_CLIENT_SECRET"),
                   help="Spotify Web API client secret. Falls back to "
                        "$SPOTIFY_CLIENT_SECRET.")
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
    client_id: str | None,
    client_secret: str | None,
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
        client_id=client_id, client_secret=client_secret,
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


def _run_batched(
    args: argparse.Namespace,
    candidates: tuple[RetryCandidate, ...],
    objects_root: Path,
    stats: RunStats,
) -> RunStats:
    """Pooled spotdl execution path. Groups candidates into batches of
    `--batch-size` and dispatches each batch to one spotdl invocation with
    internal `--threads`. Files come back as `{spotify_id}.{ext}` in a
    staging dir; the adapter moves them to canonical per-track locations
    and returns one BatchResult per input.

    Per-batch timeout scales with batch size so a stalled URL can't kill
    siblings: `(batch_size / threads) * args.timeout * 2`. The 2× safety
    factor covers spotdl startup + spotipy auth + ffmpeg post-processing.
    """
    bs = args.batch_size
    threads = args.threads
    batch_timeout = max(args.timeout * 2, (bs / threads) * args.timeout * 2)

    capped = candidates
    if args.max_tracks is not None:
        capped = candidates[: args.max_tracks]

    batches = [capped[i : i + bs] for i in range(0, len(capped), bs)]
    _log.info("pooled mode: %d batches of <=%d (threads=%d, batch_timeout=%.0fs)",
              len(batches), bs, threads, batch_timeout)

    seen = 0
    for bi, batch in enumerate(batches, 1):
        items = tuple(
            spotdl_adapter.BatchItem(
                track_id=c.track_id,
                source=MediaSource(
                    platform="spotify",
                    player_id=c.player_id,
                    url=spotify_track_url(c.player_id),
                ),
            )
            for c in batch
        )
        t_batch = time.monotonic()
        results = spotdl_adapter.download_batch(
            items, objects_root, args.audio_format,
            threads=threads, timeout_s=batch_timeout,
            client_id=args.client_id, client_secret=args.client_secret,
        )
        elapsed = time.monotonic() - t_batch

        # Per-result handling: insert AudioAsset on Ok, classify failures.
        ok_in_batch = 0
        for r in results:
            seen += 1
            tid = r.item.track_id
            match r.result:
                case Ok(asset):
                    ins_r = db_adapter.insert_audio(args.db, asset)
                    match ins_r:
                        case Err(e):
                            stats = replace(stats, db_failed=stats.db_failed + 1)
                            _log.error("[%d/%d] DB    %s insert_audio: %s",
                                       seen, len(capped), tid, e.detail)
                        case Ok(_):
                            stats = replace(stats, downloaded=stats.downloaded + 1)
                            ok_in_batch += 1
                            _log.info("[%d/%d] OK    %s -> %s",
                                      seen, len(capped), tid, asset.path)
                case Err(err):
                    if err.kind == "network" and "timeout" in (err.detail or "").lower():
                        stats = replace(stats, failed_timeout=stats.failed_timeout + 1)
                        _log.warning("[%d/%d] TIMEOUT %s (batch)",
                                     seen, len(capped), tid)
                    elif err.kind == "unavailable":
                        stats = replace(stats, failed_unavailable=stats.failed_unavailable + 1)
                        _log.info("[%d/%d] GONE  %s",
                                  seen, len(capped), tid)
                    else:
                        stats = replace(stats, failed_other=stats.failed_other + 1)
                        _log.warning("[%d/%d] FAIL  %s %s: %s",
                                     seen, len(capped), tid,
                                     err.kind, (err.detail or "")[:120])

        wall_per_track = elapsed / max(len(items), 1)
        _log.info(
            "batch %d/%d done: %d/%d ok in %.0fs (%.1fs/track wall)",
            bi, len(batches), ok_in_batch, len(items), elapsed, wall_per_track,
        )
    return stats


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

    creds_source = _detect_creds_source(args)
    _log.info(
        "candidates=%d, db=%s, audio_root=%s, timeout=%.0fs, dry_run=%s, set_filter=%s, creds=%s",
        len(candidates), args.db, args.audio_root, args.timeout, args.dry_run,
        ("bb-10-15" if args.bb_only else (args.set_ids or "all")),
        creds_source,
    )
    if creds_source == "spotdl_default" and not args.dry_run:
        _log.warning(
            "spotdl appears to be using its bundled default Spotify app creds "
            "(client_id 5f573c96...). Those are globally shared and rate-"
            "limited; expect 86400s backoff. Pass --client-id/--client-secret, "
            "set SPOTIFY_CLIENT_ID/SECRET, or edit ~/.config/spotdl/config.json. "
            "Also delete ~/.config/spotdl/.spotipy after changing creds — "
            "spotipy caches the OAuth token and will keep using the old app."
        )

    objects_root = args.audio_root / "objects"
    stats = RunStats(candidates=len(candidates))
    t0 = time.monotonic()

    if args.batch_size > 1 and not args.dry_run:
        stats = _run_batched(args, candidates, objects_root, stats)
    else:
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
                args.client_id, args.client_secret,
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
