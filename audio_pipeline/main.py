"""Top-level downloader entrypoint.

Reads a JSON job file (data/djs/*.json — produced by e.g.
data_analysis/queries/tier1_plus_bb.sql) and downloads every track in
every listed set. For each canonical track_id, joins
dj_set_track_media_links to find the YT/SC media URLs, runs yt-dlp via
the existing `download_one` adapter, and writes a `track_audio` row.

Designed for pi-storage long-running background runs:
- **Idempotent** — `already_downloaded()` skips tracks we already have
  (track_id, platform, player_id). Re-running is safe.
- **Per-track output dir** — `{audio_root}/objects/{track_id}/...` so the
  same canonical track is downloaded once even when it appears in many sets.
- **Resilient** — a single yt-dlp failure logs and moves on; one bad URL
  doesn't kill a 16k-track run.
- **YouTube preferred, SoundCloud fallback** — we try YT first; only fall
  back to SC if YT is missing or fails.

Usage:
    venvs/audio/bin/python -m audio_pipeline.main \\
        --job-file data/djs/tier1_plus_bb.json \\
        --db /mnt/storage/data/db/music_database.db \\
        --audio-root /mnt/storage

For a smoke run on Big Bootie 10-15 only:
    venvs/audio/bin/python -m audio_pipeline.main \\
        --job-file data/djs/tier1_plus_bb.json \\
        --db /mnt/storage/data/db/music_database.db \\
        --audio-root /mnt/storage \\
        --max-sets 6 --bb-only
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .adapters import db as db_adapter
from .adapters.downloader import DownloadConfig, download_one, download_set_mix
from .errors import DbError, DownloadError
from .models import AudioAsset, MediaSource, SetMediaLink, Track
from .result import Err, Ok, Result

_log = logging.getLogger("audio_pipeline.main")

# Big Bootie 10-15 set IDs — used by --bb-only convenience flag.
_BIG_BOOTIE_10_15: frozenset[str] = frozenset((
    "w1mgcjt", "2nvzlh2k", "1fsnxchk", "qj4v0wt", "1yl70ql1", "237tdqmk",
))

# Platform preference order. yt-dlp can resolve both reliably; YT has
# better coverage in our scrape (88% vs 4% in the tier-1 corpus).
_PLATFORM_PREFERENCE: tuple[str, ...] = ("youtube", "soundcloud")

# Mix-side preferences include Mixcloud since DJ sets often live there.
_MIX_PLATFORM_PREFERENCE: tuple[str, ...] = ("youtube", "soundcloud", "mixcloud")


@dataclass(frozen=True)
class RunStats:
    sets_seen: int = 0
    tracks_seen: int = 0
    skipped_already_downloaded: int = 0
    downloaded: int = 0
    failed_no_source: int = 0
    failed_download: int = 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--job-file", required=True, type=Path,
                   help="JSON job file (array of {tracklist_id, ...} objects)")
    p.add_argument("--db", type=Path,
                   default=Path(os.environ.get("TRACKLIST_DB",
                                               "/mnt/storage/data/db/music_database.db")),
                   help="SQLite DB path (default: pi-storage canonical)")
    p.add_argument("--audio-root", type=Path,
                   default=Path(os.environ.get("TRACKLIST_AUDIO_ROOT", "/mnt/storage")),
                   help="Audio storage root (default: /mnt/storage). "
                        "Files land at <root>/objects/<track_id>/...")
    p.add_argument("--max-sets", type=int, default=None,
                   help="Stop after processing N sets (smoke testing)")
    p.add_argument("--max-tracks", type=int, default=None,
                   help="Stop after downloading N tracks (smoke testing)")
    p.add_argument("--bb-only", action="store_true",
                   help="Restrict to Big Bootie 10-15 set_ids only")
    p.add_argument("--with-mixes", action="store_true",
                   help="Also download the full DJ mix audio per set into "
                        "<audio_root>/sets/<set_id>/. Off by default to keep "
                        "the per-track ref pipeline lean.")
    p.add_argument("--mixes-only", action="store_true",
                   help="Skip per-track downloads entirely; only fetch the "
                        "mix audio for each set. Implies --with-mixes.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be downloaded; do not invoke yt-dlp")
    p.add_argument("--audio-format", default="m4a",
                   help="yt-dlp postprocessor output format (default m4a)")
    p.add_argument("--retries", type=int, default=3,
                   help="Per-track yt-dlp retry count (default 3)")
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p.parse_args(argv)


def _load_job(job_file: Path) -> Result[tuple[str, ...], str]:
    """Read job file and return tuple of set_ids (1001tracklists ids)."""
    try:
        rows = json.loads(job_file.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return Err(f"failed to read {job_file}: {e}")
    if not isinstance(rows, list):
        return Err(f"{job_file} must contain a JSON array")
    set_ids: list[str] = []
    for r in rows:
        sid = r.get("tracklist_id") if isinstance(r, dict) else None
        if isinstance(sid, str) and sid:
            set_ids.append(sid)
    return Ok(tuple(set_ids))


def _pick_source(track: Track) -> MediaSource | None:
    """YouTube first, SoundCloud second. Returns None if neither is present."""
    for platform in _PLATFORM_PREFERENCE:
        src = track.source_for(platform)
        if src is not None:
            return src
    return None


def _pick_mix_link(links: tuple[SetMediaLink, ...]) -> SetMediaLink | None:
    """Choose the best mix-audio link: YT > SC > Mixcloud, first-match-wins."""
    for platform in _MIX_PLATFORM_PREFERENCE:
        for link in links:
            if link.platform == platform:
                return link
    return None


def _process_set_mix(
    db_path: Path,
    set_id: str,
    out_dir: Path,
    audio_format: str,
    retries: int,
    dry_run: bool,
) -> tuple[str, str | None]:
    """Returns (status, detail). Status: 'downloaded' | 'skip_existing' |
    'no_source' | 'download_failed' | 'db_failed' | 'dry_run'."""
    links_r = db_adapter.load_set_media_links(db_path, set_id)
    match links_r:
        case Err(err):
            return ("db_failed", f"load_set_media_links: {err.detail}")
        case Ok(links):
            pass

    chosen = _pick_mix_link(links)
    if chosen is None:
        return ("no_source", None)

    seen_r = db_adapter.already_downloaded_set(db_path, set_id, chosen.platform, chosen.url)
    match seen_r:
        case Err(err):
            return ("db_failed", f"already_downloaded_set: {err.detail}")
        case Ok(True):
            return ("skip_existing", None)
        case Ok(False):
            pass

    if dry_run:
        return ("dry_run", f"{chosen.platform} {chosen.url[:80]}")

    cfg = DownloadConfig(out_dir=out_dir, audio_format=audio_format, retries=retries)
    dl_r = download_set_mix(set_id, chosen.platform, chosen.url, cfg)
    match dl_r:
        case Err(err):
            return ("download_failed", f"{err.kind}: {err.detail[:200]}")
        case Ok(asset):
            ins_r = db_adapter.insert_set_audio(db_path, asset)
            match ins_r:
                case Err(e):
                    return ("db_failed", f"insert_set_audio: {e.detail}")
                case Ok(_):
                    return ("downloaded", asset.path)


def _process_track(
    db_path: Path,
    track: Track,
    out_dir: Path,
    audio_format: str,
    retries: int,
    dry_run: bool,
) -> tuple[str, str | None]:
    """Returns (status, detail). Status: 'downloaded' | 'skip_existing' |
    'no_source' | 'download_failed' | 'db_failed' | 'dry_run'."""
    source = _pick_source(track)
    if source is None:
        return ("no_source", None)

    seen_r = db_adapter.already_downloaded(db_path, track.track_id, source.platform, source.player_id)
    match seen_r:
        case Err(err):
            return ("db_failed", f"already_downloaded: {err.detail}")
        case Ok(True):
            return ("skip_existing", None)
        case Ok(False):
            pass

    if dry_run:
        return ("dry_run", f"{source.platform} {source.player_id}")

    cfg = DownloadConfig(out_dir=out_dir, audio_format=audio_format, retries=retries)
    dl_r = download_one(track.track_id, source, cfg)
    match dl_r:
        case Err(err):
            return ("download_failed", f"{err.kind}: {err.detail[:200]}")
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
    job_r = _load_job(args.job_file)
    match job_r:
        case Err(reason):
            _log.error(reason)
            return 1
        case Ok(set_ids):
            pass

    if args.bb_only:
        set_ids = tuple(s for s in set_ids if s in _BIG_BOOTIE_10_15)
    if args.max_sets is not None:
        set_ids = set_ids[: args.max_sets]
    if not set_ids:
        _log.error("no set_ids in job file after filters")
        return 1
    _log.info("starting: %d sets, db=%s, audio_root=%s, dry_run=%s",
              len(set_ids), args.db, args.audio_root, args.dry_run)

    objects_root = args.audio_root / "objects"
    sets_root = args.audio_root / "sets"
    with_mixes = args.with_mixes or args.mixes_only
    skip_tracks = args.mixes_only
    stats = RunStats()
    t0 = time.monotonic()

    for set_idx, set_id in enumerate(set_ids, 1):
        # --- Mix audio (full DJ mix) ---
        if with_mixes:
            mix_dir = sets_root / set_id
            mix_status, mix_detail = _process_set_mix(
                args.db, set_id, mix_dir, args.audio_format, args.retries, args.dry_run,
            )
            if mix_status == "downloaded":
                _log.info("[%d/%d] set=%s MIX OK -> %s", set_idx, len(set_ids), set_id, mix_detail)
            elif mix_status == "dry_run":
                _log.info("[%d/%d] set=%s MIX DRY %s", set_idx, len(set_ids), set_id, mix_detail)
            elif mix_status == "skip_existing":
                _log.debug("[%d/%d] set=%s MIX skip (already downloaded)", set_idx, len(set_ids), set_id)
            elif mix_status == "no_source":
                _log.warning("[%d/%d] set=%s MIX no media link", set_idx, len(set_ids), set_id)
            elif mix_status == "download_failed":
                _log.warning("[%d/%d] set=%s MIX FAIL %s", set_idx, len(set_ids), set_id, mix_detail)
            elif mix_status == "db_failed":
                _log.error("[%d/%d] set=%s MIX DB %s", set_idx, len(set_ids), set_id, mix_detail)

        if skip_tracks:
            continue

        # --- Per-track refs ---
        tracks_r = db_adapter.load_set_tracks(args.db, set_id)
        match tracks_r:
            case Err(err):
                _log.warning("[%d/%d] set=%s load_tracks failed: %s",
                             set_idx, len(set_ids), set_id, err.detail)
                continue
            case Ok(tracks):
                pass

        _log.info("[%d/%d] set=%s tracks=%d", set_idx, len(set_ids), set_id, len(tracks))
        stats = RunStats(
            sets_seen=stats.sets_seen + 1,
            tracks_seen=stats.tracks_seen,
            skipped_already_downloaded=stats.skipped_already_downloaded,
            downloaded=stats.downloaded,
            failed_no_source=stats.failed_no_source,
            failed_download=stats.failed_download,
        )

        for track in tracks:
            stats = RunStats(
                sets_seen=stats.sets_seen,
                tracks_seen=stats.tracks_seen + 1,
                skipped_already_downloaded=stats.skipped_already_downloaded,
                downloaded=stats.downloaded,
                failed_no_source=stats.failed_no_source,
                failed_download=stats.failed_download,
            )
            out_dir = objects_root / track.track_id
            status, detail = _process_track(
                args.db, track, out_dir, args.audio_format, args.retries, args.dry_run,
            )
            if status == "downloaded":
                stats = RunStats(**{**stats.__dict__, "downloaded": stats.downloaded + 1})
                _log.info("    OK    %s -> %s", track.track_id, detail)
            elif status == "dry_run":
                _log.info("    DRY   %s would fetch %s", track.track_id, detail)
            elif status == "skip_existing":
                stats = RunStats(**{**stats.__dict__,
                                    "skipped_already_downloaded": stats.skipped_already_downloaded + 1})
            elif status == "no_source":
                stats = RunStats(**{**stats.__dict__,
                                    "failed_no_source": stats.failed_no_source + 1})
                _log.debug("    SKIP  %s no YT/SC source", track.track_id)
            elif status == "download_failed":
                stats = RunStats(**{**stats.__dict__,
                                    "failed_download": stats.failed_download + 1})
                _log.warning("    FAIL  %s %s", track.track_id, detail)
            elif status == "db_failed":
                _log.error("    DB    %s %s", track.track_id, detail)

            if args.max_tracks is not None and stats.downloaded >= args.max_tracks:
                _log.info("hit --max-tracks=%d, stopping", args.max_tracks)
                _summarize(stats, time.monotonic() - t0)
                return 0

    _summarize(stats, time.monotonic() - t0)
    return 0


def _summarize(stats: RunStats, elapsed_s: float) -> None:
    _log.info(
        "DONE | sets=%d tracks_seen=%d downloaded=%d skipped=%d no_source=%d failed=%d in %.0fs",
        stats.sets_seen, stats.tracks_seen, stats.downloaded,
        stats.skipped_already_downloaded, stats.failed_no_source,
        stats.failed_download, elapsed_s,
    )


def main(argv: list[str] | None = None) -> int:
    return _run(_parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
