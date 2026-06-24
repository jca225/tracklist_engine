"""Top-level downloader entrypoint.

Reads a JSON job file (data/djs/*.json — produced by e.g.
eda/queries/tier1_plus_bb.sql) and downloads every track in
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
- **Platform fallback chain** — YouTube → spotdl/Spotify → SoundCloud.
  We walk every available scraped source and stop at the first one that
  successfully produces audio. spotdl is preferred over SoundCloud for
  Spotify-linked tracks because it finds the official mastered release
  via Spotify metadata + YouTube Music search; SoundCloud uploads tend
  to be lower-quality DJ rips.

Usage:
    venvs/audio/bin/python -m ingest.main \\
        --job-file data/djs/tier1_plus_bb.json \\
        --db /mnt/storage/data/db/music_database.db \\
        --audio-root /mnt/storage

For a smoke run on Big Bootie 10-15 only:
    venvs/audio/bin/python -m ingest.main \\
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

from core import db as db_adapter
from .adapters import spotdl_adapter
from .adapters.downloader import DownloadConfig, download_one, download_set_mix
from .preflight import check_environment
from .errors import DbError, DownloadError
from core.models import AudioAsset, MediaSource, SetMediaLink, Track
from core.result import Err, Ok, Result

_log = logging.getLogger("ingest.main")

# Big Bootie 10-15 set IDs — used by --bb-only convenience flag.
_BIG_BOOTIE_10_15: frozenset[str] = frozenset(
    (
        "w1mgcjt",
        "2nvzlh2k",
        "1fsnxchk",
        "qj4v0wt",
        "1yl70ql1",
        "237tdqmk",
    )
)

# Platform fallback order for per-track downloads. We walk these in order
# and stop at the first one that successfully produces audio for a given
# track_id.
#
# spotify (via spotdl) was originally in this chain as the second step,
# intended as a recovery path for tracks where YouTube is gone but a
# Spotify URL was scraped. After 14 hours of corpus run we observed
# zero successful spotdl downloads while accumulating 174 spotdl
# timeouts (~14.5 hours of wall-clock waste). spotdl's anonymous YT
# Music search is too slow / rate-limited to be net-positive without
# proper Spotify API credentials and a tighter timeout.
#
# It's been removed from the active chain. Re-enable later as a
# targeted retry pass over `no_source` failures, ideally with
# SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET set and timeout ≤30s.
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
    needs_resource: int = 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--job-file",
        required=True,
        type=Path,
        help="JSON job file (array of {tracklist_id, ...} objects)",
    )
    p.add_argument(
        "--db",
        type=Path,
        default=Path(
            os.environ.get("TRACKLIST_DB", "/mnt/storage/data/db/music_database.db")
        ),
        help="SQLite DB path (default: pi-storage canonical)",
    )
    p.add_argument(
        "--audio-root",
        type=Path,
        default=Path(os.environ.get("TRACKLIST_AUDIO_ROOT", "/mnt/storage")),
        help="Audio storage root (default: /mnt/storage). "
        "Files land at <root>/objects/<track_id>/...",
    )
    p.add_argument(
        "--max-sets",
        type=int,
        default=None,
        help="Stop after processing N sets (smoke testing)",
    )
    p.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="Stop after downloading N tracks (smoke testing)",
    )
    p.add_argument(
        "--bb-only",
        action="store_true",
        help="Restrict to Big Bootie 10-15 set_ids only",
    )
    p.add_argument(
        "--with-mixes",
        action="store_true",
        help="Also download the full DJ mix audio per set into "
        "<audio_root>/sets/<set_id>/. Off by default to keep "
        "the per-track ref pipeline lean.",
    )
    p.add_argument(
        "--mixes-only",
        action="store_true",
        help="Skip per-track downloads entirely; only fetch the "
        "mix audio for each set. Implies --with-mixes.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded; do not invoke yt-dlp",
    )
    p.add_argument(
        "--audio-format",
        default="m4a",
        help="yt-dlp postprocessor output format (default m4a)",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Per-track yt-dlp retry count (default 3)",
    )
    p.add_argument(
        "--cookies",
        type=Path,
        default=Path(os.environ["TRACKLIST_YT_COOKIES"])
        if os.environ.get("TRACKLIST_YT_COOKIES")
        else None,
        help="Netscape cookies.txt for age-gated YouTube. Without this, "
        "~5-15%% of tracks fail with 'Sign in to confirm your age'. "
        "Export from your browser on Mac and scp to pi-storage:  "
        "yt-dlp --cookies-from-browser chrome --cookies /tmp/yt.txt "
        "--skip-download 'https://youtube.com'",
    )
    p.add_argument(
        "--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR")
    )
    p.add_argument(
        "--sticky-skip",
        action="store_true",
        help="Legacy skip: any track_audio row blocks re-download even when "
        "the reference file is missing on disk.",
    )
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


def _pick_sources(track: Track) -> tuple[MediaSource, ...]:
    """Ordered list of MediaSources to try as fallback chain.

    Returns every available scraped source in priority order
    (youtube → spotify → soundcloud), filtering out platforms the track
    wasn't scraped on. Empty tuple = nothing downloadable for this track.
    """
    out: list[MediaSource] = []
    for platform in _PLATFORM_PREFERENCE:
        src = track.source_for(platform)
        if src is not None:
            out.append(src)
    return tuple(out)


def _download_via_platform(
    source: MediaSource,
    track_id: str,
    cfg: DownloadConfig,
) -> Result[AudioAsset, DownloadError]:
    """Dispatch to the correct adapter based on source.platform."""
    if source.platform == "spotify":
        return spotdl_adapter.download_one(track_id, source, cfg)
    return download_one(track_id, source, cfg)


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
    cookies_path: Path | None = None,
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

    seen_r = db_adapter.already_downloaded_set(
        db_path, set_id, chosen.platform, chosen.url
    )
    match seen_r:
        case Err(err):
            return ("db_failed", f"already_downloaded_set: {err.detail}")
        case Ok(True):
            return ("skip_existing", None)
        case Ok(False):
            pass

    if dry_run:
        return ("dry_run", f"{chosen.platform} {chosen.url[:80]}")

    cfg = DownloadConfig(
        out_dir=out_dir,
        audio_format=audio_format,
        retries=retries,
        cookies_path=cookies_path,
    )
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
    cookies_path: Path | None = None,
    *,
    reverify: bool = False,
) -> tuple[str, str | None]:
    """Returns (status, detail). Status: 'downloaded' | 'skip_existing' |
    'no_source' | 'download_failed' | 'db_failed' | 'dry_run'.

    Walks the platform fallback chain (youtube → spotify → soundcloud),
    trying each scraped source until one succeeds. Skips entirely if any
    track_audio row already exists for this track_id (regardless of which
    platform produced it — we don't redundantly download from a second
    source if we already have audio).
    """
    sources = _pick_sources(track)
    if not sources:
        return ("no_source", None)

    from ingest.identity_gate import should_skip_existing

    skip, verify = should_skip_existing(db_path, track.track_id, reverify=reverify)
    if skip:
        return ("skip_existing", verify.detail)

    if dry_run:
        chain = " → ".join(f"{s.platform}:{s.player_id}" for s in sources)
        return ("dry_run", chain)

    cfg = DownloadConfig(
        out_dir=out_dir,
        audio_format=audio_format,
        retries=retries,
        cookies_path=cookies_path,
    )
    last_err: DownloadError | None = None
    tried: list[str] = []
    for source in sources:
        tried.append(source.platform)
        dl_r = _download_via_platform(source, track.track_id, cfg)
        match dl_r:
            case Err(err):
                last_err = err
                _log.debug(
                    "        try %s failed: %s — falling back",
                    source.platform,
                    err.kind,
                )
                continue
            case Ok(asset):
                ins_r = db_adapter.insert_audio_or_reap(db_path, asset)
                match ins_r:
                    case Err(e):
                        return ("db_failed", f"insert_audio: {e.detail}")
                    case Ok(_):
                        return ("downloaded", f"[{'+'.join(tried)}] {asset.path}")
    detail = (
        f"all {len(sources)} platforms failed (tried {','.join(tried)}); "
        f"last={last_err.kind}: {last_err.detail[:160]}"
        if last_err
        else "no platforms tried"
    )
    return ("download_failed", detail)


def _run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    env = check_environment()
    if not env.ok:
        _log.warning("preflight: %s", env.detail)

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
    _log.info(
        "starting: %d sets, db=%s, audio_root=%s, dry_run=%s",
        len(set_ids),
        args.db,
        args.audio_root,
        args.dry_run,
    )

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
                args.db,
                set_id,
                mix_dir,
                args.audio_format,
                args.retries,
                args.dry_run,
                cookies_path=args.cookies,
            )
            if mix_status == "downloaded":
                _log.info(
                    "[%d/%d] set=%s MIX OK -> %s",
                    set_idx,
                    len(set_ids),
                    set_id,
                    mix_detail,
                )
            elif mix_status == "dry_run":
                _log.info(
                    "[%d/%d] set=%s MIX DRY %s",
                    set_idx,
                    len(set_ids),
                    set_id,
                    mix_detail,
                )
            elif mix_status == "skip_existing":
                _log.debug(
                    "[%d/%d] set=%s MIX skip (already downloaded)",
                    set_idx,
                    len(set_ids),
                    set_id,
                )
            elif mix_status == "no_source":
                _log.warning(
                    "[%d/%d] set=%s MIX no media link", set_idx, len(set_ids), set_id
                )
            elif mix_status == "download_failed":
                _log.warning(
                    "[%d/%d] set=%s MIX FAIL %s",
                    set_idx,
                    len(set_ids),
                    set_id,
                    mix_detail,
                )
            elif mix_status == "db_failed":
                _log.error(
                    "[%d/%d] set=%s MIX DB %s",
                    set_idx,
                    len(set_ids),
                    set_id,
                    mix_detail,
                )

        if skip_tracks:
            continue

        # --- Per-track refs ---
        tracks_r = db_adapter.load_set_tracks(args.db, set_id)
        match tracks_r:
            case Err(err):
                _log.warning(
                    "[%d/%d] set=%s load_tracks failed: %s",
                    set_idx,
                    len(set_ids),
                    set_id,
                    err.detail,
                )
                continue
            case Ok(tracks):
                pass

        _log.info(
            "[%d/%d] set=%s tracks=%d", set_idx, len(set_ids), set_id, len(tracks)
        )
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
                args.db,
                track,
                out_dir,
                args.audio_format,
                args.retries,
                args.dry_run,
                cookies_path=args.cookies,
                reverify=not args.sticky_skip,
            )
            if status == "downloaded":
                stats = RunStats(
                    **{**stats.__dict__, "downloaded": stats.downloaded + 1}
                )
                _log.info("    OK    %s -> %s", track.track_id, detail)
            elif status == "dry_run":
                _log.info("    DRY   %s would fetch %s", track.track_id, detail)
            elif status == "skip_existing":
                stats = RunStats(
                    **{
                        **stats.__dict__,
                        "skipped_already_downloaded": stats.skipped_already_downloaded
                        + 1,
                    }
                )
            elif status == "no_source":
                stats = RunStats(
                    **{**stats.__dict__, "failed_no_source": stats.failed_no_source + 1}
                )
                _log.debug("    SKIP  %s no YT/SC source", track.track_id)
            elif status == "download_failed":
                stats = RunStats(
                    **{**stats.__dict__, "failed_download": stats.failed_download + 1}
                )
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
        stats.sets_seen,
        stats.tracks_seen,
        stats.downloaded,
        stats.skipped_already_downloaded,
        stats.failed_no_source,
        stats.failed_download,
        elapsed_s,
    )


def main(argv: list[str] | None = None) -> int:
    return _run(_parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
