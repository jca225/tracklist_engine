"""Manually replace a track's audio. Three input modes:

  1. By URL (YouTube, YouTube Music, Spotify) — runs yt-dlp or spotdl
  2. By local file (already-downloaded m4a/wav/mp3) — copy in place
  3. List failed tracks (from Mac analyze log + DB join)

Use cases:
  - Mac analyze loop logged `analyze_track failed` for a track (corrupt audio
    or beat detector failed). Manually replace with a fresh download.
  - You have a higher-quality m4a on your machine (purchased / ripped).
  - A YT Music search picked the wrong version; you know the correct
    YT/Spotify URL and want to override.

What it does:
  1. Look up the existing track_audio row by track_id (or track_audio_id).
  2. If old row exists, DELETE it via FK cascade (kills any partial
     track_analysis / track_stems / etc. — those will be recomputed).
  3. Acquire new audio: yt-dlp / spotdl / file copy as appropriate.
  4. Place the file under /mnt/storage/objects/<track_id>/<track_id>__
     <platform>__<player_id>.<ext> per project convention.
  5. Insert a fresh track_audio row.
  6. The Mac (or Vast) analyze loop picks up the new track_audio_id on
     its next iteration since track_analysis IS NULL.

Usage:

  # See what's been failing on the running Mac loop
  venvs/audio/bin/python -m scripts.replace_track_audio --list-failed

  # Replace by YT URL (script picks adapter from URL)
  venvs/audio/bin/python -m scripts.replace_track_audio \\
      --track-id 1xkb35cf \\
      --url 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'

  # Replace by Spotify URL (uses spotdl, requires creds)
  venvs/audio/bin/python -m scripts.replace_track_audio \\
      --track-id 1xkb35cf \\
      --url 'https://open.spotify.com/track/4uLU6hMCjMI75M1A2tKUQC'

  # Replace by local file you have on disk
  venvs/audio/bin/python -m scripts.replace_track_audio \\
      --track-id 1xkb35cf \\
      --file ~/Downloads/clean_studio_audio.m4a \\
      --player-id manual_v1

  # Replace by track_audio_id (when you have the failed taid from logs)
  venvs/audio/bin/python -m scripts.replace_track_audio \\
      --track-audio-id 4011 \\
      --url 'https://www.youtube.com/watch?v=...'
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from audio_pipeline.adapters import db as db_adapter
from audio_pipeline.adapters import spotdl_adapter, ytmusic_adapter
from audio_pipeline.adapters.downloader import DownloadConfig
from audio_pipeline.errors import DownloadError
from audio_pipeline.models import AudioAsset, MediaSource
from core.result import Err, Ok

_log = logging.getLogger("replace_track_audio")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _detect_url_kind(url: str) -> str:
    """Return one of: 'youtube', 'spotify', 'unknown'.
    YT Music URLs (music.youtube.com) are treated as 'youtube' since the
    video_id namespace is shared.
    """
    p = urllib.parse.urlparse(url)
    host = (p.netloc or "").lower()
    if "spotify.com" in host:
        return "spotify"
    if "youtube.com" in host or "youtu.be" in host:
        return "youtube"
    return "unknown"


def _yt_video_id(url: str) -> str | None:
    """Extract video_id from any youtube.com / youtu.be / music.youtube.com URL."""
    p = urllib.parse.urlparse(url)
    if p.netloc.endswith("youtu.be"):
        return p.path.lstrip("/") or None
    qs = urllib.parse.parse_qs(p.query)
    return qs.get("v", [None])[0]


def _spotify_track_id(url: str) -> str | None:
    """Extract track id from open.spotify.com/track/<id>."""
    m = re.search(r"/track/([A-Za-z0-9]+)", url)
    return m.group(1) if m else None


def _resolve_track_id_from_taid(db_path: Path, taid: int) -> str | None:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT track_id FROM track_audio WHERE track_audio_id = ?", (taid,),
        ).fetchone()
    return row[0] if row else None


@dataclass(frozen=True)
class FailedTrack:
    track_audio_id: int
    track_id: str
    title: str
    artists: str
    platform: str
    failure_reason: str       # e.g. 'inference: decodeAVFrame...'


def _list_failed_from_log(log_path: Path, db_path: Path) -> list[FailedTrack]:
    """Parse a Mac/Vast analyze log for `analyze_track failed` lines and
    enrich with track metadata from canonical DB.
    """
    if not log_path.is_file():
        return []
    fails: dict[int, str] = {}
    pat = re.compile(r"\[(\d+)\] analyze_track failed: (.+?)$")
    for line in log_path.read_text().splitlines():
        m = pat.search(line)
        if m:
            taid = int(m.group(1))
            reason = m.group(2)[:140]
            fails[taid] = reason  # keep last reason
    if not fails:
        return []

    placeholders = ",".join("?" * len(fails))
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT ta.track_audio_id, ta.track_id, ta.platform,
                   COALESCE(tm.title,'') AS title,
                   COALESCE(tm.artists_json,'') AS artists_json
            FROM track_audio ta
            LEFT JOIN track_metadata tm ON tm.track_id = ta.track_id
            WHERE ta.track_audio_id IN ({placeholders})
            """,
            tuple(fails.keys()),
        ).fetchall()

    out: list[FailedTrack] = []
    for r in rows:
        out.append(FailedTrack(
            track_audio_id=r["track_audio_id"],
            track_id=r["track_id"],
            platform=r["platform"],
            title=r["title"],
            artists=r["artists_json"],
            failure_reason=fails.get(r["track_audio_id"], ""),
        ))
    return sorted(out, key=lambda f: f.track_audio_id)


def _delete_old_row_if_exists(
    db_path: Path, audio_root: Path, track_audio_id: int,
) -> None:
    """Delete the existing track_audio row + cascade-delete its analysis,
    stems, features, MERT measures. Also unlink the on-disk audio file
    and the stems dir (they belong to the old track_audio_id)."""
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT path FROM track_audio WHERE track_audio_id = ?",
            (track_audio_id,),
        ).fetchone()
        if row is None:
            return
        old_path = row[0]
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("DELETE FROM track_audio WHERE track_audio_id = ?",
                     (track_audio_id,))
        conn.commit()
    if old_path:
        p = Path(old_path)
        if p.is_file():
            p.unlink()
            _log.info("deleted old audio file %s", p)
    stems_dir = audio_root / "stems" / str(track_audio_id)
    if stems_dir.exists():
        shutil.rmtree(stems_dir, ignore_errors=True)
        _log.info("deleted old stems dir %s", stems_dir)


def _place_file_in_canonical(
    src: Path, audio_root: Path, track_id: str, platform: str, player_id: str,
) -> Path:
    """Move/copy `src` to /mnt/storage/objects/<tid>/<tid>__<platform>__
    <player_id>.<ext>. Returns destination path."""
    ext = src.suffix.lstrip(".") or "m4a"
    dst_dir = audio_root / "objects" / track_id
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{track_id}__{platform}__{player_id}.{ext}"
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)
    return dst


def _replace_via_url(
    db_path: Path, audio_root: Path, track_id: str, url: str,
    track_audio_id: int | None,
) -> int:
    """Acquire by URL and insert. Returns the new track_audio_id."""
    kind = _detect_url_kind(url)
    if kind == "spotify":
        spid = _spotify_track_id(url)
        if not spid:
            _log.error("could not extract spotify track id from %s", url)
            return 1
        return _replace_via_spotdl(db_path, audio_root, track_id, spid, track_audio_id)
    if kind == "youtube":
        vid = _yt_video_id(url)
        if not vid:
            _log.error("could not extract youtube video id from %s", url)
            return 1
        return _replace_via_ytdlp(db_path, audio_root, track_id, vid, track_audio_id)
    _log.error("unrecognized URL kind for %s", url)
    return 1


def _replace_via_spotdl(
    db_path: Path, audio_root: Path, track_id: str, spotify_id: str,
    track_audio_id: int | None,
) -> int:
    objects_root = audio_root / "objects"
    out_dir = objects_root / track_id
    cfg = DownloadConfig(out_dir=out_dir, audio_format="m4a", retries=2,
                         cookies_path=None)
    source = MediaSource(
        platform="spotify",
        player_id=spotify_id,
        url=f"https://open.spotify.com/track/{spotify_id}",
    )
    _log.info("downloading via spotdl: spotify:%s", spotify_id)
    r = spotdl_adapter.download_one(track_id, source, cfg, timeout_s=120.0)
    match r:
        case Err(err):
            _log.error("spotdl failed: %s — %s", err.kind, err.detail)
            return 1
        case Ok(asset):
            pass
    if track_audio_id is not None:
        _delete_old_row_if_exists(db_path, audio_root, track_audio_id)
    return _insert_and_report(db_path, asset)


def _replace_via_ytdlp(
    db_path: Path, audio_root: Path, track_id: str, video_id: str,
    track_audio_id: int | None,
) -> int:
    objects_root = audio_root / "objects"
    out_dir = objects_root / track_id
    cfg = DownloadConfig(out_dir=out_dir, audio_format="m4a", retries=2,
                         cookies_path=None)
    _log.info("downloading via yt-dlp (YT Music namespace): video=%s", video_id)
    r = ytmusic_adapter._ytdlp_download(
        video_id, track_id, cfg, timeout_s=120.0,
    )
    match r:
        case Err(err):
            _log.error("yt-dlp failed: %s — %s", err.kind, err.detail)
            return 1
        case Ok(path):
            pass
    asset = AudioAsset(
        track_audio_id=None,
        track_id=track_id,
        platform="youtube_music",
        source_url=f"https://www.youtube.com/watch?v={video_id}",
        player_id=video_id,
        path=str(path),
        sha256=_sha256(path),
        duration_s=None,
        sample_rate=None,
        codec="m4a",
        bitrate_kbps=None,
    )
    if track_audio_id is not None:
        _delete_old_row_if_exists(db_path, audio_root, track_audio_id)
    return _insert_and_report(db_path, asset)


def _replace_via_file(
    db_path: Path, audio_root: Path, track_id: str, file_path: Path,
    player_id: str, track_audio_id: int | None,
) -> int:
    if not file_path.is_file():
        _log.error("file does not exist: %s", file_path)
        return 1
    dst = _place_file_in_canonical(
        file_path, audio_root, track_id, "manual", player_id,
    )
    asset = AudioAsset(
        track_audio_id=None,
        track_id=track_id,
        platform="manual",
        source_url=f"file://{file_path}",
        player_id=player_id,
        path=str(dst),
        sha256=_sha256(dst),
        duration_s=None,
        sample_rate=None,
        codec=dst.suffix.lstrip("."),
        bitrate_kbps=None,
    )
    if track_audio_id is not None:
        _delete_old_row_if_exists(db_path, audio_root, track_audio_id)
    return _insert_and_report(db_path, asset)


def _insert_and_report(db_path: Path, asset: AudioAsset) -> int:
    r = db_adapter.insert_audio(db_path, asset)
    match r:
        case Err(e):
            _log.error("insert_audio failed: %s", e.detail)
            return 1
        case Ok(new_taid):
            _log.info("inserted new track_audio row taid=%d  path=%s",
                      new_taid, asset.path)
            return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--db", type=Path,
                   default=Path(os.environ.get("TRACKLIST_DB",
                                               "/mnt/storage/data/db/music_database.db")))
    p.add_argument("--audio-root", type=Path,
                   default=Path(os.environ.get("TRACKLIST_AUDIO_ROOT", "/mnt/storage")))
    p.add_argument("--list-failed", action="store_true",
                   help="Print failed tracks from the Mac analyze log and exit.")
    p.add_argument("--log-path", type=Path,
                   default=REPO / "logs" / "mac_analyze.log",
                   help="Analyze log to scan for failed tids (with --list-failed).")

    p.add_argument("--track-id", default=None,
                   help="Canonical 1001tracklists track_id of the track to replace.")
    p.add_argument("--track-audio-id", type=int, default=None,
                   help="Direct track_audio_id (use this if you have the taid "
                        "from the analyze log; we'll look up the track_id).")

    p.add_argument("--url", default=None,
                   help="YouTube / YT Music / Spotify URL to fetch from.")
    p.add_argument("--file", type=Path, default=None,
                   help="Local audio file to copy as the new audio.")
    p.add_argument("--player-id", default=None,
                   help="Identifier for the new row (defaults to filename stem "
                        "for --file mode; auto-extracted from --url).")

    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p.parse_args(argv)


def _run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.list_failed:
        fails = _list_failed_from_log(args.log_path, args.db)
        if not fails:
            print(f"no failed tracks found in {args.log_path}")
            return 0
        print(f"{len(fails)} failed tracks in {args.log_path}:")
        print(f"{'taid':>6} {'track_id':>10} {'platform':>14} {'reason':<40} title")
        for f in fails:
            artists = f.artists.replace('"', "").strip("[]") if f.artists else ""
            print(f"{f.track_audio_id:>6} {f.track_id:>10} {f.platform:>14} "
                  f"{f.failure_reason[:38]:<40} {artists} - {f.title}"[:200])
        return 0

    # Validate inputs.
    if not (args.url or args.file):
        _log.error("must provide --url or --file")
        return 2
    if args.url and args.file:
        _log.error("--url and --file are mutually exclusive")
        return 2

    track_id = args.track_id
    if track_id is None and args.track_audio_id is not None:
        track_id = _resolve_track_id_from_taid(args.db, args.track_audio_id)
        if track_id is None:
            _log.error("track_audio_id %d not found in canonical DB",
                       args.track_audio_id)
            return 1
    if track_id is None:
        _log.error("must provide --track-id or --track-audio-id")
        return 2

    if args.url:
        return _replace_via_url(args.db, args.audio_root, track_id,
                                args.url, args.track_audio_id)
    # File mode
    pid = args.player_id or args.file.stem
    return _replace_via_file(
        args.db, args.audio_root, track_id, args.file, pid,
        args.track_audio_id,
    )


def main(argv: list[str] | None = None) -> int:
    return _run(_parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
