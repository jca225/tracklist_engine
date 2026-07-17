"""Manually replace a track's audio. Three input modes:

  1. By URL (YouTube, YouTube Music, SoundCloud, Spotify) — yt-dlp / spotdl
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
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import unicodedata
import urllib.parse
from dataclasses import dataclass, replace
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core.db import connect
from core import db as db_adapter
from ingest.adapters import spotdl_adapter, ytmusic_adapter
from ingest.adapters.downloader import DownloadConfig, download_one
from ingest.errors import DownloadError
from ingest.guards import duration_sane
from core.models import AudioAsset, MediaSource, soundcloud_api_url
from core.result import Err, Ok
from core.identity import DEFAULT_STEM, normalize_stem
from ingest.corrections import Correction, latest_row, log_correction, snapshot_row

_log = logging.getLogger("replace_track_audio")

_STEM_CHOICES = ("regular", "acappella", "instrumental")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _detect_url_kind(url: str) -> str:
    """Return one of: 'youtube', 'soundcloud', 'spotify', 'unknown'.

    YT Music URLs (music.youtube.com) are treated as 'youtube' since the
    video_id namespace is shared.
    """
    p = urllib.parse.urlparse(url)
    host = (p.netloc or "").lower()
    if "spotify.com" in host:
        return "spotify"
    if "soundcloud.com" in host:
        return "soundcloud"
    if "youtube.com" in host or "youtu.be" in host:
        return "youtube"
    return "unknown"


def _soundcloud_player_id(url: str) -> str | None:
    """Extract numeric track id from api.soundcloud.com or soundcloud.com URLs."""
    m = re.search(r"soundcloud\.com/tracks/(\d+)", url)
    return m.group(1) if m else None


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


def _resolve_stem_for_replace(
    stem_arg: str | None,
    old: dict | None,
) -> str:
    """CLI --stem, else retired row's stem, else regular."""
    if stem_arg is not None:
        return normalize_stem(stem_arg)
    if old and old.get("stem"):
        return normalize_stem(str(old["stem"]))
    return DEFAULT_STEM


def _resolve_axis_for_ledger(axis: str, stem: str) -> str:
    """Default --axis version; auto-use stem when replacing acappella/instrumental."""
    if axis != "version":
        return axis
    if stem in ("acappella", "instrumental"):
        return "stem"
    return axis


def _resolve_track_id_from_taid(db_path: Path, taid: int) -> str | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT track_id FROM track_audio WHERE track_audio_id = ?",
            (taid,),
        ).fetchone()
    return row[0] if row else None


@dataclass(frozen=True)
class FailedTrack:
    track_audio_id: int
    track_id: str
    title: str
    artists: str
    platform: str
    failure_reason: str  # e.g. 'inference: decodeAVFrame...'


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
    with connect(db_path) as conn:
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
        out.append(
            FailedTrack(
                track_audio_id=r["track_audio_id"],
                track_id=r["track_id"],
                platform=r["platform"],
                title=r["title"],
                artists=r["artists_json"],
                failure_reason=fails.get(r["track_audio_id"], ""),
            )
        )
    return sorted(out, key=lambda f: f.track_audio_id)


def _delete_old_row_if_exists(
    db_path: Path,
    audio_root: Path,
    track_audio_id: int,
    keep_path: str | None = None,
) -> None:
    """Delete the existing track_audio row + cascade-delete its analysis,
    stems, features, MERT measures. Also unlink the on-disk audio file
    and the stems dir (they belong to the old track_audio_id).

    `keep_path` guards the file unlink: if the old row's path equals the
    freshly-inserted asset's path (same track_id/platform/player_id
    collision), the file on disk now belongs to the NEW row — never
    delete it."""
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT path FROM track_audio WHERE track_audio_id = ?",
            (track_audio_id,),
        ).fetchone()
        if row is None:
            return
        old_path = row[0]
        conn.execute(
            "DELETE FROM track_audio WHERE track_audio_id = ?", (track_audio_id,)
        )
        conn.commit()
    if old_path and old_path != keep_path:
        p = Path(old_path)
        if p.is_file():
            p.unlink()
            _log.info("deleted old audio file %s", p)
    stems_dir = audio_root / "stems" / str(track_audio_id)
    if stems_dir.exists():
        shutil.rmtree(stems_dir, ignore_errors=True)
        _log.info("deleted old stems dir %s", stems_dir)


def _promote_reference(db_path: Path, track_id: str, track_audio_id: int) -> None:
    """Mark one row the reference for this track_id; demote all others."""
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE track_audio SET is_reference = 0 WHERE track_id = ?",
            (track_id,),
        )
        conn.execute(
            "UPDATE track_audio SET is_reference = 1 WHERE track_audio_id = ?",
            (track_audio_id,),
        )
        conn.commit()
    _log.info(
        "promoted taid=%d to is_reference for track_id=%s", track_audio_id, track_id
    )


def _purge_sibling_rows(
    db_path: Path,
    audio_root: Path,
    track_id: str,
    keep_taid: int,
) -> None:
    """Delete every other track_audio row for this track_id (+ on-disk files)."""
    with connect(db_path) as conn:
        siblings = conn.execute(
            "SELECT track_audio_id FROM track_audio "
            "WHERE track_id = ? AND track_audio_id != ?",
            (track_id, keep_taid),
        ).fetchall()
    for (taid,) in siblings:
        _delete_old_row_if_exists(db_path, audio_root, int(taid))
    if siblings:
        _log.info("purged %d sibling row(s) for track_id=%s", len(siblings), track_id)


_PLAYER_ID_MAXLEN = 96


def _ascii_player_id(pid: str) -> str:
    """Slugify `player_id` to pure ASCII so it cannot inject raw unicode into a
    canonical path.

    A player_id derived from a manual-ingest filename stem (e.g. a name
    containing U+FF5C `｜`) otherwise survives into
    ``<tid>__<platform>__<player_id>.<ext>``; that path string then
    mojibake-corrupts crossing the Mac(UTF-8)->pi(Latin-1) SSH/sqlite boundary
    (bug A1). Real platform ids (youtube/soundcloud/spotify) are already ASCII,
    so this is a no-op for them. Idempotent; falls back to a stable hash if no
    ASCII survives (e.g. an all-CJK name)."""
    if not pid:
        return pid
    # Transliterate accented Latin (é->e, Ü->U) then drop anything non-ASCII.
    norm = unicodedata.normalize("NFKD", pid).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", norm).strip("._-")
    if len(slug) > _PLAYER_ID_MAXLEN:
        slug = slug[:_PLAYER_ID_MAXLEN].strip("._-")
    if not slug:
        # Nothing ASCII survived — deterministic, path-safe fallback.
        slug = "id_" + hashlib.sha1(pid.encode("utf-8")).hexdigest()[:16]
    return slug


def _place_file_in_canonical(
    src: Path,
    audio_root: Path,
    track_id: str,
    platform: str,
    player_id: str,
) -> Path:
    """Move/copy `src` to /mnt/storage/objects/<tid>/<tid>__<platform>__
    <player_id>.<ext>. Returns destination path."""
    player_id = _ascii_player_id(player_id)
    ext = src.suffix.lstrip(".") or "m4a"
    dst_dir = audio_root / "objects" / track_id
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{track_id}__{platform}__{player_id}.{ext}"
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)
    return dst


def _replace_via_url(
    db_path: Path,
    audio_root: Path,
    track_id: str,
    url: str,
    track_audio_id: int | None,
    stem: str = "regular",
    variant: str = "regular",
    *,
    promote_reference: bool = True,
    purge_siblings: bool = False,
    check_duration: bool = True,
) -> int:
    """Acquire by URL and insert. Returns the new track_audio_id."""
    kind = _detect_url_kind(url)
    if kind == "spotify":
        spid = _spotify_track_id(url)
        if not spid:
            _log.error("could not extract spotify track id from %s", url)
            return 1
        return _replace_via_spotdl(
            db_path,
            audio_root,
            track_id,
            spid,
            track_audio_id,
            stem,
            variant,
            promote_reference=promote_reference,
            purge_siblings=purge_siblings,
        )
    if kind == "youtube":
        vid = _yt_video_id(url)
        if not vid:
            _log.error("could not extract youtube video id from %s", url)
            return 1
        return _replace_via_ytdlp(
            db_path,
            audio_root,
            track_id,
            vid,
            track_audio_id,
            stem,
            variant,
            promote_reference=promote_reference,
            purge_siblings=purge_siblings,
        )
    if kind == "soundcloud":
        scid = _soundcloud_player_id(url)
        if not scid:
            _log.error("could not extract soundcloud track id from %s", url)
            return 1
        return _replace_via_soundcloud(
            db_path,
            audio_root,
            track_id,
            scid,
            track_audio_id,
            stem,
            variant,
            promote_reference=promote_reference,
            purge_siblings=purge_siblings,
        )
    _log.error("unrecognized URL kind for %s", url)
    return 1


def _replace_via_soundcloud(
    db_path: Path,
    audio_root: Path,
    track_id: str,
    sc_player_id: str,
    track_audio_id: int | None,
    stem: str = "regular",
    variant: str = "regular",
    *,
    promote_reference: bool = True,
    purge_siblings: bool = False,
    check_duration: bool = True,
) -> int:
    """Download via yt-dlp from api.soundcloud.com/tracks/<id>."""
    objects_root = audio_root / "objects"
    out_dir = objects_root / track_id
    cfg = DownloadConfig(
        out_dir=out_dir, audio_format="m4a", retries=2, cookies_path=None
    )
    source = MediaSource(
        platform="soundcloud",
        player_id=sc_player_id,
        url=soundcloud_api_url(sc_player_id),
    )
    _log.info("downloading via yt-dlp: soundcloud:%s", sc_player_id)
    r = download_one(track_id, source, cfg)
    match r:
        case Err(err):
            _log.error("soundcloud download failed: %s — %s", err.kind, err.detail)
            return 1
        case Ok(asset):
            pass
    asset = replace(asset, stem=stem, variant=variant)
    return _insert_and_report(
        db_path,
        audio_root,
        asset,
        promote_reference=promote_reference,
        purge_siblings=purge_siblings,
        old_track_audio_id=track_audio_id,
        check_duration=check_duration,
    )


def _replace_via_spotdl(
    db_path: Path,
    audio_root: Path,
    track_id: str,
    spotify_id: str,
    track_audio_id: int | None,
    stem: str = "regular",
    variant: str = "regular",
    *,
    promote_reference: bool = True,
    purge_siblings: bool = False,
    check_duration: bool = True,
) -> int:
    objects_root = audio_root / "objects"
    out_dir = objects_root / track_id
    cfg = DownloadConfig(
        out_dir=out_dir, audio_format="m4a", retries=2, cookies_path=None
    )
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
    asset = replace(asset, stem=stem, variant=variant)
    return _insert_and_report(
        db_path,
        audio_root,
        asset,
        promote_reference=promote_reference,
        purge_siblings=purge_siblings,
        old_track_audio_id=track_audio_id,
        check_duration=check_duration,
    )


def _replace_via_ytdlp(
    db_path: Path,
    audio_root: Path,
    track_id: str,
    video_id: str,
    track_audio_id: int | None,
    stem: str = "regular",
    variant: str = "regular",
    *,
    promote_reference: bool = True,
    purge_siblings: bool = False,
    check_duration: bool = True,
) -> int:
    objects_root = audio_root / "objects"
    out_dir = objects_root / track_id
    cfg = DownloadConfig(
        out_dir=out_dir, audio_format="m4a", retries=2, cookies_path=None
    )
    _log.info("downloading via yt-dlp (YT Music namespace): video=%s", video_id)
    r = ytmusic_adapter._ytdlp_download(
        video_id,
        track_id,
        cfg,
        timeout_s=120.0,
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
        stem=stem,
        variant=variant,
    )
    return _insert_and_report(
        db_path,
        audio_root,
        asset,
        promote_reference=promote_reference,
        purge_siblings=purge_siblings,
        old_track_audio_id=track_audio_id,
        check_duration=check_duration,
    )


def _replace_via_file(
    db_path: Path,
    audio_root: Path,
    track_id: str,
    file_path: Path,
    player_id: str,
    track_audio_id: int | None,
    stem: str = "regular",
    variant: str = "regular",
    *,
    promote_reference: bool = True,
    purge_siblings: bool = False,
    check_duration: bool = True,
) -> int:
    if not file_path.is_file():
        _log.error("file does not exist: %s", file_path)
        return 1
    # Keep the stored player_id column consistent with the ASCII-sanitized path
    # (bug A1). Idempotent with the slug applied inside _place_file_in_canonical.
    player_id = _ascii_player_id(player_id)
    dst = _place_file_in_canonical(
        file_path,
        audio_root,
        track_id,
        "manual",
        player_id,
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
        stem=stem,
        variant=variant,
    )
    return _insert_and_report(
        db_path,
        audio_root,
        asset,
        promote_reference=promote_reference,
        purge_siblings=purge_siblings,
        old_track_audio_id=track_audio_id,
        check_duration=check_duration,
    )


def _probe_duration_s(path: str) -> float | None:
    """Measured duration of an audio file via ffprobe, or None if unavailable.
    Replace assets often carry duration_s=None (unlike the main downloader),
    so the duration gate has to measure the file itself."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        out = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return float(out.stdout.strip()) if out.stdout.strip() else None
    except (subprocess.SubprocessError, ValueError):
        return None


def _duration_gate(db_path: Path, asset: AudioAsset) -> str | None:
    """Reject the preview-clip / truncated-edit class for REGULAR full tracks
    only. Acappella/instrumental stems and extended variants have different
    length semantics (a vocal stab is legitimately short — the DJ Kool 8.5s
    case), so they are never gated here. Relative to the scraped song length,
    never an absolute floor. Returns a rejection detail, or None to allow."""
    if asset.stem != "regular" or asset.variant != "regular":
        return None
    listed_r = db_adapter.load_track_listed_duration(db_path, asset.track_id)
    listed = listed_r.value if isinstance(listed_r, Ok) else None
    actual = (
        asset.duration_s
        if asset.duration_s is not None
        else _probe_duration_s(asset.path)
    )
    if duration_sane(actual, listed):
        return None
    return f"got {actual or 0:.0f}s vs listed {listed}s"


def _insert_and_report(
    db_path: Path,
    audio_root: Path,
    asset: AudioAsset,
    *,
    promote_reference: bool,
    purge_siblings: bool,
    old_track_audio_id: int | None = None,
    check_duration: bool = True,
) -> int:
    # Duration gate BEFORE any insert/delete: a suspect regular track never
    # becomes canonical audio and never triggers the old-row delete. Skipped
    # for stems/extended and overridable with --no-duration-check.
    if check_duration:
        suspect = _duration_gate(db_path, asset)
        if suspect is not None:
            _log.error(
                "duration_suspect: %s - refusing (use --no-duration-check to "
                "override); old row untouched",
                suspect,
            )
            Path(asset.path).unlink(missing_ok=True)
            return 1
    # INSERT BEFORE DELETE: the old row is removed only after the new row is
    # committed. A failed/interrupted download or insert therefore leaves the
    # existing audio intact (the delete-before-insert ordering silently ate
    # ~23 references — e.g. DJ Kool - Let Me Clear My Throat's full original).
    r = db_adapter.insert_audio_or_reap(db_path, asset)
    match r:
        case Err(e):
            _log.error("insert_audio failed: %s — old row preserved", e.detail)
            return 1
        case Ok(new_taid):
            _log.info(
                "inserted new track_audio row taid=%d  path=%s", new_taid, asset.path
            )
            if old_track_audio_id is not None and old_track_audio_id != new_taid:
                _delete_old_row_if_exists(
                    db_path, audio_root, old_track_audio_id, keep_path=asset.path
                )
            if promote_reference:
                _promote_reference(db_path, asset.track_id, new_taid)
            if purge_siblings:
                _purge_sibling_rows(db_path, audio_root, asset.track_id, new_taid)
            return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--db",
        type=Path,
        default=Path(
            os.environ.get("TRACKLIST_DB", "/mnt/storage/data/db/music_database.db")
        ),
    )
    p.add_argument(
        "--audio-root",
        type=Path,
        default=Path(os.environ.get("TRACKLIST_AUDIO_ROOT", "/mnt/storage")),
    )
    p.add_argument(
        "--list-failed",
        action="store_true",
        help="Print failed tracks from the Mac analyze log and exit.",
    )
    p.add_argument(
        "--log-path",
        type=Path,
        default=REPO / "logs" / "mac_analyze.log",
        help="Analyze log to scan for failed tids (with --list-failed).",
    )

    p.add_argument(
        "--track-id",
        default=None,
        help="Canonical 1001tracklists track_id of the track to replace.",
    )
    p.add_argument(
        "--track-audio-id",
        type=int,
        default=None,
        help="Direct track_audio_id (use this if you have the taid "
        "from the analyze log; we'll look up the track_id).",
    )

    p.add_argument(
        "--url", default=None, help="YouTube / YT Music / Spotify URL to fetch from."
    )
    p.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Local audio file to copy as the new audio.",
    )
    p.add_argument(
        "--player-id",
        default=None,
        help="Identifier for the new row (defaults to filename stem "
        "for --file mode; auto-extracted from --url).",
    )

    p.add_argument(
        "--axis",
        default="version",
        choices=("version", "variant", "stem"),
        help="Which identity axis was wrong (correction ledger). "
        "Defaults to stem when replacing acappella/instrumental rows.",
    )
    p.add_argument(
        "--stem",
        default=None,
        choices=_STEM_CHOICES,
        help="Identity stem for the new row (default: inherit from "
        "--track-audio-id row, else regular).",
    )
    p.add_argument(
        "--edit",
        default="regular",
        choices=("regular", "extended"),
        help="Edit length (Variant axis) for the new row.",
    )
    p.add_argument(
        "--reason",
        default=None,
        help="Free-text why the old audio was wrong (correction ledger).",
    )
    p.add_argument(
        "--set-id",
        default=None,
        help="Set where the mistake was noticed (correction ledger).",
    )
    p.add_argument(
        "--position",
        default=None,
        help="Published section no. / slot label (correction ledger).",
    )
    p.add_argument(
        "--no-log", action="store_true", help="Skip writing the correction-ledger row."
    )
    p.add_argument(
        "--no-promote-reference",
        action="store_true",
        help="Do not set is_reference=1 on the new row (default: promote).",
    )
    p.add_argument(
        "--purge-siblings",
        action="store_true",
        help="Delete all other track_audio rows for this track_id "
        "(and their on-disk files/stems).",
    )
    p.add_argument(
        "--no-duration-check",
        action="store_true",
        help="Skip the regular-track duration gate (preview-clip / truncated "
        "-edit guard). Only applies to stem=regular, variant=regular anyway.",
    )

    p.add_argument(
        "--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR")
    )
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
            print(
                f"{f.track_audio_id:>6} {f.track_id:>10} {f.platform:>14} "
                f"{f.failure_reason[:38]:<40} {artists} - {f.title}"[:200]
            )
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
            _log.error(
                "track_audio_id %d not found in canonical DB", args.track_audio_id
            )
            return 1
    if track_id is None:
        _log.error("must provide --track-id or --track-audio-id")
        return 2

    # Snapshot the row we're about to retire BEFORE the destructive delete,
    # so the correction ledger keeps the old identity even after cascade.
    old = (
        snapshot_row(args.db, args.track_audio_id)
        if args.track_audio_id is not None
        else None
    )
    stem = _resolve_stem_for_replace(args.stem, old)
    ledger_axis = _resolve_axis_for_ledger(args.axis, stem)

    promote = not args.no_promote_reference
    if args.url:
        rc = _replace_via_url(
            args.db,
            args.audio_root,
            track_id,
            args.url,
            args.track_audio_id,
            stem=stem,
            variant=args.edit,
            promote_reference=promote,
            purge_siblings=args.purge_siblings,
            check_duration=not args.no_duration_check,
        )
    else:  # File mode
        pid = args.player_id or args.file.stem
        rc = _replace_via_file(
            args.db,
            args.audio_root,
            track_id,
            args.file,
            pid,
            args.track_audio_id,
            stem=stem,
            variant=args.edit,
            promote_reference=promote,
            purge_siblings=args.purge_siblings,
            check_duration=not args.no_duration_check,
        )

    if rc == 0 and not args.no_log:
        _log_to_ledger(args, track_id, old, ledger_axis=ledger_axis, stem=stem)
    if rc == 0:
        _emit_case_record(args, track_id, ledger_axis=ledger_axis, stem=stem)
    return rc


def _emit_case_record(
    args: argparse.Namespace, track_id: str, *, ledger_axis: str, stem: str
) -> None:
    """Print one ACQUISITION_CASE line for the Mac-side logger to persist.

    This runs on pi-storage, so it cannot write the Mac case corpus directly; it
    emits a record that ``scripts/log_acquisition.py --from-stdin`` consumes. No-op
    without --set-id + --position (the case is keyed per set/slot).
    """
    if not (args.set_id and args.position):
        return
    problem = (
        "wrong_version"
        if ledger_axis == "version"
        else (
            "suboptimal_stem"
            if stem in ("acappella", "instrumental")
            else "missing_asset"
        )
    )
    ref_source = "reference" if ledger_axis == "version" else "online_candidate"
    payload = {
        "set_id": args.set_id,
        "slot_label": str(args.position),
        "recording_id": track_id,
        "url": args.url or (f"file://{args.file}" if args.file else None),
        "reason": args.reason,
        "action": "ingest_url",
        "ref_source": ref_source,
        "verdict": "promote" if not args.no_promote_reference else "accept",
        "stem": stem,
        "problems": [problem],
    }
    print("ACQUISITION_CASE\t" + json.dumps(payload, ensure_ascii=False))


def _log_to_ledger(
    args: argparse.Namespace,
    track_id: str,
    old: dict | None,
    *,
    ledger_axis: str,
    stem: str,
) -> None:
    """Append a correction row for a successful replace/add (non-fatal)."""
    new = latest_row(args.db, track_id, stem)
    action = "replace" if args.track_audio_id is not None else "add"
    c = Correction(
        track_id=track_id,
        axis=ledger_axis,
        action=action,
        set_id=args.set_id,
        position=args.position,
        old_track_audio_id=args.track_audio_id,
        old_platform=(old or {}).get("platform"),
        old_player_id=(old or {}).get("player_id"),
        old_url=(old or {}).get("source_url"),
        new_track_audio_id=(new or {}).get("track_audio_id"),
        new_platform=(new or {}).get("platform"),
        new_player_id=(new or {}).get("player_id"),
        new_url=(new or {}).get("source_url"),
        stem_value=(new or {}).get("stem"),
        reason=args.reason,
        source="replace_track_audio",
    )
    match log_correction(args.db, c):
        case Ok(cid):
            _log.info("logged correction_id=%d (%s/%s)", cid, args.axis, action)
        case Err(e):
            _log.warning("correction log failed (non-fatal): %s — %s", e.kind, e.detail)


def main(argv: list[str] | None = None) -> int:
    return _run(_parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
