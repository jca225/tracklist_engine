"""YT Music adapter: query → search → top studio audio → yt-dlp download.

Why this exists alongside spotdl_adapter:
  spotdl works by Spotify-track-ID → ytmusicapi search → yt-dlp download.
  Two problems: (a) every call hits Spotify's Web API which has app-level
  rate limits we routinely trip; (b) we already know the artist + title
  from `track_metadata` (populated by web_crawler/tokenizer) — going
  through Spotify is an unnecessary hop.

  This adapter cuts Spotify out: ytmusicapi search seeded by our local
  track_metadata, then yt-dlp on the resulting videoId. Same audio
  source as spotdl (YT Music's "songs" filter — Topic-channel-style
  studio uploads), no Spotify rate limit.

Schema:
  Resulting `track_audio` rows use platform='youtube_music' so they're
  distinguishable from raw 1001tracklists YT scrapes (platform='youtube').
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..errors import DownloadError
from ..models import AudioAsset, MediaSource
from ..result import Err, Ok, Result
from .downloader import DownloadConfig


def _ytdlp_bin() -> str | None:
    """Find yt-dlp. Prefers <repo>/venvs/audio/bin (same env that runs the
    main downloader), falls back to PATH."""
    repo_root = Path(__file__).resolve().parents[2]
    audio_bin = repo_root / "venvs" / "audio" / "bin" / "yt-dlp"
    if audio_bin.is_file():
        return str(audio_bin)
    return shutil.which("yt-dlp")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class YTMSearchHit:
    """One YT Music search hit (filtered to `songs` resultType)."""
    video_id: str
    title: str
    artists: tuple[str, ...]
    duration_s: int | None         # None if ytmusicapi returned malformed duration


@dataclass(frozen=True)
class BatchItem:
    """One unit of work for `download_batch` — a canonical track_id paired
    with the search query string we pass to YT Music."""
    track_id: str
    query: str                     # "Artist - Title" or similar


@dataclass(frozen=True)
class BatchResult:
    item: BatchItem
    result: Result[AudioAsset, DownloadError]


def _parse_duration(s: str | None) -> int | None:
    """ytmusicapi returns durations like '4:02', '1:23:45', or None."""
    if not s:
        return None
    parts = s.split(":")
    try:
        ints = [int(p) for p in parts]
    except ValueError:
        return None
    if len(ints) == 2:
        return ints[0] * 60 + ints[1]
    if len(ints) == 3:
        return ints[0] * 3600 + ints[1] * 60 + ints[2]
    return None


# Module-level YTMusic singleton — initialization is cheap (~50 ms) but
# constructs an HTTP session each time, so reuse across calls.
_ytmusic_singleton = None


def _get_ytmusic():
    global _ytmusic_singleton
    if _ytmusic_singleton is None:
        from ytmusicapi import YTMusic
        _ytmusic_singleton = YTMusic()
    return _ytmusic_singleton


def search(query: str, limit: int = 5) -> Result[tuple[YTMSearchHit, ...], DownloadError]:
    """Search YT Music's 'songs' filter for `query` and return up to `limit`
    hits. Filter='songs' restricts to clean studio uploads (Topic channels
    + label-uploaded album tracks); excludes music videos, live recordings,
    and lyric uploads.
    """
    try:
        yt = _get_ytmusic()
        raw = yt.search(query, filter="songs", limit=limit)
    except Exception as e:
        return Err(DownloadError(
            kind="network", url=query,
            detail=f"ytmusicapi search failed: {type(e).__name__}: {e}"[:300],
        ))
    hits = []
    for r in raw[:limit]:
        vid = r.get("videoId")
        if not vid:
            continue
        artists = tuple(a.get("name", "") for a in (r.get("artists") or []))
        hits.append(YTMSearchHit(
            video_id=vid,
            title=r.get("title", ""),
            artists=artists,
            duration_s=_parse_duration(r.get("duration")),
        ))
    if not hits:
        return Err(DownloadError(
            kind="unavailable", url=query,
            detail=f"ytmusicapi returned no song hits for {query!r}",
        ))
    return Ok(tuple(hits))


def _ytdlp_download(
    video_id: str,
    track_id: str,
    cfg: DownloadConfig,
    timeout_s: float,
    cookies_path: Path | None = None,
) -> Result[Path, DownloadError]:
    """Run yt-dlp on https://www.youtube.com/watch?v=<video_id> and write
    audio to cfg.out_dir. Returns the path of the produced file."""
    bin_path = _ytdlp_bin()
    if bin_path is None:
        return Err(DownloadError(
            kind="parse", url=f"yt:{video_id}",
            detail="yt-dlp not on PATH",
        ))
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    # Write to canonical track-audio name directly to avoid post-download rename.
    dst = cfg.out_dir / f"{track_id}__youtube_music__{video_id}.{cfg.audio_format}"
    if dst.exists():
        dst.unlink()

    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        bin_path, url,
        "-x", "--audio-format", cfg.audio_format,
        "-o", str(dst),
        "--no-playlist", "--quiet",
        "--retries", str(cfg.retries),
    ]
    if cookies_path is not None and cookies_path.is_file():
        cmd += ["--cookies", str(cookies_path)]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return Err(DownloadError(
            kind="network", url=url,
            detail=f"yt-dlp timeout after {timeout_s}s",
        ))
    except OSError as e:
        return Err(DownloadError(kind="parse", url=url, detail=str(e)))

    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "")[:300]
        kind = "unavailable" if (
            "unavailable" in msg.lower() or "not available" in msg.lower()
        ) else "parse"
        return Err(DownloadError(kind=kind, url=url, detail=msg))

    if not dst.is_file():
        return Err(DownloadError(
            kind="parse", url=url,
            detail=f"yt-dlp finished cleanly but {dst} missing",
        ))
    return Ok(dst)


def download_one(
    track_id: str,
    query: str,
    cfg: DownloadConfig,
    timeout_s: float = 120.0,
    cookies_path: Path | None = None,
) -> Result[AudioAsset, DownloadError]:
    """Search YT Music for `query`, pick the top 'songs' hit, download it
    via yt-dlp, return an AudioAsset.

    `query` is typically 'Artist - Title' from track_metadata. The top
    'songs' filter result is studio audio (no music video noise).
    """
    sr = search(query, limit=5)
    match sr:
        case Err(err):
            return Err(err)
        case Ok(hits):
            pass

    top = hits[0]
    dl = _ytdlp_download(top.video_id, track_id, cfg, timeout_s, cookies_path)
    match dl:
        case Err(err):
            return Err(err)
        case Ok(path):
            pass

    return Ok(AudioAsset(
        track_audio_id=None,
        track_id=track_id,
        platform="youtube_music",
        source_url=f"https://www.youtube.com/watch?v={top.video_id}",
        player_id=top.video_id,
        path=str(path),
        sha256=_sha256(path),
        duration_s=float(top.duration_s) if top.duration_s else None,
        sample_rate=None,
        codec=cfg.audio_format,
        bitrate_kbps=None,
    ))


def download_batch(
    items: tuple[BatchItem, ...],
    objects_root: Path,
    audio_format: str = "m4a",
    threads: int = 4,
    timeout_s_per_track: float = 120.0,
    cookies_path: Path | None = None,
) -> tuple[BatchResult, ...]:
    """Pooled batch — runs `threads` yt-dlp processes concurrently, one per
    item. Unlike spotdl_adapter.download_batch (which hands all URLs to one
    spotdl invocation), this adapter parallelizes at the item level because
    yt-dlp doesn't natively pool multiple URLs in one process.

    Each item gets its own search → yt-dlp call. Failure of one item does
    not affect siblings.
    """
    if not items:
        return ()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _one(it: BatchItem) -> BatchResult:
        out_dir = objects_root / it.track_id
        cfg = DownloadConfig(
            out_dir=out_dir, audio_format=audio_format, retries=2,
            cookies_path=cookies_path,
        )
        r = download_one(
            it.track_id, it.query, cfg,
            timeout_s=timeout_s_per_track, cookies_path=cookies_path,
        )
        return BatchResult(item=it, result=r)

    results: list[BatchResult] = [None] * len(items)  # type: ignore[list-item]
    idx_by_item = {id(it): i for i, it in enumerate(items)}
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futures = [ex.submit(_one, it) for it in items]
        for it, fut in zip(items, futures):
            results[idx_by_item[id(it)]] = fut.result()
    return tuple(results)
