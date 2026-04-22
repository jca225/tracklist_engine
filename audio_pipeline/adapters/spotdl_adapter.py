"""spotdl fallback adapter — for tracks that have ONLY a Spotify link and no
scraped YT/SC URL. Uses spotdl's CLI (stable entry point) rather than importing
its internal API, which changes frequently between versions.

Why use spotdl at all when YT/SC is preferred? For the ~10% of tracks where
tracklist scraped no direct YT/SC link, spotdl performs a YT Music search
seeded by Spotify metadata — much more accurate than an ad-hoc title search.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..errors import DownloadError
from ..models import AudioAsset, MediaSource, spotify_track_url
from ..result import Err, Ok, Result


@dataclass(frozen=True)
class SpotdlConfig:
    out_dir: Path
    spotdl_binary: str = "spotdl"    # absolute path or name on PATH
    audio_format: str = "m4a"
    retries: int = 2


def _ext_from_format(fmt: str) -> str:
    return {"m4a": "m4a", "mp3": "mp3", "opus": "opus", "flac": "flac", "wav": "wav"}.get(fmt, fmt)


def download_one_via_spotdl(
    track_id: str, source: MediaSource, cfg: SpotdlConfig
) -> Result[AudioAsset, DownloadError]:
    if source.platform != "spotify":
        return Err(DownloadError(
            kind="unsupported_platform", url=source.url,
            detail="spotdl adapter only handles spotify sources",
        ))

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    url = spotify_track_url(source.player_id)
    # spotdl writes via its own template; we rename after.
    tmp_template = str(cfg.out_dir / f"__spotdl_{track_id}__{{artists}} - {{title}}.{{output-ext}}")
    ext = _ext_from_format(cfg.audio_format)

    try:
        proc = subprocess.run(
            [
                cfg.spotdl_binary, "download", url,
                "--output", tmp_template,
                "--format", cfg.audio_format,
                "--print-errors",
            ],
            capture_output=True, text=True, timeout=600,
        )
    except FileNotFoundError:
        return Err(DownloadError(kind="tool_missing", url=url, detail="spotdl not on PATH"))
    except subprocess.TimeoutExpired:
        return Err(DownloadError(kind="network", url=url, detail="spotdl timed out"))

    if proc.returncode != 0:
        kind = "unavailable" if "not found" in proc.stderr.lower() else "parse"
        return Err(DownloadError(kind=kind, url=url, detail=proc.stderr[:500]))

    produced = list(cfg.out_dir.glob(f"__spotdl_{track_id}__*.{ext}"))
    if not produced:
        return Err(DownloadError(kind="disk", url=url, detail="spotdl reported success but no file found"))
    src_file = produced[0]
    dest = cfg.out_dir / f"{track_id}__spotify__{source.player_id}.{ext}"
    src_file.replace(dest)

    import hashlib
    h = hashlib.sha256()
    with dest.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return Ok(AudioAsset(
        track_audio_id=None,
        track_id=track_id,
        platform="spotify",
        source_url=url,
        player_id=source.player_id,
        path=str(dest),
        sha256=h.hexdigest(),
        duration_s=None,
        sample_rate=None,
        codec=ext,
        bitrate_kbps=None,
    ))
