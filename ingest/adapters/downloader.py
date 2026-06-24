"""Download adapter — wraps yt-dlp. Catches yt_dlp.utils.DownloadError.

Preferred path: yt-dlp with the resolved YT/SC URL scraped from tracklist.
spotdl is a separate fallback for tracks only linked to Spotify (see
spotdl_adapter.py) — this module intentionally does not mix the two.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError as YtDlpDownloadError, ExtractorError

from ..errors import DownloadError
from core.models import AudioAsset, MediaSource, SetAudioAsset
from core.result import Err, Ok, Result


@dataclass(frozen=True)
class DownloadConfig:
    out_dir: Path
    audio_format: str = "m4a"  # yt-dlp postprocessor target
    audio_quality: str = "0"  # best
    retries: int = 3
    cookies_path: Path | None = None  # Netscape cookies.txt for age-gated YouTube
    # (export from a browser via
    # `yt-dlp --cookies-from-browser <name>
    #   --cookies cookies.txt
    #   --skip-download "https://youtube.com"`)
    mac_profile: bool = False  # apply the Mac web_safari/Safari-cookie/EJS
    # profile (see ingest/ytdlp_profile.py); Mac only


def _detect_node() -> str | None:
    """Locate a Node.js binary for yt-dlp's n-challenge JS runtime.

    YouTube's n-parameter obfuscation means yt-dlp needs a JS runtime to
    deobfuscate stream URLs. yt-dlp_ejs provides the solver scripts, but
    they need an interpreter — node works on every box we run on.
    Without this, ~all current YouTube videos return only image formats.
    """
    return shutil.which("node") or shutil.which("nodejs")


def _ydl_opts(cfg: DownloadConfig, out_template: str) -> dict:
    opts: dict = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
        "retries": cfg.retries,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": cfg.audio_format,
                "preferredquality": cfg.audio_quality,
            }
        ],
    }
    if cfg.cookies_path is not None:
        opts["cookiefile"] = str(cfg.cookies_path)
    node_path = _detect_node()
    if node_path is not None:
        opts["js_runtimes"] = {"node": {"location": node_path}}
    if cfg.mac_profile:
        from ..ytdlp_profile import apply_mac_ytdlp_opts

        opts = apply_mac_ytdlp_opts(opts)
    return opts


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_one(
    track_id: str, source: MediaSource, cfg: DownloadConfig
) -> Result[AudioAsset, DownloadError]:
    """Download `source` to disk and return an AudioAsset (not yet DB-persisted)."""
    if source.platform not in ("youtube", "soundcloud"):
        return Err(
            DownloadError(
                kind="unsupported_platform",
                url=source.url,
                detail=f"platform {source.platform!r} not downloadable via yt-dlp here",
            )
        )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    # Name files by stable identity; avoids yt-dlp's title-based template collisions.
    out_template = str(
        cfg.out_dir / f"{track_id}__{source.platform}__{source.player_id}.%(ext)s"
    )

    try:
        with YoutubeDL(_ydl_opts(cfg, out_template)) as ydl:
            info = ydl.extract_info(source.url, download=True)
    except YtDlpDownloadError as e:
        msg = str(e).lower()
        if "unavailable" in msg or "private" in msg or "removed" in msg:
            kind = "unavailable"
        elif "network" in msg or "timed out" in msg or "connection" in msg:
            kind = "network"
        else:
            kind = "parse"
        return Err(DownloadError(kind=kind, url=source.url, detail=str(e)))
    except ExtractorError as e:
        return Err(DownloadError(kind="parse", url=source.url, detail=str(e)))
    except OSError as e:
        return Err(DownloadError(kind="disk", url=source.url, detail=str(e)))

    final_path = Path(out_template.replace("%(ext)s", cfg.audio_format))
    if not final_path.exists():
        candidates = list(
            cfg.out_dir.glob(f"{track_id}__{source.platform}__{source.player_id}.*")
        )
        if not candidates:
            return Err(
                DownloadError(
                    kind="disk",
                    url=source.url,
                    detail=f"expected output missing at {final_path}",
                )
            )
        final_path = candidates[0]

    return Ok(
        AudioAsset(
            track_audio_id=None,
            track_id=track_id,
            platform=source.platform,
            source_url=source.url,
            player_id=source.player_id,
            path=str(final_path),
            sha256=_sha256(final_path),
            duration_s=float(info.get("duration")) if info.get("duration") else None,
            sample_rate=info.get("asr"),
            codec=cfg.audio_format,
            bitrate_kbps=int(info["abr"]) if info.get("abr") else None,
        )
    )


def download_set_mix(
    set_id: str, platform: str, source_url: str, cfg: DownloadConfig
) -> Result[SetAudioAsset, DownloadError]:
    """Download the full-mix audio for a DJ set (as posted on YT/SC/Mixcloud)."""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(cfg.out_dir / f"SET__{set_id}__{platform}.%(ext)s")

    try:
        with YoutubeDL(_ydl_opts(cfg, out_template)) as ydl:
            info = ydl.extract_info(source_url, download=True)
    except YtDlpDownloadError as e:
        msg = str(e).lower()
        if "unavailable" in msg or "private" in msg or "removed" in msg:
            kind = "unavailable"
        elif "network" in msg or "timed out" in msg or "connection" in msg:
            kind = "network"
        else:
            kind = "parse"
        return Err(DownloadError(kind=kind, url=source_url, detail=str(e)))
    except ExtractorError as e:
        return Err(DownloadError(kind="parse", url=source_url, detail=str(e)))
    except OSError as e:
        return Err(DownloadError(kind="disk", url=source_url, detail=str(e)))

    final_path = Path(out_template.replace("%(ext)s", cfg.audio_format))
    if not final_path.exists():
        candidates = list(cfg.out_dir.glob(f"SET__{set_id}__{platform}.*"))
        if not candidates:
            return Err(
                DownloadError(
                    kind="disk",
                    url=source_url,
                    detail=f"expected output missing at {final_path}",
                )
            )
        final_path = candidates[0]

    return Ok(
        SetAudioAsset(
            set_audio_id=None,
            set_id=set_id,
            platform=platform,
            source_url=source_url,
            path=str(final_path),
            sha256=_sha256(final_path),
            duration_s=float(info.get("duration")) if info.get("duration") else None,
            sample_rate=info.get("asr"),
            codec=cfg.audio_format,
            bitrate_kbps=int(info["abr"]) if info.get("abr") else None,
        )
    )
