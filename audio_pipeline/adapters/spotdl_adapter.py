"""spotdl adapter — Spotify-link → audio file via Spotify-metadata-seeded
YouTube Music search. Used as a fallback when YouTube/SoundCloud tries fail
or when the only scraped link is a Spotify track URL.

spotdl is a separate CLI tool (different binary, different cred handling
from yt-dlp). We shell out rather than embed because spotdl's Python API
isn't a clean library boundary — its argument parsing eats sys.argv.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ..errors import DownloadError
from ..models import AudioAsset, MediaSource
from ..result import Err, Ok, Result
from .downloader import DownloadConfig


def _spotdl_bin() -> str | None:
    """Find spotdl. Search order:
      1. <repo>/venvs/spotdl/bin/spotdl  — dedicated isolated venv (preferred;
         spotdl's fastapi pin conflicts with streamlit/web_crawler)
      2. Same venv as the running Python (works on Mac dev where there's
         only one audio venv)
      3. PATH fallback
    """
    # __file__ = <repo>/audio_pipeline/adapters/spotdl_adapter.py → parents[2]
    repo_root = Path(__file__).resolve().parents[2]
    isolated = repo_root / "venvs" / "spotdl" / "bin" / "spotdl"
    if isolated.is_file():
        return str(isolated)
    same_venv = Path(sys.executable).parent / "spotdl"
    if same_venv.is_file():
        return str(same_venv)
    return shutil.which("spotdl")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_one(
    track_id: str, source: MediaSource, cfg: DownloadConfig, timeout_s: float = 120.0,
) -> Result[AudioAsset, DownloadError]:
    """Download a Spotify track via spotdl + yt-music search.

    spotdl writes one file with a metadata-derived name (e.g.
    "Artist - Title.m4a"). We rename to our `<track_id>__spotify__<player_id>.<ext>`
    convention so the file is consistent with the yt-dlp downloader.
    """
    if source.platform != "spotify":
        return Err(DownloadError(
            kind="unsupported_platform", url=source.url,
            detail=f"spotdl_adapter requires platform='spotify', got {source.platform!r}",
        ))

    bin_path = _spotdl_bin()
    if bin_path is None:
        return Err(DownloadError(
            kind="parse", url=source.url,
            detail="spotdl not on PATH (install via `pip install spotdl` in venvs/audio)",
        ))

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    # Snapshot pre-download contents so we can identify spotdl's new file.
    before = set(cfg.out_dir.iterdir())

    # spotdl 4.x: operation arg ('download') first, --format (not --output-format).
    cmd = [
        bin_path, "download", source.url,
        "--output", str(cfg.out_dir),
        "--format", cfg.audio_format,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return Err(DownloadError(
            kind="network", url=source.url,
            detail=f"spotdl timeout after {timeout_s}s",
        ))
    except OSError as e:
        return Err(DownloadError(kind="parse", url=source.url, detail=str(e)))

    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "")[:300]
        kind = "unavailable" if "not found" in msg.lower() or "no songs" in msg.lower() else "parse"
        return Err(DownloadError(kind=kind, url=source.url, detail=msg))

    new_files = sorted(set(cfg.out_dir.iterdir()) - before)
    new_audio = [p for p in new_files if p.suffix.lower().lstrip(".") == cfg.audio_format]
    if not new_audio:
        return Err(DownloadError(
            kind="parse", url=source.url,
            detail=f"spotdl produced no .{cfg.audio_format} file in {cfg.out_dir}",
        ))
    src_path = new_audio[0]

    # Rename to the project's stable convention.
    dst_path = cfg.out_dir / f"{track_id}__spotify__{source.player_id}.{cfg.audio_format}"
    if dst_path.exists() and dst_path != src_path:
        dst_path.unlink()
    src_path.rename(dst_path)

    return Ok(AudioAsset(
        track_audio_id=None,
        track_id=track_id,
        platform="spotify",
        source_url=source.url,
        player_id=source.player_id,
        path=str(dst_path),
        sha256=_sha256(dst_path),
        duration_s=None,        # spotdl doesn't surface this cleanly via CLI
        sample_rate=None,
        codec=cfg.audio_format,
        bitrate_kbps=None,
    ))
