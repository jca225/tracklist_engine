from __future__ import annotations

import sys
import time
from pathlib import Path

from downloader.commands import run_command
from downloader.storage import detect_downloaded_file, list_audio_files_with_mtime


def _load_local_module(local_dir: Path, module_name: str):
    local_dir = local_dir.expanduser().resolve()
    was_present = str(local_dir) in sys.path
    if not was_present:
        sys.path.insert(0, str(local_dir))
    try:
        return __import__(module_name, fromlist=["*"])
    finally:
        if not was_present:
            try:
                sys.path.remove(str(local_dir))
            except ValueError:
                pass


def _download_spotify_with_spotdl_api(
    url: str,
    output_dir: Path,
    spotify_downloader_dir: Path,
) -> tuple[bool, str]:
    """Try downloading with spotdl's Python API from a local repository checkout."""
    if not spotify_downloader_dir.exists():
        return False, f"spotdl directory not found: {spotify_downloader_dir}"

    try:
        spotdl_module = _load_local_module(spotify_downloader_dir, "spotdl")
        config_module = _load_local_module(spotify_downloader_dir, "spotdl.utils.config")
    except Exception as exc:
        return False, f"spotdl API import failed: {exc}"

    spotdl_client = None
    try:
        downloader_settings = dict(config_module.DOWNLOADER_OPTIONS)
        downloader_settings.update(
            {
                "format": "mp3",
                "bitrate": "320k",
                "output": str(output_dir / "{artists} - {title}.{output-ext}"),
                "simple_tui": True,
                "threads": 1,
            }
        )

        spotdl_client = spotdl_module.Spotdl(
            client_id=config_module.DEFAULT_CONFIG["client_id"],
            client_secret=config_module.DEFAULT_CONFIG["client_secret"],
            user_auth=False,
            cache_path=config_module.DEFAULT_CONFIG["cache_path"],
            no_cache=True,
            headless=True,
            downloader_settings=downloader_settings,
        )

        songs = spotdl_client.search([url])
        if not songs:
            return False, f"spotdl API found no songs for URL: {url}"

        results = spotdl_client.download_songs(songs)
        if any(path is not None for _, path in results):
            return True, ""

        downloader_errors = getattr(spotdl_client.downloader, "errors", []) or []
        if downloader_errors:
            return False, "; ".join(str(err) for err in downloader_errors[:3])
        return False, "spotdl API completed without generating an output file"
    except Exception as exc:
        return False, f"spotdl API download failed: {exc}"
    finally:
        if spotdl_client is not None:
            try:
                spotdl_client.downloader.progress_handler.close()
            except Exception:
                pass


def download_spotify(url: str, output_dir: Path, spotify_downloader_dir: Path) -> tuple[bool, str, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    before_state = list_audio_files_with_mtime(output_dir)
    started_at = time.time()

    api_ok, api_err = _download_spotify_with_spotdl_api(
        url=url,
        output_dir=output_dir,
        spotify_downloader_dir=spotify_downloader_dir,
    )
    if api_ok:
        return True, "", detect_downloaded_file(output_dir, before_state, started_at)

    candidates = [
        [
            "uv",
            "run",
            "spotdl",
            url,
            "--format",
            "mp3",
            "--bitrate",
            "320k",
            "--output",
            str(output_dir),
            "--headless",
        ],
        [
            "python",
            "-m",
            "spotdl",
            url,
            "--format",
            "mp3",
            "--bitrate",
            "320k",
            "--output",
            str(output_dir),
            "--headless",
        ],
        [
            "spotdl",
            url,
            "--format",
            "mp3",
            "--bitrate",
            "320k",
            "--output",
            str(output_dir),
            "--headless",
        ],
    ]

    last_error = f"spotdl API failed: {api_err}"
    for cmd in candidates:
        ok, err = run_command(cmd, cwd=spotify_downloader_dir)
        if ok:
            return True, "", detect_downloaded_file(output_dir, before_state, started_at)
        last_error = f"{last_error}; CLI failed ({' '.join(cmd[:3])}): {err}"
    return False, last_error, None


def download_with_ytdl(url: str, output_dir: Path) -> tuple[bool, str, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    before_state = list_audio_files_with_mtime(output_dir)
    started_at = time.time()
    output_template = str(output_dir / "%(title)s [%(id)s].%(ext)s")

    candidates = [
        ["yt-dl", "--no-playlist", "-x", "--audio-format", "mp3", "-o", output_template, url],
        ["yt-dlp", "--no-playlist", "-x", "--audio-format", "mp3", "-o", output_template, url],
    ]

    last_error = ""
    for cmd in candidates:
        ok, err = run_command(cmd)
        if ok:
            return True, "", detect_downloaded_file(output_dir, before_state, started_at)
        last_error = err
    return False, last_error, None
