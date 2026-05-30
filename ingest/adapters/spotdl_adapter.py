"""spotdl adapter — Spotify-link → audio file via Spotify-metadata-seeded
YouTube Music search. Used as a fallback when YouTube/SoundCloud tries fail
or when the only scraped link is a Spotify track URL.

spotdl is a separate CLI tool (different binary, different cred handling
from yt-dlp). We shell out rather than embed because spotdl's Python API
isn't a clean library boundary — its argument parsing eats sys.argv.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from ..errors import DownloadError
from core.models import AudioAsset, MediaSource
from core.result import Err, Ok, Result
from .downloader import DownloadConfig


def _spotdl_bin() -> str | None:
    """Find spotdl. Search order:
      1. <repo>/venvs/spotdl/bin/spotdl  — dedicated isolated venv (preferred;
         spotdl's fastapi pin conflicts with streamlit/web_crawler)
      2. Same venv as the running Python (works on Mac dev where there's
         only one audio venv)
      3. PATH fallback
    """
    # __file__ = <repo>/ingest/adapters/spotdl_adapter.py → parents[2]
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
    client_id: str | None = None, client_secret: str | None = None,
) -> Result[AudioAsset, DownloadError]:
    """Download a Spotify track via spotdl + yt-music search.

    spotdl writes one file with a metadata-derived name (e.g.
    "Artist - Title.m4a"). We rename to our `<track_id>__spotify__<player_id>.<ext>`
    convention so the file is consistent with the yt-dlp downloader.

    `client_id` / `client_secret` (optional): Spotify Web API app credentials.
    spotdl ships with default shared creds, but they're globally rate-limited
    (24h backoff per "Your application has reached a rate/request limit"). Pass
    real app creds for any production-volume run; fall back to env vars
    SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET when args are None.
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

    cid = client_id or os.environ.get("SPOTIFY_CLIENT_ID")
    csec = client_secret or os.environ.get("SPOTIFY_CLIENT_SECRET")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    # Snapshot pre-download contents so we can identify spotdl's new file.
    before = set(cfg.out_dir.iterdir())

    # spotdl 4.x: operation arg ('download') first, --format (not --output-format).
    cmd = [
        bin_path, "download", source.url,
        "--output", str(cfg.out_dir),
        "--format", cfg.audio_format,
    ]
    if cid and csec:
        cmd += ["--client-id", cid, "--client-secret", csec]
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

    # Rename to the project's stable convention. If the rename fails, reap the
    # staged bare-name file ("Artist - Title.m4a") so a failed canonicalization
    # can't leave it on disk as an orphan (the spotdl variant of the orphan bug).
    dst_path = cfg.out_dir / f"{track_id}__spotify__{source.player_id}.{cfg.audio_format}"
    if dst_path.exists() and dst_path != src_path:
        dst_path.unlink()
    try:
        src_path.rename(dst_path)
    except OSError as e:
        src_path.unlink(missing_ok=True)
        return Err(DownloadError(
            kind="parse", url=source.url,
            detail=f"canonical rename failed, reaped staged file: {e}",
        ))

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


@dataclass(frozen=True)
class BatchItem:
    """One unit of work for `download_batch` — a canonical track_id paired
    with the resolved Spotify MediaSource we want spotdl to fetch."""
    track_id: str
    source: MediaSource


@dataclass(frozen=True)
class BatchResult:
    """Per-item outcome of a batched spotdl run.

    `result` is `Ok(AudioAsset)` on success or `Err(DownloadError)` when the
    item didn't produce a file. Items where spotdl crashed mid-batch are
    reported as `kind='unavailable'` since we can't tell from outside
    whether the URL was rejected or simply never reached.
    """
    item: BatchItem
    result: Result[AudioAsset, DownloadError]


def download_batch(
    items: tuple[BatchItem, ...],
    objects_root: Path,
    audio_format: str = "m4a",
    threads: int = 4,
    timeout_s: float = 600.0,
    client_id: str | None = None,
    client_secret: str | None = None,
) -> tuple[BatchResult, ...]:
    """Pooled spotdl run: many URLs, one process, internal thread pool.

    Why pooling matters: each spotdl invocation pays ~3s Python startup and
    ~3-5s spotipy auth. Passing N URLs to one call amortizes both costs and
    lets spotdl run yt-music searches concurrently across `threads` workers.
    For the BB-only smoke test we measured ~38s/track sequential; pooled at
    threads=4 should land closer to ~12-15s/track wall.

    Mapping files back to inputs uses spotdl's `{track-id}` template — a
    documented variable that resolves to the Spotify track ID. Files land
    as `{spotify_id}.{ext}` in a temp staging dir; we walk the dir after,
    match each input by player_id, move to canonical per-track location,
    and rename to `{track_id}__spotify__{player_id}.{ext}`.

    Failure modes:
    - File for an item didn't appear  → that item gets Err(unavailable)
    - spotdl exited non-zero / timed out → items without files get Err(network);
      items WITH files (spotdl had partial success before dying) still succeed
    - spotdl binary missing → all items Err(parse)
    """
    if not items:
        return ()

    bin_path = _spotdl_bin()
    if bin_path is None:
        err = DownloadError(
            kind="parse", url="",
            detail="spotdl not on PATH (install via `pip install spotdl`)",
        )
        return tuple(BatchResult(it, Err(err)) for it in items)

    cid = client_id or os.environ.get("SPOTIFY_CLIENT_ID")
    csec = client_secret or os.environ.get("SPOTIFY_CLIENT_SECRET")

    staging = Path(tempfile.mkdtemp(prefix=f"spotdl_batch_{os.getpid()}_", dir=objects_root.parent))
    try:
        # spotdl will write {track-id}.{output-ext} → e.g. 1jUT2mNI...UbY.m4a
        template = str(staging / "{track-id}.{output-ext}")
        cmd = [
            bin_path, "download",
            *[it.source.url for it in items],
            "--output", template,
            "--format", audio_format,
            "--threads", str(threads),
        ]
        if cid and csec:
            cmd += ["--client-id", cid, "--client-secret", csec]

        batch_err: DownloadError | None = None
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
            if proc.returncode != 0:
                msg = (proc.stderr or proc.stdout or "")[:300]
                batch_err = DownloadError(
                    kind="network", url="",
                    detail=f"spotdl batch exit {proc.returncode}: {msg}",
                )
        except subprocess.TimeoutExpired:
            batch_err = DownloadError(
                kind="network", url="",
                detail=f"spotdl batch timeout after {timeout_s}s "
                       f"(items={len(items)}, threads={threads})",
            )
        except OSError as e:
            batch_err = DownloadError(kind="parse", url="", detail=str(e))

        # Walk staging regardless — partial successes survive batch failure.
        results: list[BatchResult] = []
        for it in items:
            staged = staging / f"{it.source.player_id}.{audio_format}"
            if not staged.is_file():
                # File missing → unavailable (or batch died before reaching it).
                err = batch_err or DownloadError(
                    kind="unavailable", url=it.source.url,
                    detail=f"spotdl produced no file for spotify:{it.source.player_id}",
                )
                results.append(BatchResult(it, Err(err)))
                continue

            # Move + rename to canonical per-track convention.
            track_dir = objects_root / it.track_id
            track_dir.mkdir(parents=True, exist_ok=True)
            dst = track_dir / f"{it.track_id}__spotify__{it.source.player_id}.{audio_format}"
            if dst.exists():
                dst.unlink()
            try:
                shutil.move(str(staged), str(dst))
            except OSError as e:
                # Reap any partial destination; staged stays in temp (cleaned in finally).
                Path(dst).unlink(missing_ok=True)
                results.append(BatchResult(it, Err(DownloadError(
                    kind="parse", url=it.source.url,
                    detail=f"canonical move failed: {e}",
                ))))
                continue

            asset = AudioAsset(
                track_audio_id=None,
                track_id=it.track_id,
                platform="spotify",
                source_url=it.source.url,
                player_id=it.source.player_id,
                path=str(dst),
                sha256=_sha256(dst),
                duration_s=None,
                sample_rate=None,
                codec=audio_format,
                bitrate_kbps=None,
            )
            results.append(BatchResult(it, Ok(asset)))
        return tuple(results)
    finally:
        # Always clean up the staging dir, even on exception.
        shutil.rmtree(staging, ignore_errors=True)
        # Time import is still needed somewhere — keep it referenced even if
        # we trim later. (Used for batch ETA logging in main_retry.)
        _ = time.time
