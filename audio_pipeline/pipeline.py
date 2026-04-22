"""Pipeline composition — picks best source per track, downloads, persists."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .adapters import db as db_adapter
from .adapters.downloader import DownloadConfig, download_one
from .adapters.spotdl_adapter import SpotdlConfig, download_one_via_spotdl
from .errors import DbError, DownloadError
from .models import DOWNLOAD_PLATFORM_PRIORITY, AudioAsset, MediaSource, Track
from .result import Err, Ok, Result


@dataclass(frozen=True)
class DownloadOutcome:
    track_id: str
    attempted: tuple[str, ...]              # platforms we tried, in order
    success: AudioAsset | None
    last_error: DownloadError | DbError | None


def _pick_source(t: Track) -> MediaSource | None:
    for platform in DOWNLOAD_PLATFORM_PRIORITY:
        s = t.source_for(platform)
        if s is not None:
            return s
    return t.source_for("spotify")  # triggers spotdl fallback


def _download(source: MediaSource, track_id: str, dl: DownloadConfig, sd: SpotdlConfig) -> Result[AudioAsset, DownloadError]:
    if source.platform in ("youtube", "soundcloud"):
        return download_one(track_id, source, dl)
    if source.platform == "spotify":
        return download_one_via_spotdl(track_id, source, sd)
    return Err(DownloadError(kind="unsupported_platform", url=source.url, detail=source.platform))


def process_track(
    t: Track, db_path: Path, dl: DownloadConfig, sd: SpotdlConfig
) -> DownloadOutcome:
    attempted: list[str] = []
    source = _pick_source(t)
    if source is None:
        return DownloadOutcome(t.track_id, (), None, DownloadError(
            kind="unavailable", url="", detail="no downloadable source on any platform",
        ))

    already = db_adapter.already_downloaded(db_path, t.track_id, source.platform, source.player_id)
    match already:
        case Ok(True):
            return DownloadOutcome(t.track_id, (source.platform,), None, None)  # skip
        case Err(e):
            return DownloadOutcome(t.track_id, (), None, e)

    attempted.append(source.platform)
    dl_result = _download(source, t.track_id, dl, sd)
    match dl_result:
        case Err(e):
            return DownloadOutcome(t.track_id, tuple(attempted), None, e)
        case Ok(asset):
            ins = db_adapter.insert_audio(db_path, asset)
            match ins:
                case Err(e):
                    return DownloadOutcome(t.track_id, tuple(attempted), asset, e)
                case Ok(new_id):
                    persisted = AudioAsset(**{**asset.__dict__, "track_audio_id": new_id})
                    return DownloadOutcome(t.track_id, tuple(attempted), persisted, None)


def process_set(
    db_path: Path, set_id: str, dl: DownloadConfig, sd: SpotdlConfig
) -> Result[tuple[DownloadOutcome, ...], DbError]:
    tracks_r = db_adapter.load_set_tracks(db_path, set_id)
    match tracks_r:
        case Err(e):
            return Err(e)
        case Ok(tracks):
            return Ok(tuple(process_track(t, db_path, dl, sd) for t in tracks))
