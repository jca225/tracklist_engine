"""Set-level pipeline: download the full-mix audio for a DJ set and persist
the tokenized timeline alongside it.

Composition:
  load_set_media_links  →  pick best URL  →  yt-dlp download  →  insert_set_audio
  load rows             →  tokenize      →  build_timeline  →  upsert_timeline
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from .adapters import db as db_adapter
from .adapters.downloader import DownloadConfig, download_set_mix
from .errors import DbError, DownloadError
from .models import SetAudioAsset, SetMediaLink, SetTimeline
from .result import Err, Ok, Result
from .timeline import build_timeline, timeline_to_json


# Preferred source for the full-mix audio. Mixcloud is risky (yt-dlp support
# is spotty and they gate quality), so we treat it as a last resort.
SET_PLATFORM_PRIORITY = ("youtube", "soundcloud", "mixcloud", "other")


@dataclass(frozen=True)
class SetDownloadOutcome:
    set_id: str
    attempted: tuple[str, ...]
    audio: SetAudioAsset | None
    timeline: SetTimeline | None
    last_error: DownloadError | DbError | None


def _pick_link(links: tuple[SetMediaLink, ...]) -> SetMediaLink | None:
    if not links:
        return None
    by_platform: dict[str, SetMediaLink] = {}
    for l in links:
        by_platform.setdefault(l.platform, l)
    for p in SET_PLATFORM_PRIORITY:
        if p in by_platform:
            return by_platform[p]
    return next(iter(links))


def _tokenize_rows(db_path: Path, set_id: str) -> pd.DataFrame:
    """Load rows for this set and run the tokenizer. Importing big_bootie here
    (not at module top) keeps `audio_pipeline` independent of pandas at import
    time for the unit-test subset that doesn't exercise this path."""
    from big_bootie import tokenize_rows
    with sqlite3.connect(db_path) as conn:
        rows = pd.read_sql_query(
            "SELECT * FROM dj_set_rows WHERE set_id = ? ORDER BY row_index",
            conn, params=(set_id,),
        )
    return tokenize_rows(rows)


def process_set(db_path: Path, set_id: str, dl: DownloadConfig) -> SetDownloadOutcome:
    """Download full-mix + build+persist the timeline sidecar. Idempotent."""
    attempted: list[str] = []

    links_r = db_adapter.load_set_media_links(db_path, set_id)
    match links_r:
        case Err(e):
            return SetDownloadOutcome(set_id, (), None, None, e)
        case Ok(links):
            pass

    chosen = _pick_link(links)
    if chosen is None:
        err = DownloadError(kind="unavailable", url="",
                            detail=f"no set-level media link for {set_id}")
        return SetDownloadOutcome(set_id, (), None, None, err)

    attempted.append(chosen.platform)
    already = db_adapter.already_downloaded_set(db_path, set_id, chosen.platform, chosen.url)
    match already:
        case Err(e):
            return SetDownloadOutcome(set_id, tuple(attempted), None, None, e)
        case Ok(True):
            audio_asset = None
            set_audio_id = None   # could look up the id, but not required for timeline build
        case Ok(False):
            dl_r = download_set_mix(set_id, chosen.platform, chosen.url, dl)
            match dl_r:
                case Err(e):
                    return SetDownloadOutcome(set_id, tuple(attempted), None, None, e)
                case Ok(asset):
                    ins = db_adapter.insert_set_audio(db_path, asset)
                    match ins:
                        case Err(e):
                            return SetDownloadOutcome(set_id, tuple(attempted), asset, None, e)
                        case Ok(new_id):
                            audio_asset = replace(asset, set_audio_id=new_id)
                            set_audio_id = new_id

    # Build the timeline sidecar regardless of whether we just downloaded or skipped.
    try:
        tokens = _tokenize_rows(db_path, set_id)
    except Exception as e:
        return SetDownloadOutcome(set_id, tuple(attempted), audio_asset, None,
                                  DbError(kind="query_failed", detail=f"tokenize failed: {e}"))
    timeline = build_timeline(set_id, tokens, set_audio_id=set_audio_id)
    ts_r = db_adapter.upsert_timeline(db_path, set_id, set_audio_id, timeline_to_json(timeline))
    match ts_r:
        case Err(e):
            return SetDownloadOutcome(set_id, tuple(attempted), audio_asset, timeline, e)
    return SetDownloadOutcome(set_id, tuple(attempted), audio_asset, timeline, None)
