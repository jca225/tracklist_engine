"""Tests for the pipeline composition, mocking out the I/O adapters."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from audio_pipeline import pipeline as pipeline_mod
from audio_pipeline.errors import DbError, DownloadError
from audio_pipeline.models import AudioAsset, MediaSource, Track
from audio_pipeline.result import Err, Ok


# --- helpers -----------------------------------------------------------------

_NO_DL = pipeline_mod.DownloadConfig(out_dir=Path("/tmp/fake_out"))
_NO_SD = pipeline_mod.SpotdlConfig(out_dir=Path("/tmp/fake_out"))


def _asset(track_id: str, platform: str, player_id: str) -> AudioAsset:
    return AudioAsset(
        track_audio_id=None,
        track_id=track_id, platform=platform,
        source_url=f"https://x/{platform}/{player_id}",
        player_id=player_id,
        path=f"/tmp/{track_id}.m4a", sha256="f0",
        duration_s=120.0, sample_rate=44100, codec="m4a", bitrate_kbps=128,
    )


def _track(track_id: str, *platforms: str) -> Track:
    sources = tuple(
        MediaSource(platform=p, player_id=f"id-{p}", url=f"https://x/{p}") for p in platforms
    )
    return Track(track_id=track_id, tlp_ids=(), sources=sources)


# --- tests -------------------------------------------------------------------

def test_skip_when_already_downloaded():
    t = _track("T1", "youtube")
    with patch.object(pipeline_mod.db_adapter, "already_downloaded", return_value=Ok(True)) as m_ad, \
         patch.object(pipeline_mod, "download_one") as m_dl:
        outcome = pipeline_mod.process_track(t, Path("db"), _NO_DL, _NO_SD)
    m_dl.assert_not_called()
    assert outcome.success is None
    assert outcome.last_error is None
    assert outcome.attempted == ("youtube",)


def test_picks_youtube_over_spotify():
    """Download priority: YT > SC; Spotify only as fallback via spotdl."""
    t = _track("T1", "spotify", "youtube")
    returned = _asset("T1", "youtube", "id-youtube")
    with patch.object(pipeline_mod.db_adapter, "already_downloaded", return_value=Ok(False)), \
         patch.object(pipeline_mod.db_adapter, "insert_audio", return_value=Ok(42)), \
         patch.object(pipeline_mod, "download_one", return_value=Ok(returned)) as m_dl, \
         patch.object(pipeline_mod, "download_one_via_spotdl") as m_sd:
        outcome = pipeline_mod.process_track(t, Path("db"), _NO_DL, _NO_SD)
    m_dl.assert_called_once()
    m_sd.assert_not_called()
    assert outcome.attempted == ("youtube",)
    assert outcome.success is not None
    assert outcome.success.track_audio_id == 42


def test_falls_back_to_spotdl_when_only_spotify():
    t = _track("T2", "spotify")
    returned = _asset("T2", "spotify", "id-spotify")
    with patch.object(pipeline_mod.db_adapter, "already_downloaded", return_value=Ok(False)), \
         patch.object(pipeline_mod.db_adapter, "insert_audio", return_value=Ok(7)), \
         patch.object(pipeline_mod, "download_one_via_spotdl", return_value=Ok(returned)) as m_sd, \
         patch.object(pipeline_mod, "download_one") as m_dl:
        outcome = pipeline_mod.process_track(t, Path("db"), _NO_DL, _NO_SD)
    m_sd.assert_called_once()
    m_dl.assert_not_called()
    assert outcome.attempted == ("spotify",)
    assert outcome.success is not None


def test_no_downloadable_source_returns_unavailable_error():
    # Track with no known sources at all.
    t = _track("T3")
    outcome = pipeline_mod.process_track(t, Path("db"), _NO_DL, _NO_SD)
    assert outcome.success is None
    assert isinstance(outcome.last_error, DownloadError)
    assert outcome.last_error.kind == "unavailable"


def test_download_error_propagates_without_insert():
    t = _track("T1", "youtube")
    err = DownloadError(kind="network", url="https://x", detail="timeout")
    with patch.object(pipeline_mod.db_adapter, "already_downloaded", return_value=Ok(False)), \
         patch.object(pipeline_mod, "download_one", return_value=Err(err)), \
         patch.object(pipeline_mod.db_adapter, "insert_audio") as m_ins:
        outcome = pipeline_mod.process_track(t, Path("db"), _NO_DL, _NO_SD)
    m_ins.assert_not_called()
    assert outcome.success is None
    assert outcome.last_error is err


def test_db_error_on_already_downloaded_short_circuits():
    t = _track("T1", "youtube")
    dberr = DbError(kind="query_failed", detail="boom")
    with patch.object(pipeline_mod.db_adapter, "already_downloaded", return_value=Err(dberr)), \
         patch.object(pipeline_mod, "download_one") as m_dl:
        outcome = pipeline_mod.process_track(t, Path("db"), _NO_DL, _NO_SD)
    m_dl.assert_not_called()
    assert outcome.last_error is dberr
