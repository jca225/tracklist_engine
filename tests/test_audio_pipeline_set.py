"""Tests for the set-level pipeline: URL normalization, set-audio DB
round-trip, timeline construction, and pipeline composition with mocks.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from audio_pipeline import set_pipeline
from audio_pipeline.adapters import db as db_adapter
from audio_pipeline.errors import DbError, DownloadError
from audio_pipeline.models import SetAudioAsset, SetMediaLink, normalize_set_media_url
from audio_pipeline.result import Err, Ok
from audio_pipeline.timeline import (
    build_timeline, concurrent_groups, reference_track_ids, timeline_to_json,
)


_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "web_crawler" / "database" / "schema.sql"


# ---------- URL normalization -----------------------------------------------

def test_normalize_unwraps_sc_widget():
    raw = ("https://w.soundcloud.com/player/?url=https://api.soundcloud.com/tracks/"
           "260059381&show_artwork=true&color=%23ff5500")
    assert normalize_set_media_url(raw) == "https://api.soundcloud.com/tracks/260059381"


def test_normalize_passes_youtube_through():
    raw = "https://www.youtube.com/watch?v=abc"
    assert normalize_set_media_url(raw) == raw


def test_normalize_handles_empty():
    assert normalize_set_media_url("") == ""


# ---------- DB adapter round-trip -------------------------------------------

@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    schema = _SCHEMA_PATH.read_text()
    conn = sqlite3.connect(path)
    conn.executescript(schema)
    conn.execute(
        "INSERT INTO dj_sets (set_id, set_url, title) VALUES (?, ?, ?)",
        ("S", "u", "Two Friends - Big Bootie Mix Vol. 99"),
    )
    conn.executemany(
        "INSERT INTO dj_set_media_links (set_id, platform, url) VALUES (?, ?, ?)",
        [
            ("S", "youtube", "https://www.youtube.com/watch?v=abc"),
            ("S", "soundcloud", "https://w.soundcloud.com/player/?url=https://api.soundcloud.com/tracks/42"),
            ("S", "mixcloud", "https://www.mixcloud.com/x/y/"),
        ],
    )
    conn.commit(); conn.close()
    return path


def test_load_set_media_links_normalizes_and_preserves_platforms(db_path: Path):
    r = db_adapter.load_set_media_links(db_path, "S")
    assert isinstance(r, Ok)
    links = {l.platform: l.url for l in r.value}
    assert links["youtube"] == "https://www.youtube.com/watch?v=abc"
    assert links["soundcloud"] == "https://api.soundcloud.com/tracks/42"
    assert "mixcloud" in links


def test_insert_set_audio_idempotent(db_path: Path):
    asset = SetAudioAsset(
        set_audio_id=None, set_id="S", platform="youtube",
        source_url="https://www.youtube.com/watch?v=abc",
        path="/tmp/mix.m4a", sha256="f0",
        duration_s=3600.0, sample_rate=44100, codec="m4a", bitrate_kbps=128,
    )
    first = db_adapter.insert_set_audio(db_path, asset)
    assert isinstance(first, Ok)

    marked = db_adapter.already_downloaded_set(db_path, "S", "youtube", "https://www.youtube.com/watch?v=abc")
    assert isinstance(marked, Ok) and marked.value is True

    second = db_adapter.insert_set_audio(db_path, asset)
    assert isinstance(second, Ok) and second.value == first.value


def test_upsert_timeline_replaces_on_conflict(db_path: Path):
    r1 = db_adapter.upsert_timeline(db_path, "S", None, '{"v":1}')
    assert isinstance(r1, Ok)
    r2 = db_adapter.upsert_timeline(db_path, "S", None, '{"v":2}')
    assert isinstance(r2, Ok)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT payload_json FROM set_timeline WHERE set_id=?", ("S",)).fetchone()
    assert row[0] == '{"v":2}'


# ---------- timeline construction -------------------------------------------

def _tokens(records: list[dict]) -> pd.DataFrame:
    """Minimal token DataFrame — only the columns timeline.py reads."""
    base = {
        "row_kind": "track", "track_key": None, "row_dom_id": None,
        "title": None, "artists": None, "cue_seconds_section": None,
        "is_ided": False, "is_concurrent": False, "is_remixish": False,
        "has_yt": False, "has_sc": False, "has_sp": False, "has_ap": False,
    }
    return pd.DataFrame([{**base, **r} for r in records])


def test_build_timeline_orders_by_row_index():
    df = _tokens([
        {"row_index": 2, "track_key": "T2", "title": "B", "cue_seconds_section": 100.0},
        {"row_index": 0, "track_key": "T0", "title": "A", "cue_seconds_section": 0.0},
        {"row_index": 1, "track_key": "T1", "title": "A2", "cue_seconds_section": 0.0, "is_concurrent": True},
    ])
    tl = build_timeline("S", df)
    assert [s.row_index for s in tl.segments] == [0, 1, 2]
    assert tl.segments[1].is_concurrent is True
    assert tl.segments[2].cue_seconds_section == 100.0


def test_build_timeline_parses_pipe_separated_artists():
    df = _tokens([{"row_index": 0, "track_key": "T", "title": "X", "artists": "Alice|Bob"}])
    tl = build_timeline("S", df)
    assert tl.segments[0].artists == ("Alice", "Bob")


def test_build_timeline_excludes_non_track_rows():
    df = _tokens([
        {"row_index": 0, "row_kind": "player_widget"},
        {"row_index": 1, "row_kind": "track", "track_key": "T", "title": "A"},
        {"row_index": 2, "row_kind": "save_footer"},
    ])
    tl = build_timeline("S", df)
    assert len(tl.segments) == 1
    assert tl.segments[0].row_index == 1


def test_timeline_to_json_is_stable_and_parseable():
    df = _tokens([{"row_index": 0, "track_key": "T", "title": "A", "artists": "X|Y"}])
    tl = build_timeline("S", df, set_audio_id=7)
    payload = json.loads(timeline_to_json(tl))
    assert payload["set_id"] == "S"
    assert payload["set_audio_id"] == 7
    assert payload["segments"][0]["artists"] == ["X", "Y"]


# ---------- tokenizer-informed helpers --------------------------------------

def test_reference_track_ids_excludes_remixish_and_not_ided():
    df = _tokens([
        {"row_index": 0, "track_key": "A", "is_ided": True,  "is_remixish": False},
        {"row_index": 1, "track_key": "B", "is_ided": True,  "is_remixish": True},   # skip (remix)
        {"row_index": 2, "track_key": "C", "is_ided": False, "is_remixish": False},  # skip (ID unknown)
        {"row_index": 3, "track_key": "D", "is_ided": True,  "is_remixish": False},
    ])
    assert set(reference_track_ids(df)) == {"A", "D"}


def test_concurrent_groups_groups_by_cue_section():
    df = _tokens([
        {"row_index": 0, "track_key": "A", "cue_seconds_section": 10.0},
        {"row_index": 1, "track_key": "B", "cue_seconds_section": 10.0},
        {"row_index": 2, "track_key": "C", "cue_seconds_section": 20.0},
    ])
    groups = concurrent_groups(df)
    assert tuple(tuple(sorted(g)) for g in groups) == ((0, 1), (2,))


# ---------- pipeline composition with mocks ---------------------------------

def test_process_set_downloads_then_builds_timeline(db_path: Path):
    # Seed a track row so the tokenizer has something to do.
    with sqlite3.connect(db_path) as c:
        c.execute(
            """INSERT INTO dj_set_rows (set_id, row_index, element_id, classes,
               data_attrs_json, text_excerpt, raw_html)
               VALUES ('S', 1, 'tlp_1', 'tlpTog bItm tlpItem', NULL, 'Foo',
               '<div class="tlpTog bItm tlpItem" data-id="1" data-trno="0" data-trackid="T" data-isided="true"><meta itemprop="name" content="Foo"/></div>')"""
        )
        c.commit()

    dl_cfg = set_pipeline.DownloadConfig(out_dir=Path("/tmp/ignored"))
    returned = SetAudioAsset(
        set_audio_id=None, set_id="S", platform="youtube",
        source_url="https://www.youtube.com/watch?v=abc",
        path="/tmp/fake.m4a", sha256="f0",
        duration_s=60.0, sample_rate=44100, codec="m4a", bitrate_kbps=128,
    )
    with patch.object(set_pipeline, "download_set_mix", return_value=Ok(returned)) as m_dl:
        out = set_pipeline.process_set(db_path, "S", dl_cfg)

    m_dl.assert_called_once()
    assert out.audio is not None and out.audio.set_audio_id is not None
    assert out.timeline is not None
    assert out.last_error is None
    with sqlite3.connect(db_path) as c:
        n = c.execute("SELECT COUNT(*) FROM set_timeline WHERE set_id='S'").fetchone()[0]
    assert n == 1


def test_process_set_no_links_returns_unavailable(db_path: Path):
    # Wipe the seeded media links to simulate the 9 "other" platform edge-cases.
    with sqlite3.connect(db_path) as c:
        c.execute("DELETE FROM dj_set_media_links")
        c.commit()
    out = set_pipeline.process_set(db_path, "S", set_pipeline.DownloadConfig(out_dir=Path("/tmp/x")))
    assert out.audio is None and out.timeline is None
    assert isinstance(out.last_error, DownloadError)
    assert out.last_error.kind == "unavailable"
