"""Tests for analysis scaffolding.

Covers the pure pieces (section slicing, bpm estimation, measure derivation)
and the DB persistence adapter against the real schema. Heavy integration
tests that actually invoke torch/demucs/MERT models are not included here
— those require downloaded audio fixtures.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from archive.audio_pipeline.adapters import db as db_adapter
from archive.audio_pipeline.analysis import pipeline
from archive.audio_pipeline.analysis.adapters import beat_this_adapter
from archive.audio_pipeline.analysis.models import (
    BeatGrid,
    CuePoints,
    LoudnessReading,
    SectionEmbedding,
    StemAsset,
    StemSet,
    TrackAnalysisResult,
)
from archive.audio_pipeline.models import AudioAsset
from archive.audio_pipeline.result import Ok

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "web_crawler" / "database" / "schema.sql"


# ---------- pure helpers ----------------------------------------------------

def test_section_bounds_no_cues_returns_whole_track() -> None:
    assert pipeline._section_bounds((), 30.0) == ((0.0, 30.0),)


def test_section_bounds_splits_on_cues_sorted_and_deduped() -> None:
    bounds = pipeline._section_bounds((10.0, 5.0, 10.0, 25.0), 30.0)
    assert bounds == ((0.0, 5.0), (5.0, 10.0), (10.0, 25.0), (25.0, 30.0))


def test_section_bounds_drops_zero_length_spans() -> None:
    # A cue exactly at the end plus duplicates shouldn't produce empty spans.
    bounds = pipeline._section_bounds((0.0, 30.0), 30.0)
    assert bounds == ((0.0, 30.0),)


def test_slice_clamps_to_signal_extent() -> None:
    sr = 1000
    samples = np.arange(5 * sr, dtype=np.float32)
    sliced = pipeline._slice(samples, sr, 1.0, 100.0)  # past the end
    assert sliced.size == 4 * sr
    assert sliced[0] == 1 * sr


def test_slice_empty_when_start_past_end() -> None:
    assert pipeline._slice(np.zeros(1000, dtype=np.float32), 1000, 2.0, 3.0).size == 0


# ---------- beat_this helpers (pure, no model load) -------------------------

def test_estimate_bpm_from_beat_times() -> None:
    beats = tuple(i * 0.5 for i in range(8))   # 120 bpm
    assert abs(beat_this_adapter.estimate_bpm(beats) - 120.0) < 1e-6


def test_estimate_bpm_zero_for_short_input() -> None:
    assert beat_this_adapter.estimate_bpm(()) == 0.0
    assert beat_this_adapter.estimate_bpm((1.0,)) == 0.0


def test_measure_times_pass_through_downbeats() -> None:
    downbeats = (0.0, 2.0, 4.0, 6.0)
    assert beat_this_adapter.measure_times(downbeats) == downbeats


# ---------- persist_analysis round-trip -------------------------------------

@pytest.fixture
def db_with_audio(tmp_path: Path) -> tuple[Path, int]:
    """Build a DB with one track_audio row and return (db_path, track_audio_id)."""
    path = tmp_path / "analysis.db"
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA_PATH.read_text())
    conn.execute(
        "INSERT INTO dj_sets (set_id, set_url, title) VALUES ('S1', 'x', 't')",
    )
    conn.execute(
        "INSERT INTO dj_set_track_media_links (set_id, track_id, platform, player_id) "
        "VALUES ('S1', 'T1', 'youtube', 'vid-1')",
    )
    conn.commit()
    conn.close()

    asset = AudioAsset(
        track_audio_id=None, track_id="T1", platform="youtube",
        source_url="https://www.youtube.com/watch?v=vid-1", player_id="vid-1",
        path="/tmp/t.wav", sha256=None, duration_s=120.0, sample_rate=44100,
        codec="wav", bitrate_kbps=None,
    )
    r = db_adapter.insert_audio(path, asset)
    assert isinstance(r, Ok)
    return path, r.value


def _fake_result(tid: int) -> TrackAnalysisResult:
    emb = np.arange(24, dtype=np.float16).reshape(2, 12).tobytes()
    return TrackAnalysisResult(
        track_audio_id=tid,
        stems=StemSet(
            track_audio_id=tid,
            stems=(
                StemAsset(tid, "vocals", "/tmp/v.wav", "wav"),
                StemAsset(tid, "drums", "/tmp/d.wav", "wav"),
                StemAsset(tid, "bass", "/tmp/b.wav", "wav"),
                StemAsset(tid, "other", "/tmp/o.wav", "wav"),
            ),
        ),
        beats=BeatGrid(
            track_audio_id=tid,
            beat_times=(0.0, 0.5, 1.0, 1.5),
            downbeat_times=(0.0, 2.0),
            measure_times=(0.0, 2.0),
            bpm=120.0,
        ),
        cues=CuePoints(track_audio_id=tid, cue_times=(10.0, 60.0), model_version="v1"),
        loudness=LoudnessReading(track_audio_id=tid, integrated_lufs=-14.2),
        sections=(
            SectionEmbedding(tid, 0, 0.0, 10.0, 2, 12, "float16", emb),
            SectionEmbedding(tid, 1, 10.0, 60.0, 2, 12, "float16", emb),
        ),
        analyzer_versions={"demucs": "htdemucs_ft", "beat_this": "final0"},
    )


def test_persist_analysis_writes_all_four_tables(db_with_audio: tuple[Path, int]) -> None:
    path, tid = db_with_audio
    r = db_adapter.persist_analysis(path, _fake_result(tid))
    assert isinstance(r, Ok)

    conn = sqlite3.connect(path)
    stems = conn.execute(
        "SELECT stem_name, path FROM track_stems WHERE track_audio_id = ? ORDER BY stem_name",
        (tid,),
    ).fetchall()
    assert [s[0] for s in stems] == ["bass", "drums", "other", "vocals"]

    ta = conn.execute(
        "SELECT cue_points_json, measure_times_json FROM track_analysis WHERE track_audio_id = ?",
        (tid,),
    ).fetchone()
    assert ta[0] == "[10.0, 60.0]"
    assert ta[1] == "[0.0, 2.0]"

    feat = conn.execute(
        "SELECT bpm, lufs FROM track_audio_features WHERE track_audio_id = ?",
        (tid,),
    ).fetchone()
    assert feat == (120.0, -14.2)

    sections = conn.execute(
        "SELECT section_idx, n_frames, dim, dtype FROM track_mert_sections "
        "WHERE track_audio_id = ? ORDER BY section_idx",
        (tid,),
    ).fetchall()
    assert sections == [(0, 2, 12, "float16"), (1, 2, 12, "float16")]
    conn.close()


def test_persist_analysis_replaces_prior_sections(db_with_audio: tuple[Path, int]) -> None:
    """Re-running analysis should not leave stale MERT rows for the track."""
    path, tid = db_with_audio
    first = _fake_result(tid)
    assert isinstance(db_adapter.persist_analysis(path, first), Ok)

    # Re-run with a single-section result — the prior two rows must be gone.
    emb = np.zeros((1, 12), dtype=np.float16).tobytes()
    replay = TrackAnalysisResult(
        track_audio_id=tid,
        stems=first.stems,
        beats=first.beats,
        cues=CuePoints(track_audio_id=tid, cue_times=(), model_version="v1"),
        loudness=first.loudness,
        sections=(SectionEmbedding(tid, 0, 0.0, 120.0, 1, 12, "float16", emb),),
        analyzer_versions=first.analyzer_versions,
    )
    assert isinstance(db_adapter.persist_analysis(path, replay), Ok)

    conn = sqlite3.connect(path)
    rows = conn.execute(
        "SELECT section_idx FROM track_mert_sections WHERE track_audio_id = ?",
        (tid,),
    ).fetchall()
    conn.close()
    assert [r[0] for r in rows] == [0]
