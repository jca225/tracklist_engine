"""Tests for scripts/train_identity_verifier.py DB loading and training pipeline."""

from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from analysis.identity_data import _decode_mert, _load_recordings, _mean_pool_mert
from scripts.train_identity_verifier import main


def _create_test_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "music_database.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE work (
            work_id TEXT PRIMARY KEY,
            artists_json TEXT,
            full_name TEXT
        );
        CREATE TABLE recording (
            recording_id TEXT PRIMARY KEY,
            work_id TEXT,
            version TEXT,
            stem TEXT,
            variant TEXT,
            full_name TEXT
        );
        CREATE TABLE track_metadata (
            recording_id TEXT PRIMARY KEY,
            artists_json TEXT,
            full_name TEXT,
            title TEXT,
            version TEXT
        );
        CREATE TABLE track_audio (
            track_audio_id INTEGER PRIMARY KEY,
            recording_id TEXT,
            platform TEXT,
            is_reference INTEGER,
            downloaded_at TEXT
        );
        CREATE TABLE track_audio_features (
            track_audio_id INTEGER PRIMARY KEY,
            bpm REAL,
            key_pc INTEGER,
            key_mode TEXT,
            energy REAL,
            danceability REAL,
            instrumentalness REAL,
            lufs REAL
        );
        CREATE TABLE track_mert_measures (
            track_audio_id INTEGER,
            measure_idx INTEGER,
            start_s REAL,
            end_s REAL,
            dim INTEGER,
            dtype TEXT,
            embedding BLOB,
            PRIMARY KEY (track_audio_id, measure_idx)
        );
        CREATE TABLE track_fingerprints (
            recording_id TEXT,
            stem TEXT,
            fingerprint BLOB,
            duration_s REAL,
            PRIMARY KEY (recording_id, stem)
        );
        """
    )

    # Insert two works with enough recordings for 5-fold CV.
    conn.execute(
        "INSERT INTO work (work_id, full_name) VALUES ('w1', 'Song One'), ('w2', 'Song Two')"
    )
    recs = [
        ("r1", "w1", "original", "Artist - Song One"),
        ("r2", "w1", "remix", "Artist - Song One (Remix)"),
        ("r3", "w1", "original", "Artist - Song One"),
        ("r4", "w1", "remix", "Artist - Song One (Remix)"),
        ("r5", "w2", "original", "Artist - Song Two"),
    ]
    vals = ",".join(
        f"('{rid}', '{wid}', '{ver}', 'regular', 'regular', '{fn}')"
        for rid, wid, ver, fn in recs
    )
    conn.execute(
        f"INSERT INTO recording (recording_id, work_id, version, stem, variant, full_name) VALUES {vals}"
    )

    def _title(fn: str) -> str:
        return fn.split(" - ")[1].split(" (")[0]

    meta_vals = ",".join(
        f"('{rid}', '[\"Artist\"]', '{fn}', '{_title(fn)}', '{ver}')"
        for rid, _, ver, fn in recs
    )
    conn.execute(
        f"INSERT INTO track_metadata (recording_id, artists_json, full_name, title, version) VALUES {meta_vals}"
    )

    audio_vals = ",".join(
        f"({i + 1}, '{rid}', 'youtube', 1, '2026-01-01')"
        for i, (rid, _, _, _) in enumerate(recs)
    )
    conn.execute(
        f"INSERT INTO track_audio (track_audio_id, recording_id, platform, is_reference, downloaded_at) VALUES {audio_vals}"
    )

    feat_rows = [
        (1, 128.0, 7, "minor", 0.8, 0.7, 0.1, -8.0),
        (2, 140.0, 7, "minor", 0.6, 0.8, 0.2, -9.0),
        (3, 128.0, 7, "minor", 0.81, 0.71, 0.11, -8.1),
        (4, 140.0, 7, "minor", 0.59, 0.79, 0.21, -9.1),
        (5, 120.0, 2, "major", 0.5, 0.6, 0.3, -10.0),
    ]
    conn.executemany(
        "INSERT INTO track_audio_features (track_audio_id, bpm, key_pc, key_mode, energy, danceability, instrumentalness, lufs) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        feat_rows,
    )

    rng = np.random.default_rng(42)
    for taid in range(1, 6):
        for m in range(4):
            emb = rng.normal(size=16).astype(np.float16).tobytes()
            conn.execute(
                "INSERT INTO track_mert_measures (track_audio_id, measure_idx, start_s, end_s, dim, dtype, embedding) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (taid, m, float(m), float(m + 1), 16, "float16", emb),
            )

    for i, (rid, _, _, _) in enumerate(recs):
        fp = (
            np.random.default_rng(i + 1)
            .integers(0, 2**32, size=64, dtype=np.uint32)
            .tobytes()
        )
        conn.execute(
            "INSERT INTO track_fingerprints (recording_id, stem, fingerprint, duration_s) VALUES (?, ?, ?, ?)",
            (rid, "regular", fp, 180.0),
        )

    conn.commit()
    conn.close()
    return db_path


def test_decode_mert_roundtrip():
    arr = np.array([0.5, -0.25, 0.75], dtype=np.float16)
    decoded = _decode_mert(arr.tobytes(), 3)
    assert decoded is not None
    np.testing.assert_array_almost_equal(decoded, arr.astype(np.float32), decimal=3)


def test_mean_pool_mert():
    a = np.ones(8, dtype=np.float16).tobytes()
    b = np.zeros(8, dtype=np.float16).tobytes()
    pooled = _mean_pool_mert([(a, 8), (b, 8)])
    np.testing.assert_array_almost_equal(
        pooled, np.full(8, 0.5, dtype=np.float32), decimal=3
    )


def test_load_recordings_from_test_db(tmp_path: Path):
    db_path = _create_test_db(tmp_path)
    recordings = _load_recordings(db_path)
    assert len(recordings) == 5
    assert recordings["r1"].work_id == "w1"
    assert recordings["r1"].artist == "Artist"
    assert recordings["r1"].essentia is not None
    assert recordings["r1"].mert_mean is not None
    assert recordings["r1"].fingerprint is not None


def test_main_trains_and_saves_model(tmp_path: Path):
    db_path = _create_test_db(tmp_path)
    output = tmp_path / "model.pkl"
    assert (
        main(["--db", str(db_path), "--output", str(output), "--min-recordings", "3"])
        == 0
    )
    assert output.exists()
    with open(output, "rb") as f:
        bundle = pickle.load(f)
    assert "song" in bundle
    assert "version" in bundle
    assert "metrics" in bundle
    assert "version_cv_roc_auc" in bundle["metrics"]
