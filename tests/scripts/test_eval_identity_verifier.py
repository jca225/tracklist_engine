"""Tests for scripts/eval_identity_verifier.py lab evaluation harness."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from analysis.identity_data import _load_recordings
from scripts.eval_identity_verifier import (
    DEFAULT_TEST_SETS,
    _build_test_pairs,
    _probabilistic_roc_auc,
    _recording_ids_for_set,
    _run,
    _split_recordings,
    _ternary_metrics,
    _worst_cases,
)


def _create_test_db(tmp_path: Path) -> Path:
    """Create a minimal DB with two sets and enough recordings for CV + eval.

    Layout:
      - work w1: Song One, with original and remix versions
      - work w2: Song Two, original only
      - train set (ts0): r3 (w1 original), r4 (w1 remix), r6 (w2 original), r8 (w1 remix)
      - test set  (ts1): r1 (w1 original), r2 (w1 remix), r5 (w2 original), r7 (w1 original)
    This gives both positive and negative same-song and same-version pairs in
    train and test while keeping the DB tiny.
    """
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
        CREATE TABLE set_ground_truth (
            set_id TEXT,
            recording_id TEXT,
            set_start_s REAL,
            set_end_s REAL,
            PRIMARY KEY (set_id, recording_id)
        );
        """
    )

    conn.execute(
        "INSERT INTO work (work_id, full_name) VALUES ('w1', 'Song One'), ('w2', 'Song Two')"
    )
    recs = [
        ("r1", "w1", "original", "Artist - Song One", "Song One"),
        ("r2", "w1", "remix", "Artist - Song One (Remix)", "Song One (Remix)"),
        ("r3", "w1", "original", "Artist - Song One", "Song One"),
        ("r4", "w1", "remix", "Artist - Song One (Remix)", "Song One (Remix)"),
        ("r5", "w2", "original", "Artist - Song Two", "Song Two"),
        ("r6", "w2", "original", "Artist - Song Two", "Song Two"),
        ("r7", "w1", "original", "Artist - Song One", "Song One"),
        ("r8", "w1", "remix", "Artist - Song One (Remix)", "Song One (Remix)"),
    ]
    conn.executemany(
        "INSERT INTO recording (recording_id, work_id, version, stem, variant, full_name)"
        " VALUES (?, ?, ?, 'regular', 'regular', ?)",
        [(rid, wid, ver, fn) for rid, wid, ver, fn, _ in recs],
    )
    conn.executemany(
        "INSERT INTO track_metadata (recording_id, artists_json, full_name, title, version)"
        " VALUES (?, '[\"Artist\"]', ?, ?, ?)",
        [(rid, fn, title, ver) for rid, _, ver, fn, title in recs],
    )
    conn.executemany(
        "INSERT INTO track_audio (track_audio_id, recording_id, platform, is_reference, downloaded_at)"
        " VALUES (?, ?, 'youtube', 1, '2026-01-01')",
        [(i + 1, rid) for i, (rid, _, _, _, _) in enumerate(recs)],
    )

    # Make same-version pairs similar; different-version pairs divergent.
    feat_rows = [
        (1, 128.0, 0, "major", 0.80, 0.70, 0.20, -8.0),  # w1 original test
        (2, 130.0, 0, "major", 0.82, 0.72, 0.25, -8.2),  # w1 remix test
        (3, 128.0, 0, "major", 0.81, 0.71, 0.21, -8.1),  # w1 original train
        (4, 130.0, 0, "major", 0.83, 0.73, 0.24, -8.3),  # w1 remix train
        (5, 120.0, 5, "minor", 0.60, 0.50, 0.40, -10.0),  # w2 original test
        (6, 120.0, 5, "minor", 0.61, 0.51, 0.41, -10.1),  # w2 original train
        (7, 128.0, 0, "major", 0.79, 0.69, 0.19, -8.0),  # w1 original test
        (8, 130.0, 0, "major", 0.84, 0.74, 0.26, -8.4),  # w1 remix train
    ]
    conn.executemany(
        "INSERT INTO track_audio_features VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        feat_rows,
    )

    # MERT embeddings: similar within version, distinct across works.
    dim = 4
    embeddings = {
        1: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float16),
        2: np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float16),
        3: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float16),
        4: np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float16),
        5: np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float16),
        6: np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float16),
        7: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float16),
        8: np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float16),
    }
    for taid, emb in embeddings.items():
        conn.execute(
            "INSERT INTO track_mert_measures (track_audio_id, measure_idx, start_s, end_s, dim, dtype, embedding)"
            " VALUES (?, 0, 0.0, 10.0, ?, 'float16', ?)",
            (taid, dim, emb.tobytes()),
        )

    fps = {
        "r1": np.array([0xAAAA_AAAA, 0xAAAA_AAAA], dtype=np.uint32),
        "r2": np.array([0xAAAA_AAAE, 0xAAAA_AAAA], dtype=np.uint32),
        "r3": np.array([0xAAAA_AAAA, 0xAAAA_AAAA], dtype=np.uint32),
        "r4": np.array([0xAAAA_AAAE, 0xAAAA_AAAA], dtype=np.uint32),
        "r5": np.array([0x5555_5555, 0x5555_5555], dtype=np.uint32),
        "r6": np.array([0x5555_5555, 0x5555_5555], dtype=np.uint32),
        "r7": np.array([0xAAAA_AAAA, 0xAAAA_AAAA], dtype=np.uint32),
        "r8": np.array([0xAAAA_AAAE, 0xAAAA_AAAA], dtype=np.uint32),
    }
    conn.executemany(
        "INSERT INTO track_fingerprints (recording_id, stem, fingerprint, duration_s)"
        " VALUES (?, 'regular', ?, 100.0)",
        [(rid, fp.tobytes()) for rid, fp in fps.items()],
    )

    conn.executemany(
        "INSERT INTO set_ground_truth (set_id, recording_id, set_start_s, set_end_s) VALUES (?, ?, 0.0, 100.0)",
        [
            ("ts0", "r3"),
            ("ts0", "r4"),
            ("ts0", "r6"),
            ("ts0", "r8"),
            ("ts1", "r1"),
            ("ts1", "r2"),
            ("ts1", "r5"),
            ("ts1", "r7"),
        ],
    )
    conn.commit()
    conn.close()
    return db_path


def test_recording_ids_for_set(tmp_path: Path) -> None:
    db_path = _create_test_db(tmp_path)
    assert _recording_ids_for_set(db_path, "ts1") == {"r1", "r2", "r5", "r7"}
    assert _recording_ids_for_set(db_path, "missing") == set()


def test_split_recordings(tmp_path: Path) -> None:
    db_path = _create_test_db(tmp_path)
    recordings = _load_recordings(db_path)
    test_ids = {"r1", "r2", "r5", "r7"}
    train, test = _split_recordings(recordings, test_ids)
    assert set(train.keys()) == {"r3", "r4", "r6", "r8"}
    assert set(test.keys()) == {"r1", "r2", "r5", "r7"}


def test_split_recordings_downsamples_train(tmp_path: Path) -> None:
    db_path = _create_test_db(tmp_path)
    recordings = _load_recordings(db_path)
    test_ids = {"r1", "r2", "r5", "r7"}
    train, test = _split_recordings(recordings, test_ids, max_train=2, seed=42)
    assert len(train) == 2
    assert not set(train.keys()) & test_ids
    assert set(test.keys()) == {"r1", "r2", "r5", "r7"}


def test_build_song_pairs(tmp_path: Path) -> None:
    db_path = _create_test_db(tmp_path)
    recordings = list(_load_recordings(db_path).values())
    refs, cands, labels = _build_test_pairs(recordings, "song")
    assert len(refs) == len(cands) == len(labels) > 0
    for ref, cand, label in zip(refs, cands, labels):
        if ref.work_id == cand.work_id:
            assert label == 1
        else:
            assert label == 0


def test_build_version_pairs(tmp_path: Path) -> None:
    db_path = _create_test_db(tmp_path)
    recordings = list(_load_recordings(db_path).values())
    refs, cands, labels = _build_test_pairs(recordings, "version")
    for ref, cand, label in zip(refs, cands, labels):
        assert ref.work_id == cand.work_id
        if ref.version == cand.version:
            assert label == 1
        else:
            assert label == 0


def test_ternary_metrics_perfect_predictions() -> None:
    y_true = [1, 1, 0, 0]
    y_prob = [0.9, 0.8, 0.1, 0.2]
    y_pred, metrics = _ternary_metrics(y_true, y_prob)
    assert y_pred == [1, 1, 0, 0]
    assert metrics.n_ambiguous == 0
    assert metrics.accuracy == 1.0
    assert metrics.f1 == 1.0


def test_ternary_metrics_with_ambiguous() -> None:
    y_true = [1, 0]
    y_prob = [0.55, 0.45]  # inside [0.3, 0.7] -> ambiguous
    y_pred, metrics = _ternary_metrics(y_true, y_prob)
    assert y_pred == [None, None]
    assert metrics.n_ambiguous == 2
    assert metrics.accuracy is None


def test_probabilistic_roc_auc_requires_both_classes() -> None:
    assert _probabilistic_roc_auc([1, 1], [0.9, 0.8]) is None
    assert _probabilistic_roc_auc([1, 0], [0.9, 0.1]) is not None


def test_worst_cases_selects_confident_errors() -> None:
    from scripts.eval_identity_verifier import PairResult

    results = [
        PairResult(
            "r1", "A", "r2", "A", "song", 1, 0.9, None, "same", "same"
        ),  # correct
        PairResult(
            "r1", "A", "r2", "B", "song", 0, 0.99, None, "same", "same"
        ),  # wrong, very confident
        PairResult(
            "r1", "A", "r2", "B", "song", 1, 0.05, None, "different", "different"
        ),  # wrong, confident
    ]
    worst = _worst_cases(results, "song")
    assert len(worst) == 2
    probs = {w["prob"] for w in worst}
    assert 0.99 in probs
    assert 0.05 in probs
    assert worst[0]["prob"] == 0.99


def test_run_eval_end_to_end(tmp_path: Path) -> None:
    db_path = _create_test_db(tmp_path)
    output = tmp_path / "report.json"
    rc = _run(
        db_path=db_path,
        test_set_ids=["ts1"],
        model_path=None,
        output_path=output,
        min_train_recordings=1,
        max_train_recordings=None,
    )
    assert rc == 0
    assert output.exists()


def test_default_test_sets_are_bb11_and_bb12() -> None:
    assert DEFAULT_TEST_SETS == ["2nvzlh2k", "1fsnxchk"]
