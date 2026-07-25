"""Brick 8 — the canonical backfill runner over a tiny fixture slice.

GPU/network-free: pi pull and the mert_store cache are faked, but the
fingerprint compute, the registry plumbing, and the laws checker are the real
thing. The acceptance properties: one CENTRAL store per run root, every
feature artifact derives from its audio artifact, re-running duplicates
neither artifacts nor process specs (append-only run history only), and the
executable §16 laws hold on the resulting store.
"""

from __future__ import annotations

import json
import wave

import numpy as np
import pytest

from core.provenance import LawVerdict, check_laws
from core.provenance.store import ArtifactStore, connect
from workspaces.alignment_prototype import producers, provenance_backfill
from workspaces.alignment_prototype.provenance_backfill import (
    cached_recording_ids,
    run_backfill,
)

SET_ID = "fakeset"
RECORDINGS = ("rec_a", "rec_b")


def _tiny_wav(path, freq_hz: float, dur_s: float = 3.0, sr: int = 22050) -> None:
    t = np.arange(int(dur_s * sr)) / sr
    rng = np.random.default_rng(int(freq_hz))
    y = (
        0.3 * np.sin(2 * np.pi * freq_hz * t)
        + 0.15 * np.sin(2 * np.pi * 3.1 * freq_hz * t)
        + 0.02 * rng.standard_normal(t.size)
    )
    pcm = np.clip(y * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


@pytest.fixture
def fake_world(tmp_path, monkeypatch):
    """Fake mert cache (2 recordings) + fake pi pull writing distinct wavs."""
    cache = tmp_path / f"{SET_ID}_mert.npz"
    arrays = {}
    for i, rid in enumerate(RECORDINGS):
        arrays[f"ref_{rid}_start"] = np.arange(3, dtype=np.float64)
        arrays[f"ref_{rid}_end"] = np.arange(3, dtype=np.float64) + 1.0
        arrays[f"ref_{rid}_vec"] = np.full((3, 8), 0.25 * (i + 1), dtype=np.float32)
    np.savez(
        cache,
        set_audio_id=np.int64(1),
        mix_start=np.zeros(2),
        mix_end=np.ones(2),
        mix_vec=np.zeros((2, 8), dtype=np.float32),
        ref_ids=json.dumps(list(RECORDINGS)),
        **arrays,
    )
    monkeypatch.setattr(producers, "_cache_path", lambda set_id: cache)

    freqs = {rid: 300.0 + 140.0 * i for i, rid in enumerate(RECORDINGS)}

    def fake_pull(recording_id, dest_dir):
        dest_dir.mkdir(parents=True, exist_ok=True)
        p = dest_dir / f"{recording_id}.wav"
        if not p.is_file():
            _tiny_wav(p, freqs[recording_id])
        return p

    db_root = tmp_path / "central"
    return db_root, fake_pull


def _backfill(db_root, fake_pull, limit=2):
    return run_backfill(
        SET_ID,
        limit,
        db_root,
        pull=fake_pull,
        code_commit="test0000",
        run_hubert=False,
        run_whisper=False,
    )


def test_cached_recording_ids_reads_the_mert_cache(fake_world):
    assert cached_recording_ids(SET_ID) == sorted(RECORDINGS)


def test_backfill_builds_central_store_with_derived_features(fake_world):
    db_root, fake_pull = fake_world
    summary = _backfill(db_root, fake_pull)
    assert [t.recording_id for t in summary.tracks] == sorted(RECORDINGS)
    assert summary.skipped == []
    assert (db_root / "provenance.db").is_file()

    conn = connect(db_root)
    store = ArtifactStore(db_root, conn)
    for track in summary.tracks:
        assert track.mert is not None
        assert track.audio.kind.value == "audio"
        assert track.fingerprint.kind == "landmark_fingerprint"
        assert track.mert.kind == "mert_features"
        for art in (track.audio, track.fingerprint, track.mert):
            assert store.verify(art.content_sha256)
        # Each feature artifact derives from ITS audio artifact.
        for feat, relation in (
            (track.fingerprint, "analyze.landmark_fp"),
            (track.mert, "migrated_from_cache"),
        ):
            der = conn.execute(
                "SELECT parent_sha256, relation FROM derivation WHERE child_sha256=?",
                (feat.content_sha256,),
            ).fetchall()
            assert [(d["parent_sha256"], d["relation"]) for d in der] == [
                (track.audio.content_sha256, relation)
            ]
    # ONE central DB: both recordings' rows live in the same store.
    n_audio = conn.execute(
        "SELECT COUNT(*) FROM artifact WHERE kind='audio'"
    ).fetchone()[0]
    assert n_audio == len(RECORDINGS)


def test_rerun_is_idempotent_no_duplicate_artifacts_or_specs(fake_world):
    db_root, fake_pull = fake_world
    _backfill(db_root, fake_pull)
    conn = connect(db_root)

    def counts():
        return {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]  # noqa: S608
            for t in (
                "artifact",
                "process_spec",
                "parameter_set",
                "environment_spec",
                "run",
            )
        }

    first = counts()
    _backfill(db_root, fake_pull)
    second = counts()
    # Content addressing + deterministic ids: artifacts and specs never dup.
    for table in ("artifact", "process_spec", "parameter_set", "environment_spec"):
        assert second[table] == first[table], table
    # Run history is append-only — the re-run IS recorded.
    assert second["run"] == 2 * first["run"]


def test_laws_pass_on_the_backfilled_store(fake_world):
    db_root, fake_pull = fake_world
    _backfill(db_root, fake_pull)
    conn = connect(db_root)
    store = ArtifactStore(db_root, conn)
    results = check_laws(conn, store)
    violated = [r for r in results if r.verdict is LawVerdict.VIOLATED]
    assert violated == [], [f"{r.number}: {r.offenders}" for r in violated]
    # Laws 1-3 + 14 actually CHECKED something (not vacuously N/A): the store
    # holds registered-kind feature artifacts with versioned producing runs.
    checked = {r.number: r.verdict for r in results}
    for n in (1, 2, 3, 14):
        assert checked[n] is LawVerdict.PASS, (n, results)


def test_unresolvable_audio_is_skipped_not_fabricated(fake_world, tmp_path):
    db_root, _ = fake_world

    def no_audio(recording_id, dest_dir):
        return None

    summary = run_backfill(
        SET_ID,
        2,
        db_root,
        pull=no_audio,
        code_commit="test0000",
        run_hubert=False,
        run_whisper=False,
    )
    assert summary.tracks == []
    assert summary.skipped == sorted(RECORDINGS)
    conn = connect(db_root)
    assert conn.execute("SELECT COUNT(*) FROM artifact").fetchone()[0] == 0


def test_missing_mert_cache_is_an_empty_backfill(tmp_path, monkeypatch):
    monkeypatch.setattr(
        producers, "_cache_path", lambda set_id: tmp_path / "absent.npz"
    )
    summary = run_backfill(
        "noset",
        2,
        tmp_path / "central",
        pull=lambda rid, d: pytest.fail("must not pull without a cache"),
        code_commit="test0000",
        run_hubert=False,
        run_whisper=False,
    )
    assert summary.tracks == [] and summary.skipped == []


def test_default_db_root_is_the_single_central_store():
    assert provenance_backfill._CENTRAL_ROOT.name == "central"
    assert provenance_backfill.DEFAULT_SET_ID == "1fsnxchk"
