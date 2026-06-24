"""Persist and load landmark fingerprints (``track_fingerprints`` + local cache)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from core.db import connect
from core.result import Err, Ok, Result

from .landmark_fp import LandmarkFingerprint, fingerprint_from_audio

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "fp_index"


@dataclass(frozen=True)
class FpKey:
    recording_id: str
    stem: str

    def cache_name(self) -> str:
        safe = self.recording_id.replace("/", "_")
        return f"{safe}__{self.stem}.landmark"


def cache_path(key: FpKey, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    return cache_dir / key.cache_name()


def load_cached(
    key: FpKey, cache_dir: Path = DEFAULT_CACHE_DIR
) -> LandmarkFingerprint | None:
    path = cache_path(key, cache_dir)
    if not path.is_file():
        return None
    return LandmarkFingerprint.from_blob(path.read_bytes())


def save_cached(
    fp: LandmarkFingerprint, key: FpKey, cache_dir: Path = DEFAULT_CACHE_DIR
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(key, cache_dir)
    path.write_bytes(fp.to_blob())
    return path


def load_db(key: FpKey, db_path: Path) -> LandmarkFingerprint | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT fingerprint FROM track_fingerprints WHERE recording_id=? AND stem=?",
            (key.recording_id, key.stem),
        ).fetchone()
    if row is None:
        return None
    return LandmarkFingerprint.from_blob(bytes(row["fingerprint"]))


def upsert_db(fp: LandmarkFingerprint, key: FpKey, db_path: Path) -> None:
    blob = fp.to_blob()
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO track_fingerprints (recording_id, stem, fingerprint, duration_s)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(recording_id, stem) DO UPDATE SET
              fingerprint=excluded.fingerprint,
              duration_s=excluded.duration_s,
              created_at=CURRENT_TIMESTAMP
            """,
            (key.recording_id, key.stem, blob, fp.duration_s),
        )
        conn.commit()


def load(
    key: FpKey, *, cache_dir: Path = DEFAULT_CACHE_DIR, db_path: Path | None = None
) -> LandmarkFingerprint | None:
    hit = load_cached(key, cache_dir)
    if hit is not None:
        return hit
    if db_path is not None:
        hit = load_db(key, db_path)
        if hit is not None:
            save_cached(hit, key, cache_dir)
            return hit
    return None


def compute_from_file(audio_path: Path) -> Result[LandmarkFingerprint, str]:
    try:
        import librosa
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, _ = librosa.load(str(audio_path), sr=22050, mono=True)
        return Ok(fingerprint_from_audio(y))
    except OSError as exc:
        return Err(str(exc))
