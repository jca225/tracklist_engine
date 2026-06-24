"""Download SoundCloud tracks + MERT layer-6 summary → user taste prior vectors."""
from __future__ import annotations

import json
import logging
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from core.result import Err, Ok
from analysis.adapters import audio_io, mert_adapter
from analysis.adapters.mert_adapter import MERT_DEFAULT_LAYER, MERT_MODEL
from personalization.persistence import (
    clean_user_ids,
    get_track_mert,
    upsert_track_mert,
    upsert_user_prior,
)

logger = logging.getLogger(__name__)

MERT_DIM = 1024
MAX_TRACK_S = 480.0  # cap long mixes for prior embedding


@dataclass(frozen=True)
class TrackRef:
    sc_track_id: int
    url: str


def _yt_dlp() -> str:
    import shutil
    root = Path(__file__).resolve().parents[1]
    for candidate in (
        root / "venvs" / "audio" / "bin" / "yt-dlp",
        Path("/venv/main/bin/yt-dlp"),          # Vast PyTorch image
    ):
        if candidate.is_file():
            return str(candidate)
    found = shutil.which("yt-dlp")
    return found if found else "yt-dlp"


def sc_track_url(raw_json: str) -> str | None:
    rec = json.loads(raw_json)
    permalink = rec.get("track_permalink")
    artist = rec.get("track_artist_permalink") or rec.get("track_artist_username")
    if permalink and artist:
        return f"https://soundcloud.com/{artist}/{permalink}"
    return None


def pick_track_refs(conn: sqlite3.Connection, mix_id: str, *, limit: int) -> list[TrackRef]:
    """Most-liked SC track IDs among clean cohort users."""
    rows = conn.execute(
        """
        SELECT sl.track_id, sl.raw_json, COUNT(*) AS n
        FROM sc_likes sl
        JOIN listeners l ON l.user_id = sl.user_id
        LEFT JOIN listener_bot_scores b ON b.user_id = l.user_id
        WHERE sl.mix_id = ? AND COALESCE(b.is_bot, 0) = 0
        GROUP BY sl.track_id
        ORDER BY n DESC
        LIMIT ?
        """,
        (mix_id, limit),
    ).fetchall()
    out: list[TrackRef] = []
    for row in rows:
        url = sc_track_url(str(row["raw_json"]))
        if url:
            out.append(TrackRef(sc_track_id=int(row["track_id"]), url=url))
    return out


def download_track(url: str, dest_dir: Path) -> Path | None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(dest_dir / "%(id)s.%(ext)s")
    cmd = [
        _yt_dlp(),
        "-x",
        "--audio-format",
        "m4a",
        "--no-playlist",
        "-o",
        outtmpl,
        url,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning("download failed url=%s err=%s", url, e)
        return None
    files = sorted(dest_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def embed_layer6_mean(h, samples_24k: np.ndarray) -> np.ndarray | None:
    """Mean-pool MERT layer-6 over full track (chunked)."""
    import torch

    chunk_size = int(mert_adapter.MERT_CHUNK_S * mert_adapter.MERT_SR)
    pieces: list[np.ndarray] = []
    for i in range(0, samples_24k.size, chunk_size):
        chunk = samples_24k[i : i + chunk_size]
        if chunk.size < mert_adapter.MERT_SR // 10:
            continue
        inputs = h._processor(chunk, sampling_rate=mert_adapter.MERT_SR, return_tensors="pt")
        inputs = {k: v.to(h.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = h._model(**inputs, output_hidden_states=True)
        hidden = out.hidden_states[MERT_DEFAULT_LAYER].squeeze(0).to("cpu").numpy()
        pieces.append(hidden)
    if not pieces:
        return None
    arr = np.concatenate(pieces, axis=0)
    return arr.mean(axis=0).astype(np.float32)


def cache_track_embeddings(
    conn: sqlite3.Connection,
    refs: list[TrackRef],
    *,
    cache_dir: Path,
    device: str = "auto",
    mert_version: str = MERT_MODEL,
) -> dict[str, int]:
    """Download + MERT embed tracks not yet cached."""
    load = mert_adapter.load(device=device)
    match load:
        case Err(e):
            return {"error": 1, "detail": e.detail, "cached": 0, "downloaded": 0, "failed": 0, "skipped": 0}
        case Ok(h):
            stats = {"cached": 0, "downloaded": 0, "failed": 0, "skipped": 0}
            for ref in refs:
                if get_track_mert(conn, ref.sc_track_id, mert_version) is not None:
                    stats["skipped"] += 1
                    continue
                audio_path = download_track(ref.url, cache_dir / str(ref.sc_track_id))
                if audio_path is None:
                    stats["failed"] += 1
                    continue
                stats["downloaded"] += 1
                match audio_io.load_mono(audio_path, target_sr=mert_adapter.MERT_SR):
                    case Err(_):
                        stats["failed"] += 1
                        continue
                    case Ok(wf):
                        samples = wf.samples
                max_samples = int(MAX_TRACK_S * mert_adapter.MERT_SR)
                if samples.size > max_samples:
                    samples = samples[:max_samples]
                vec = embed_layer6_mean(h, samples)
                if vec is None:
                    stats["failed"] += 1
                    continue
                upsert_track_mert(
                    conn,
                    sc_track_id=ref.sc_track_id,
                    mert_version=mert_version,
                    embedding=vec.astype(np.float16).tobytes(),
                    dim=MERT_DIM,
                    source_url=ref.url,
                )
                stats["cached"] += 1
            return stats
    return {"cached": 0, "downloaded": 0, "failed": 0, "skipped": 0}


def _user_liked_track_ids(conn: sqlite3.Connection, user_id: str) -> set[int]:
    rows = conn.execute("SELECT track_id FROM sc_likes WHERE user_id = ?", (user_id,)).fetchall()
    ids = {int(r["track_id"]) for r in rows}
    for row in conn.execute("SELECT track_ids_json FROM sc_playlists WHERE user_id = ?", (user_id,)):
        try:
            ids.update(int(t) for t in json.loads(row["track_ids_json"]) if t is not None)
        except json.JSONDecodeError:
            pass
    return ids


def build_user_priors(
    conn: sqlite3.Connection,
    mix_id: str,
    *,
    max_users: int = 100,
    exclude_bots: bool = True,
    mert_version: str = MERT_MODEL,
) -> dict[str, int]:
    """Average cached track MERT vectors per user."""
    user_ids = clean_user_ids(conn, mix_id, exclude_bots=exclude_bots)
    # prefer users with more likes
    scored: list[tuple[int, str]] = []
    for uid in user_ids:
        n = conn.execute("SELECT COUNT(*) FROM sc_likes WHERE user_id = ?", (uid,)).fetchone()[0]
        scored.append((int(n), uid))
    scored.sort(reverse=True)
    targets = [uid for _, uid in scored[:max_users]]

    built = 0
    skipped = 0
    for uid in targets:
        track_ids = _user_liked_track_ids(conn, uid)
        vecs: list[np.ndarray] = []
        for tid in track_ids:
            blob = get_track_mert(conn, tid, mert_version)
            if blob is None:
                continue
            vecs.append(np.frombuffer(blob, dtype=np.float16).astype(np.float32))
        if len(vecs) < 3:
            skipped += 1
            continue
        mean = np.mean(np.stack(vecs, axis=0), axis=0)
        norm = float(np.linalg.norm(mean))
        if norm > 0:
            mean = mean / norm
        sc_uid = conn.execute(
            "SELECT sc_user_id FROM listeners WHERE user_id = ?", (uid,)
        ).fetchone()
        upsert_user_prior(
            conn,
            user_id=uid,
            mix_id=mix_id,
            sc_user_id=int(sc_uid["sc_user_id"]) if sc_uid and sc_uid["sc_user_id"] else None,
            mert_version=mert_version,
            dim=MERT_DIM,
            n_tracks_used=len(vecs),
            embedding=mean.astype(np.float16).tobytes(),
        )
        built += 1
    return {"users_built": built, "users_skipped": skipped, "users_targeted": len(targets)}


def run_prior_pipeline(
    conn: sqlite3.Connection,
    mix_id: str,
    cache_dir: Path,
    *,
    max_tracks: int = 150,
    max_users: int = 100,
    device: str = "auto",
) -> dict[str, object]:
    refs = pick_track_refs(conn, mix_id, limit=max_tracks)
    cache_stats = cache_track_embeddings(conn, refs, cache_dir=cache_dir, device=device)
    prior_stats = build_user_priors(conn, mix_id, max_users=max_users)
    return {"track_cache": cache_stats, "user_priors": prior_stats}
