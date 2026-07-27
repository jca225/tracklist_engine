"""Audio → fixed-rate feature stacks compatible with MertSeries."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..mert_store import MertSeries

_SR_CHROMA = 22050
_HOP = 512
_COL_S = 1.0
_POOL = max(1, round(_COL_S * _SR_CHROMA / _HOP))

_MERT_SR = 24000
_MERT_BIN_S = 2.0
_CACHE = Path(__file__).resolve().parents[1] / ".cache" / "external_features"


def _load_mono(path: Path, sr: int) -> np.ndarray:
    import librosa

    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y


def chroma_columns(y: np.ndarray, *, sr: int = _SR_CHROMA) -> np.ndarray:
    """CQT chroma mean-pooled to ~COL_S columns, L2-normalised per column."""
    import librosa

    c = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=_HOP)
    n = c.shape[1] // _POOL
    if n < 1:
        raise ValueError("audio too short for chroma grid")
    c = c[:, : n * _POOL].reshape(12, n, _POOL).mean(2)
    return (c / (np.linalg.norm(c, axis=0, keepdims=True) + 1e-9)).astype(np.float32)


def series_from_chroma(path: Path) -> MertSeries:
    y = _load_mono(path, _SR_CHROMA)
    cols = chroma_columns(y)
    n = cols.shape[1]
    t0 = np.arange(n, dtype=np.float64) * _COL_S
    t1 = t0 + _COL_S
    return MertSeries(start_s=t0, end_s=t1, vectors=cols.T.copy())


def _cache_key(path: Path, kind: str) -> Path:
    stat = path.stat()
    digest = hashlib.sha256(
        f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{kind}".encode()
    ).hexdigest()[:16]
    _CACHE.mkdir(parents=True, exist_ok=True)
    return _CACHE / f"{digest}_{kind}.npz"


def series_from_mert(path: Path, *, device: str = "auto") -> MertSeries:
    """Layer-6 MERT vectors on fixed-width bins (no beat_this grid)."""
    cache = _cache_key(path, "mert")
    if cache.is_file():
        z = np.load(cache)
        return MertSeries(
            start_s=z["start_s"],
            end_s=z["end_s"],
            vectors=z["vectors"],
        )

    from analysis.adapters.mert_adapter import (
        MERT_DEFAULT_LAYER,
        embed_section,
        load as load_mert,
    )
    from core.result import Err, Ok

    y = _load_mono(path, _MERT_SR)
    dur = len(y) / _MERT_SR
    match load_mert(device=device):
        case Ok(handle):
            pass
        case Err(e):
            raise RuntimeError(f"MERT load failed: {e.detail}") from None

    starts: list[float] = []
    ends: list[float] = []
    vecs: list[np.ndarray] = []
    t = 0.0
    while t < dur:
        t_end = min(t + _MERT_BIN_S, dur)
        if t_end - t < 0.25:
            break
        i0 = int(t * _MERT_SR)
        i1 = int(t_end * _MERT_SR)
        match embed_section(
            handle,
            y[i0:i1],
            track_audio_id=0,
            section_idx=len(starts),
            start_s=t,
            end_s=t_end,
            layer=MERT_DEFAULT_LAYER,
        ):
            case Ok(emb):
                pass
            case Err(e):
                raise RuntimeError(f"MERT embed failed: {e.detail}") from None
        arr = np.frombuffer(emb.embedding_bytes, dtype=np.float16).reshape(
            emb.n_frames, emb.dim
        )
        vec = arr.astype(np.float32).mean(axis=0)
        starts.append(t)
        ends.append(t_end)
        vecs.append(vec)
        t = t_end

    if not vecs:
        raise ValueError(f"no MERT bins for {path}")

    series = MertSeries(
        start_s=np.asarray(starts, dtype=np.float64),
        end_s=np.asarray(ends, dtype=np.float64),
        vectors=np.stack(vecs, axis=0),
    )
    np.savez_compressed(
        cache,
        start_s=series.start_s,
        end_s=series.end_s,
        vectors=series.vectors,
    )
    return series


def build_mix_bundle(
    mix_audio: Path,
    ref_paths: dict[str, Path],
    *,
    feature_kind: str,
    device: str = "auto",
) -> tuple[MertSeries, dict[str, MertSeries]]:
    if feature_kind == "chroma":
        mix = series_from_chroma(mix_audio)
        refs = {rid: series_from_chroma(p) for rid, p in ref_paths.items()}
        return mix, refs
    if feature_kind == "mert":
        mix = series_from_mert(mix_audio, device=device)
        refs = {rid: series_from_mert(p, device=device) for rid, p in ref_paths.items()}
        return mix, refs
    raise ValueError(f"unknown feature_kind {feature_kind!r}; use chroma or mert")
