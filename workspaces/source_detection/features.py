"""Feature extraction + on-disk cache. Everything expensive (chroma, beats,
loaded audio fragments) is memoized under .cache/, keyed by file content hash +
params, so the pipeline is resumable. I/O lives here; the matcher stays pure."""
from __future__ import annotations

import hashlib
import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from . import config

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- cache
def file_key(path: Path) -> str:
    """Cheap, collision-safe content key: size + mtime + path (not a full hash —
    audio files are large and immutable in practice)."""
    st = path.stat()
    raw = f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _cache_path(kind: str, key: str) -> Path:
    d = config.CACHE_ROOT / kind
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.npy"


def _cached_npy(kind: str, key: str, compute):
    p = _cache_path(kind, key)
    if p.is_file():
        return np.load(p, allow_pickle=False)
    arr = compute()
    if arr is not None:
        tmp = p.with_suffix(".tmp.npy")
        np.save(tmp, arr)
        tmp.replace(p)
    return arr


# --------------------------------------------------------------------------- audio
def load_mono(path: Path, sr: int = config.SR) -> np.ndarray:
    import librosa
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


# --------------------------------------------------------------------------- chroma
def _chroma(y: np.ndarray) -> np.ndarray:
    """CQT chroma, L2-normalized per frame — same recipe as the proven
    refine_ref_offsets matched filter."""
    import librosa
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = librosa.feature.chroma_cqt(y=y, sr=config.SR, hop_length=config.HOP)
    return librosa.util.normalize(c, axis=0).astype(np.float32)


def chroma_of(path: Path) -> np.ndarray:
    """(12, n_frames) chroma for an audio file, cached."""
    return _cached_npy("chroma", file_key(path), lambda: _chroma(load_mono(path)))


# --------------------------------------------------------------------------- tempo
def bpm_of(path: Path) -> Optional[float]:
    """Global tempo estimate (librosa beat tracker), cached. Used to derive the
    stretch band; None when estimation is unreliable (e.g. beatless vocals)."""
    key = file_key(path)
    arr = _cached_npy("bpm", key, lambda: _bpm(load_mono(path)))
    if arr is None:
        return None
    v = float(np.asarray(arr).reshape(-1)[0])
    return v if v > 0 else None


def _bpm(y: np.ndarray) -> np.ndarray:
    import librosa
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tempo, _ = librosa.beat.beat_track(y=y, sr=config.SR, hop_length=config.HOP)
    return np.array([float(np.atleast_1d(tempo)[0])], dtype=np.float32)


def local_bpm_curve(path: Path) -> Optional[np.ndarray]:
    """Per-frame local tempo of the mix (static estimate broadcast for now —
    a placeholder for a windowed tempogram). Returns None if unavailable."""
    b = bpm_of(path)
    return None if b is None else np.array([b], dtype=np.float32)


# --------------------------------------------------------------------------- MERT
def load_mert_npz(set_id: str) -> Optional[dict]:
    """Load the alignment_prototype MERT export for a set, if present.

    Schema (per export_mert_from_pi.py): keys `mix_vec` (n_measures, 1024),
    `mix_start`/`mix_end` (n_measures,), `ref_ids`, and per-ref
    `ref_<id>_vec` / `ref_<id>_start` / `ref_<id>_end`."""
    p = config.REPO_ROOT / "workspaces" / "alignment_prototype" / ".cache" / "mert" / f"{set_id}_mert.npz"
    if not p.is_file():
        return None
    z = np.load(p, allow_pickle=True)
    out: dict = {"mix_vec": z["mix_vec"].astype(np.float32),
                 "mix_start": z["mix_start"], "mix_end": z["mix_end"], "refs": {}}
    ref_ids = str(z["ref_ids"]) if "ref_ids" in z else ""
    for rid in [r for r in ref_ids.split(",") if r]:
        k = f"ref_{rid}_vec"
        if k in z:
            out["refs"][rid] = z[k].astype(np.float32)
    if not out["refs"]:  # fall back to scraping ref_*_vec keys directly
        for k in z.files:
            if k.startswith("ref_") and k.endswith("_vec"):
                out["refs"][k[4:-4]] = z[k].astype(np.float32)
    return out
