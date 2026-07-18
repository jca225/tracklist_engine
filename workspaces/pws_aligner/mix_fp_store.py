"""Persistent per-mix landmark-fingerprint file cache.

One compact ``{cache_root}/{key}.fp`` blob per mix (``key`` = set_audio_id).
Removes the ~10 min/set live recompute: the corpus harvest reads the cached
fingerprint instead of running the full-mix STFT. Builds are memory-bounded
(streaming) so warming the cache is safe even on the pis.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from workspaces.alignment_prototype.landmark_fp import (
    LandmarkFingerprint,
    fingerprint_from_file_streaming,
)


def load_or_build(
    cache_root: str | Path,
    key: str,
    mix_path: str | Path,
    *,
    chunk_s: float = 120.0,
    overlap_s: float = 3.0,
    build: Callable[..., LandmarkFingerprint] | None = None,
) -> LandmarkFingerprint:
    """Return the fingerprint for ``mix_path``, reading ``{cache_root}/{key}.fp``
    if present (rebuilding on a corrupt blob), else building (streaming) and
    persisting atomically.
    """
    root = Path(cache_root)
    cache_file = root / f"{key}.fp"
    if cache_file.is_file() and cache_file.stat().st_size > 0:
        try:
            return LandmarkFingerprint.from_blob(cache_file.read_bytes())
        except Exception:
            pass  # corrupt/incompatible → rebuild below

    builder = build or fingerprint_from_file_streaming
    fp = builder(mix_path, chunk_s=chunk_s, overlap_s=overlap_s)

    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f"{key}.fp.tmp"
    tmp.write_bytes(fp.to_blob())
    os.replace(tmp, cache_file)
    return fp
