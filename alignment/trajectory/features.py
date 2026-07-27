"""Pooled, stem-routed features for trajectory training.

Reuses the existing on-disk caches (`.feat_cache`: chroma/HuBERT via
`path_decode._ensure_feat`) and adds a cached mel-magnitude in the
`recon_probe` parameterization (the reconstruction-loss space). Everything
is pooled onto a common `bin_s` raster: bin k covers [k*bin_s, (k+1)*bin_s)
regardless of the source frame rate, so match features and mel line up
frame-for-frame.

Stem routing follows the axis doctrine (workspace CLAUDE.md): acappella ->
HuBERT on mix_vocals vs the ref vocal stem (key-invariant), instrumental ->
chroma on mix_instrumental vs the ref instrumental stem, regular -> chroma
on the full mix vs the full ref.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from labeling.acquire.audio_index import has_audio_index, load_audio_index, lookup_ref
from alignment.continuity_refine import _FEAT_CACHE
from alignment.path_decode import _ensure_feat
from alignment.recon_probe import _melmag, find_mix_stems
from alignment.recon_probe import SR as MEL_SR
from alignment.refine_ref_offsets import _STEM_FILE, HOP, SR
from alignment.stem_resolve import resolve_stem

BIN_S = 0.5
HUBERT_LAYER = 9
FPS_MATCH = SR / HOP  # 43.07 — the chroma/HuBERT grid
FPS_MEL = MEL_SR / 512  # 31.25 — the recon_probe mel grid

_MATCH_FEATURE = {"regular": "chroma", "acappella": "hubert", "instrumental": "chroma"}
FEAT_KIND = {"chroma": 0, "hubert": 1}


@dataclass(frozen=True)
class SpanAudio:
    """Resolved audio sources for one GT span."""

    mix_path: Path
    ref_path: Path
    match_feature: str  # chroma | hubert
    stem_routed: bool


def resolve_span_audio(
    aligning_dir: Path,
    by_tid: dict[str, dict],
    mix_stems: dict[str, Path],
    mix_full_path: Path,
    row: dict,
) -> SpanAudio | str:
    """SpanAudio, or a skip-reason string (loudly reported, never zero-filled)."""
    tid = str(row.get("track_id") or "")
    t = by_tid.get(tid)
    if t is None:
        return f"track_id {tid or '<none>'} not in manifest"
    if has_audio_index(aligning_dir):
        indexed = lookup_ref(load_audio_index(aligning_dir), t.get("track_audio_id"))
        if indexed is None:
            return (
                "audio_index has no ref for "
                f"track_audio_id={t.get('track_audio_id', '<none>')}"
            )
        local = str(indexed)
    else:
        local = t.get("local_path") or ""
    if not local or not Path(local).exists():
        return f"ref audio missing: {local or '<no local_path>'}"
    cs = str(row.get("claimed_stem") or "regular")
    stem_key = _STEM_FILE.get(cs)
    if stem_key and stem_key in mix_stems:
        sp = resolve_stem(aligning_dir, row.get("slot_label"), t, stem_key)
        if sp is not None:
            return SpanAudio(
                mix_path=mix_stems[stem_key],
                ref_path=sp,
                match_feature=_MATCH_FEATURE[cs],
                stem_routed=True,
            )
    return SpanAudio(
        mix_path=Path(mix_full_path),
        ref_path=Path(local),
        match_feature="chroma",
        stem_routed=False,
    )


# --- mel cache (same pattern as the chroma/hubert cache) -------------------


def _mel_cache_path(audio_path: Path | str) -> Path:
    key = hashlib.md5(str(audio_path).encode()).hexdigest()[:16]
    return _FEAT_CACHE / f"{key}_mel64.npy"


def ensure_mel(audio_path: Path | str) -> Path:
    cf = _mel_cache_path(audio_path)
    if cf.is_file():
        return cf
    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(str(audio_path), sr=MEL_SR, mono=True)
    m = _melmag(y)
    if m is None:
        raise ValueError(f"audio too short for mel: {audio_path}")
    _FEAT_CACHE.mkdir(parents=True, exist_ok=True)
    np.save(cf, m.astype(np.float32))
    return cf


# --- pooling ----------------------------------------------------------------


def pool_bins(f: np.ndarray, fps: float, bin_s: float) -> np.ndarray:
    """(D, T) frames at `fps` -> (Tb, D) mean-pooled bins, L2-normed rows."""
    n = f.shape[1]
    tb = int(n / (fps * bin_s))
    if tb < 1:
        return np.zeros((0, f.shape[0]), dtype=np.float32)
    edges = np.minimum(np.round(np.arange(tb + 1) * fps * bin_s).astype(int), n)
    sums = np.add.reduceat(np.asarray(f, dtype=np.float32), edges[:-1], axis=1)
    counts = np.maximum(np.diff(edges), 1)
    out = (sums / counts).T
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (out / norms).astype(np.float32)


class FeatureBank:
    """In-process cache of pooled features keyed by (path, feature)."""

    def __init__(self, bin_s: float = BIN_S) -> None:
        self.bin_s = bin_s
        self._pooled: dict[tuple[str, str], np.ndarray] = {}

    def match(self, audio_path: Path | str, feature: str) -> np.ndarray:
        """(Tb, D) pooled chroma or HuBERT for the whole file."""
        key = (str(audio_path), feature)
        if key not in self._pooled:
            p = _ensure_feat(audio_path, str(audio_path), feature, HUBERT_LAYER)
            f = np.load(p, mmap_mode="r")
            self._pooled[key] = pool_bins(f, FPS_MATCH, self.bin_s)
        return self._pooled[key]

    def mel(self, audio_path: Path | str) -> np.ndarray:
        """(Tb, 64) pooled mel-magnitude for the whole file."""
        key = (str(audio_path), "mel64")
        if key not in self._pooled:
            f = np.load(ensure_mel(audio_path), mmap_mode="r")
            self._pooled[key] = pool_bins(f, FPS_MEL, self.bin_s)
        return self._pooled[key]


__all__ = [
    "BIN_S",
    "FEAT_KIND",
    "FPS_MATCH",
    "FPS_MEL",
    "HUBERT_LAYER",
    "FeatureBank",
    "SpanAudio",
    "ensure_mel",
    "find_mix_stems",
    "pool_bins",
    "resolve_span_audio",
]
