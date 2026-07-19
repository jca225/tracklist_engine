"""Offline denotational render of labeled Ableton layers → mono buffer.

This is the *evaluation* side of the ALS interpreter: given timed clips with
ref offsets / tempo ratios / gain curves, sum what the labeling session asserts
should be audible. Not Live DSP (no plugins/sends) — enough to catch GT shaping
bugs (distant-clip merges) that XML round-trip cannot see.

Kept outside the codec-core file set so it may use numpy/scipy; it must not
import project-side modules (export / workspaces).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_SR = 22_050


@dataclass(frozen=True)
class RenderClip:
    """One audible layer contribution in set-seconds."""

    audio_path: Path
    set_start_s: float
    set_end_s: float
    ref_start_s: float
    # ref seconds consumed per set second (export: ref_span / set_span).
    tempo_ratio: float
    gain_curve: tuple[tuple[float, float], ...] = ()
    pitch_shift_semi: int = 0


def resolve_clip_audio_path(path_str: str, set_dir: Path) -> Path | None:
    """Resolve an Ableton clip path against the aligning set directory."""
    raw = (path_str or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if p.is_file():
        return p
    # Absolute path from another machine — try basename under set_dir trees.
    name = p.name
    for sub in ("tracks", "stems", "online", "candidates"):
        cand = set_dir / sub / name
        if cand.is_file():
            return cand
    # Recursive basename hit (stems/<slot>/vocals.flac etc.).
    hits = list(set_dir.rglob(name))
    if len(hits) == 1 and hits[0].is_file():
        return hits[0]
    # Relative to set_dir.
    rel = set_dir / raw
    if rel.is_file():
        return rel
    return None


def load_mono(path: Path, sr: int = DEFAULT_SR) -> np.ndarray:
    """Load mono float32 audio at ``sr``. Prefers soundfile; falls back to librosa."""
    try:
        import soundfile as sf

        y, file_sr = sf.read(str(path), always_2d=False)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if int(file_sr) != sr:
            import librosa

            y = librosa.resample(y, orig_sr=int(file_sr), target_sr=sr)
        return y.astype(np.float32)
    except Exception:
        import librosa

        y, _ = librosa.load(str(path), sr=sr, mono=True)
        return np.asarray(y, dtype=np.float32)


def gain_at(
    curve: tuple[tuple[float, float], ...],
    t: float,
    *,
    default: float = 1.0,
) -> float:
    """Piecewise-linear gain at set-time ``t`` (unity if curve empty)."""
    if not curve:
        return default
    if t <= curve[0][0]:
        return float(curve[0][1])
    if t >= curve[-1][0]:
        return float(curve[-1][1])
    for i in range(len(curve) - 1):
        t0, g0 = curve[i]
        t1, g1 = curve[i + 1]
        if t0 <= t <= t1:
            if t1 <= t0:
                return float(g0)
            u = (t - t0) / (t1 - t0)
            return float(g0 + u * (g1 - g0))
    return default


def sum_render_clips(
    clips: list[RenderClip],
    *,
    sr: int = DEFAULT_SR,
    audio_cache: dict[Path, np.ndarray] | None = None,
) -> np.ndarray:
    """Sum RenderClips into a mono float32 buffer spanning [0, max set_end]."""
    if not clips:
        return np.zeros(0, dtype=np.float32)
    end_s = max(c.set_end_s for c in clips)
    if end_s <= 0:
        return np.zeros(0, dtype=np.float32)
    n = int(np.ceil(end_s * sr)) + 1
    out = np.zeros(n, dtype=np.float64)
    cache = audio_cache if audio_cache is not None else {}

    for clip in clips:
        span = clip.set_end_s - clip.set_start_s
        if span <= 0:
            continue
        path = clip.audio_path
        if path not in cache:
            cache[path] = load_mono(path, sr=sr)
        src = cache[path]
        ratio = clip.tempo_ratio if clip.tempo_ratio > 0 else 1.0
        i0 = max(0, int(np.floor(clip.set_start_s * sr)))
        i1 = min(n, int(np.ceil(clip.set_end_s * sr)))
        if i1 <= i0:
            continue
        # Vectorized sample: set-time → ref-time → source index.
        idx = np.arange(i0, i1)
        t = idx / sr
        ref_t = clip.ref_start_s + (t - clip.set_start_s) * ratio
        src_idx = ref_t * sr
        # Linear interpolate in source; out-of-range → 0.
        x0 = np.floor(src_idx).astype(np.int64)
        frac = src_idx - x0
        x1 = x0 + 1
        valid = (x0 >= 0) & (x1 < len(src))
        samp = np.zeros(len(idx), dtype=np.float64)
        if np.any(valid):
            samp[valid] = (1.0 - frac[valid]) * src[x0[valid]] + frac[valid] * src[
                x1[valid]
            ]
        if clip.gain_curve:
            gains = np.array(
                [gain_at(clip.gain_curve, float(tt)) for tt in t], dtype=np.float64
            )
            samp *= gains
        # Pitch: cheap integer-shift via resampling the source slice (optional).
        if clip.pitch_shift_semi:
            try:
                import librosa

                # Apply to the already-extracted set-rate samples.
                samp = librosa.effects.pitch_shift(
                    samp.astype(np.float32), sr=sr, n_steps=clip.pitch_shift_semi
                ).astype(np.float64)
            except Exception:
                pass
        out[i0:i1] += samp

    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1e-6:
        out = out / peak
    return out.astype(np.float32)
