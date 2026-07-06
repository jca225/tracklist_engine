#!/usr/bin/env python3
"""Self-alignment of one audio signal: find repeated regions as (pair, lag).

Why not reuse `ref_fibers`: its HuBERT labels are validated for *scoring*
equivalence but under-detect repeats (HuBERT is blind to melodic repeats —
see ref_fibers.compute_fibers_fp docstring; the 2026-07-06 audit-v1 run found
only 26 repeat pairs across 19 BB12 acappellas). The audit and the Phase-2
DJ-loop detector both need *pairs with exact lags*, not merged label blobs:

  1. log-mel self-similarity, lag-diagonal scan  -> candidate repeat pairs
     (region [a0,a1] recurs at [a0+L, a1+L]), silence-gated on both ends,
  2. waveform xcorr around the known lag         -> sample-accurate lag +
     normalized correlation r; residual 1-r^2 splits CLONE (digital copy)
     from DISTINCT TAKE (re-sung / re-mixed rendition).

Used on the REF (Phase 1 audit: is the song's own repetition clone-tied?)
and on the MIX span (Phase 2: DJ loops are digital copies at the clone
floor; the song's natural repeats are not).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from workspaces.alignment_prototype.refine_ref_offsets import HOP, SR


@dataclass(frozen=True)
class RepeatPair:
    """Region [a0,a1) recurs at [a0+lag, a1+lag); times in seconds."""

    a0: float
    a1: float
    lag_s: float
    diag_sim: float  # mel diagonal similarity (content-sameness evidence)


def load_audio(path: str) -> np.ndarray:
    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(path, sr=SR, mono=True)
    return y


def mel_features(y: np.ndarray, ds: int = 4) -> tuple[np.ndarray, float]:
    """Whitened log-mel columns at ~FPS/ds Hz: (D, T), hz.

    Each mel band is z-scored over TIME before column L2-normalization —
    raw log-mel cosine is ~1 everywhere for content with a stable spectral
    envelope (the chroma failure mode); correlating the *deviations* makes
    the self-similarity diagonal mean sameness of content, not of timbre."""
    import librosa

    m = librosa.feature.melspectrogram(y=y, sr=SR, hop_length=HOP, n_mels=64)
    g = np.log1p(m)
    t = (g.shape[1] // ds) * ds
    g = g[:, :t].reshape(g.shape[0], -1, ds).mean(axis=2)
    g = (g - g.mean(axis=1, keepdims=True)) / (g.std(axis=1, keepdims=True) + 1e-9)
    g = g / (np.linalg.norm(g, axis=0, keepdims=True) + 1e-9)
    return g.astype(np.float32), SR / HOP / ds


def _nonsilent(y: np.ndarray, hz: float, n: int, ratio: float) -> np.ndarray:
    import librosa

    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    step = max(1, int(round((SR / HOP) / hz)))
    rms = rms[::step][:n]
    out = np.zeros(n, dtype=bool)
    floor = float(np.median(rms[rms > 0])) * ratio if (rms > 0).any() else 0.0
    out[: rms.size] = rms >= floor
    return out


def repeat_pairs(
    y: np.ndarray,
    *,
    min_lag_s: float = 2.0,
    min_repeat_s: float = 4.0,
    diag_thresh: float = 0.6,
    silence_ratio: float = 0.35,
    smooth_s: float = 2.0,
    lag_merge_s: float = 0.35,
    gap_close_s: float = 1.5,
    min_voiced_frac: float = 0.4,
) -> list[RepeatPair]:
    """Scan every lag diagonal of the mel self-similarity for sustained runs.

    Unlike ref_fibers._long_repeats this KEEPS the (region, lag) pairing —
    the audit needs to know which instance aligns to which at what lag.
    Near-duplicate detections at adjacent lags (a wide diagonal ridge) keep
    only the best-similarity lag per overlapping region.

    Silence handling: sparse vocals have breath gaps every ~1.6 s, so a
    per-frame silence gate would fragment every run below min_repeat_s.
    Gaps up to `gap_close_s` are closed; instead each detected run must be
    >= `min_voiced_frac` truly voiced on BOTH sides (silence is perfectly
    self-similar and must not form repeats)."""
    from scipy.ndimage import binary_closing, uniform_filter1d

    g, hz = mel_features(y)
    t = g.shape[1]
    if t < 8:
        return []
    voiced = _nonsilent(y, hz, t, silence_ratio)
    gate = binary_closing(voiced, structure=np.ones(max(1, int(gap_close_s * hz))))
    s = (g.T @ g).astype(np.float32)
    pw = max(1, int(smooth_s * hz))
    ml = max(2, int(min_repeat_s * hz))
    raw: list[RepeatPair] = []
    for lag in range(int(min_lag_s * hz), t - ml):
        d = uniform_filter1d(np.diagonal(s, lag).astype(np.float32), pw, mode="nearest")
        good = (d > diag_thresh) & gate[: t - lag] & gate[lag:t]
        i = 0
        while i < good.size:
            if good[i]:
                j = i
                while j < good.size and good[j]:
                    j += 1
                if (
                    j - i >= ml
                    and voiced[i:j].mean() >= min_voiced_frac
                    and voiced[i + lag : j + lag].mean() >= min_voiced_frac
                ):
                    raw.append(
                        RepeatPair(
                            round(i / hz, 3),
                            round(j / hz, 3),
                            round(lag / hz, 3),
                            round(float(d[i:j].mean()), 4),
                        )
                    )
                i = j
            else:
                i += 1
    return _dedup_pairs(raw, lag_merge_s)


def _dedup_pairs(pairs: list[RepeatPair], lag_merge_s: float) -> list[RepeatPair]:
    """A wide diagonal ridge fires on several adjacent lags — keep the
    best-similarity detection among pairs whose regions overlap and whose
    lags are within lag_merge_s."""
    kept: list[RepeatPair] = []
    for p in sorted(pairs, key=lambda q: -q.diag_sim):
        dup = any(
            abs(p.lag_s - k.lag_s) <= lag_merge_s
            and min(p.a1, k.a1) - max(p.a0, k.a0) > 0
            for k in kept
        )
        if not dup:
            kept.append(p)
    return sorted(kept, key=lambda q: (q.a0, q.lag_s))


def verify_pair(
    y: np.ndarray, p: RepeatPair, *, pad_s: float = 1.0
) -> tuple[float, float]:
    """Sample-accurate check of one repeat pair: normalized xcorr of the
    region against its lagged image, searching ± pad around the coarse lag.
    Returns (r, exact_lag_s). Per-window energy normalization makes r
    gain-invariant (residual after optimal gain = 1 - r^2)."""
    from scipy.signal import fftconvolve

    sa = y[int(p.a0 * SR) : int(p.a1 * SR)].astype(np.float64)
    b0 = max(0.0, p.a0 + p.lag_s - pad_s)
    sb = y[int(b0 * SR) : int((p.a1 + p.lag_s + pad_s) * SR)].astype(np.float64)
    n = sa.size
    if n < 8 or sb.size < n:
        return 0.0, p.lag_s
    sa = sa - sa.mean()
    na = np.linalg.norm(sa)
    if na < 1e-9:
        return 0.0, p.lag_s
    num = fftconvolve(sb, sa[::-1], mode="valid")
    e = np.concatenate([[0.0], np.cumsum(sb * sb)])
    den = np.sqrt(np.maximum(e[n:] - e[:-n], 1e-12)) * na
    r = np.abs(num / np.maximum(den, 1e-12))
    k = int(np.argmax(r))
    r_sub, d_sub = _subsample_refine(sa, sb, k)
    return max(float(r[k]), r_sub), float(b0 + (k + d_sub) / SR - p.a0)


def _subsample_refine(
    sa: np.ndarray, sb: np.ndarray, k: int, up: int = 8
) -> tuple[float, float]:
    """r at sub-sample lag resolution around integer peak k.

    A digital copy sitting at a fractional-sample offset (e.g. after
    44.1k->22.05k resampling) caps integer-lag r at ~0.94 (-9 dB) — inside
    the DISTINCT band. Upsample both windows x`up` and re-correlate over
    ±1 original sample; returns (r_best, lag_delta_in_original_samples)."""
    from scipy.signal import resample_poly

    m = sb[max(0, k - 2) : k + sa.size + 2]
    if m.size < sa.size:
        return 0.0, 0.0
    ua = resample_poly(sa, up, 1)
    ub = resample_poly(m, up, 1)
    off0 = (k - max(0, k - 2)) * up  # position of the integer peak inside ub
    na = np.linalg.norm(ua)
    best_r, best_d = 0.0, 0.0
    for d in range(-up, up + 1):
        o = off0 + d
        w = ub[o : o + ua.size]
        if w.size < ua.size:
            continue
        nb = np.linalg.norm(w)
        if nb < 1e-12:
            continue
        r = abs(float(np.dot(ua, w))) / (na * nb + 1e-12)
        if r > best_r:
            best_r, best_d = r, d / up
    return best_r, best_d


def residual_db(r: float) -> float:
    """Aligned-and-gain-fitted residual energy, in dB: 10*log10(1 - r^2)."""
    return float(10.0 * np.log10(max(1.0 - r * r, 1e-12)))
