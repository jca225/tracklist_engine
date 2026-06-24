"""Mix-side landmark matching → placement scores and ``set_fingerprint_hits`` rows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .landmark_fp import LandmarkFingerprint, SR, fp_offset

HIT_MIN_VOTES = 25
HIT_MIN_SHARPNESS = 1.2


@dataclass(frozen=True)
class MixFpHit:
    mix_start_s: float
    mix_end_s: float
    recording_id: str
    stem: str
    score: float
    votes: int
    sharpness: float


def score_mix_window(
    mix_y: np.ndarray,
    *,
    ref_fp: LandmarkFingerprint,
    ref_y: np.ndarray | None = None,
    stretches: tuple[float, ...] = (0.98, 1.0, 1.02),
) -> tuple[int, float, float]:
    """Return (votes, sharpness, stretch) for one mix excerpt vs one ref."""
    _off, votes, _st, sharp = fp_offset(
        mix_y,
        ref_y,
        ref_fp=ref_fp,
        stretches=stretches,
    )
    return votes, sharp, _st


def scan_band(
    mix_y: np.ndarray,
    *,
    ref_fp: LandmarkFingerprint,
    ref_y: np.ndarray | None,
    lo_s: float,
    hi_s: float,
    win_s: float,
    step_s: float,
    recording_id: str,
    stem: str,
    stretches: tuple[float, ...] = (0.98, 1.0, 1.02),
) -> tuple[MixFpHit, ...]:
    """Slide a fixed window; emit hits where landmark evidence is peaked."""
    dur = len(mix_y) / SR
    lo_s = max(0.0, lo_s)
    hi_s = min(dur, hi_s)
    if hi_s - lo_s < win_s * 0.5:
        return ()

    scores: list[tuple[float, int, float, float]] = []
    t = lo_s
    while t + win_s <= hi_s + 1e-6:
        i0 = int(t * SR)
        i1 = int(min((t + win_s) * SR, len(mix_y)))
        chunk = mix_y[i0:i1]
        if len(chunk) < SR // 2:
            break
        votes, sharp, _st = score_mix_window(
            chunk, ref_fp=ref_fp, ref_y=ref_y, stretches=stretches
        )
        scores.append((t, votes, sharp, win_s))
        t += step_s

    if not scores:
        return ()

    vote_arr = np.array([s[1] for s in scores], dtype=np.float64)
    sharp_arr = np.array([s[2] for s in scores], dtype=np.float64)
    # z-score sharpness relative to the band (peak/second selector from fine_placement_plan)
    mu, sig = sharp_arr.mean(), sharp_arr.std() + 1e-9
    z = (sharp_arr - mu) / sig

    hits: list[MixFpHit] = []
    for (start, votes, sharp, w), zz in zip(scores, z):
        if votes < HIT_MIN_VOTES or sharp < HIT_MIN_SHARPNESS:
            continue
        if zz < 1.0:
            continue
        hits.append(
            MixFpHit(
                mix_start_s=start,
                mix_end_s=start + w,
                recording_id=recording_id,
                stem=stem,
                score=float(zz),
                votes=int(votes),
                sharpness=float(sharp),
            )
        )
    return tuple(hits)


def placement_curve(
    mix_y: np.ndarray,
    *,
    ref_fp: LandmarkFingerprint,
    ref_y: np.ndarray | None,
    measure_mid_s: np.ndarray,
    coarse_start_s: float,
    band_s: float,
    win_s: float = 12.0,
    stretches: tuple[float, ...] = (0.98, 1.0, 1.02),
) -> np.ndarray:
    """Per-measure placement emission scores aligned to ``measure_mid_s``.

    Returns (T,) float64; invalid starts masked to -1e18 (sequence_decode convention).
    """
    from .sequence_decode import NEG

    t = measure_mid_s
    lo = max(0.0, coarse_start_s - band_s)
    hi = min(len(mix_y) / SR, coarse_start_s + band_s + win_s)
    step = max(0.5, float(np.median(np.diff(t))) if len(t) > 1 else 2.0)

    grid_t: list[float] = []
    grid_v: list[float] = []
    cur = lo
    while cur + win_s <= hi + 1e-6:
        i0 = int(cur * SR)
        i1 = int(min((cur + win_s) * SR, len(mix_y)))
        chunk = mix_y[i0:i1]
        if len(chunk) >= SR // 2:
            votes, sharp, _ = score_mix_window(
                chunk, ref_fp=ref_fp, ref_y=ref_y, stretches=stretches
            )
            grid_t.append(cur)
            grid_v.append(float(votes) * float(sharp))
        cur += step

    curve = np.full(len(t), NEG, dtype=np.float64)
    if not grid_t:
        return curve

    gt = np.asarray(grid_t)
    gv = np.asarray(grid_v)
    for i, mid in enumerate(t):
        if mid < lo or mid > hi:
            continue
        j = int(np.argmin(np.abs(gt - mid)))
        curve[i] = gv[j]
    return curve


def load_mix_mono(path: Path, *, sr: int = SR) -> np.ndarray:
    import librosa

    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y
