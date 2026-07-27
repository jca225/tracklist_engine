#!/usr/bin/env python3
"""Phase 4 — discriminative frame weighting for repeat-instance ties.

From the Phase-1 self-alignment: for every frame of the ORIGINAL inside a
repeated region, how much do the aligned repeats differ there? Near zero =
identical chorus body (contributes nothing to an instance decision); high =
breaths / ad-libs / timing jitter — the few frames that CAN decide which
instance is playing.

Used post-decode (`reselect_instances`): the Phase-3 DP already finds the
right content far more often than the right instance (BB11 repeat-aware 60%
vs strict 38%). For each decoded segment whose song region has repeat
images, re-score every image with discriminability-WEIGHTED landmark
support only, and switch when a challenger clearly beats the incumbent.
The identical 95% of the chorus contributes ~nothing to the tie; the
discriminative 5% decides. A margin guard keeps the incumbent on ties
(then the deterministic clone tie-break rule applies upstream).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from alignment.looptrace import selfsim
from alignment.looptrace.audit import clone_images
from alignment.looptrace.config import DISCRIM_V1, DiscrimConfig

_CACHE = Path(__file__).parent / ".lm_cache"
_REPEAT = frozenset({"clone", "distinct"})


def discriminability_mask(
    ref_path: str, pairs: list[dict], cfg: DiscrimConfig = DISCRIM_V1
) -> tuple[np.ndarray, float]:
    """(mask, hz): per-frame discriminability of the reference in [0, 1].

    1 outside any repeat; inside a repeat pair's region, the whitened-mel
    cosine DISTANCE to the aligned image (min over covering pairs — a frame
    identical to any image cannot decide that tie). Cached to disk."""
    tag = hashlib.md5(f"{ref_path}|{cfg.version}|mask".encode()).hexdigest()[:16]
    f = _CACHE / f"{tag}.npz"
    if f.is_file():
        d = np.load(f)
        return d["mask"], float(d["hz"])
    g, hz = selfsim.mel_features(selfsim.load_audio(ref_path))
    t = g.shape[1]
    mask = np.ones(t, dtype=np.float32)
    for p in pairs:
        if p["class"] not in _REPEAT:
            continue
        lag = int(round(p["lag_s"] * hz))
        i0, i1 = int(p["a0"] * hz), min(int(p["a1"] * hz), t)
        for i in range(i0, i1):
            j = i + lag
            if 1 <= j < t - 1:
                # non-integer frame lags misalign transients by up to half
                # a frame (~46 ms) — take the best of ±1 frame
                sim = max(
                    float(np.dot(g[:, i], g[:, j - 1])),
                    float(np.dot(g[:, i], g[:, j])),
                    float(np.dot(g[:, i], g[:, j + 1])),
                )
                d = min(max(1.0 - sim, 0.0), 1.0)
                mask[i] = min(mask[i], d)
                mask[j] = min(mask[j], d)
    _CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(f, mask=mask, hz=hz)
    return mask, hz


def _weighted_support(
    points: np.ndarray,
    intercept: float,
    slope: float,
    m0: float,
    m1: float,
    mask: np.ndarray,
    hz: float,
    cfg: DiscrimConfig,
) -> float:
    """Discriminability-weighted inlier support of one diagonal over one
    mix window."""
    b = points[:, 1] - slope * points[:, 0]
    sel = (
        (np.abs(b - intercept) <= cfg.inlier_tol_s)
        & (points[:, 0] >= m0)
        & (points[:, 0] <= m1)
    )
    if not sel.any():
        return 0.0
    idx = np.clip((points[sel, 1] * hz).astype(np.int64), 0, mask.size - 1)
    return float(mask[idx].sum())


def reselect_instances(
    segs: list[tuple[float, float, float]],
    points: np.ndarray,
    slope: float,
    pairs: list[dict],
    ref_path: str,
    span_dur_s: float,
    cfg: DiscrimConfig = DISCRIM_V1,
) -> list[tuple[float, float, float]]:
    """For each decoded segment whose song region has repeat images, pick
    the image with the best discriminability-weighted support; switch only
    when the challenger beats the incumbent by `switch_margin`."""
    if not pairs or len(points) == 0:
        return segs
    mask, hz = discriminability_mask(ref_path, pairs, cfg)
    out: list[tuple[float, float, float]] = []
    for i, (m0, s0, s1) in enumerate(segs):
        m1 = segs[i + 1][0] if i + 1 < len(segs) else span_dur_s
        mid = (s0 + s1) / 2
        images = clone_images(mid, pairs, classes=_REPEAT, max_hops=3)
        if len(images) < 2:
            out.append((m0, s0, s1))
            continue
        b_inc = s0 - slope * m0
        best = (m0, s0, s1)
        inc_sup = _weighted_support(points, b_inc, slope, m0, m1, mask, hz, cfg)
        best_sup = inc_sup
        for img in images:
            d = img - mid
            if abs(d) < 1e-6:
                continue
            sup = _weighted_support(points, b_inc + d, slope, m0, m1, mask, hz, cfg)
            if sup > best_sup and sup > inc_sup * (1.0 + cfg.switch_margin):
                best_sup = sup
                best = (m0, round(s0 + d, 3), round(s1 + d, 3))
        out.append(best)
    return out
