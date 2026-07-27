#!/usr/bin/env python3
"""Phase 3 — landmark point cloud between a mix span and a reference.

Reuses the proven spectral-peak constellation (`landmark_fp.constellation`)
but replaces the fixed-frequency hash with a PITCH- AND TEMPO-TOLERANT key
(~31% of BB11 acappellas are re-pitched, which shifts every peak by up to
±8% — far beyond the legacy (f1//2, f2//2, dt) key's tolerance):

    key = ( q(log2 f1)        coarse — absolute pitch, repitch-tolerant,
            q(log2 f2/f1)     finer  — pitch-shift INVARIANT interval,
            q(log2 dt) )      log    — tempo-stretch tolerant

Matching emits RAW anchor-time pairs (t_mix, t_song) — no offset histogram.
The whole point (brief §6.5): a wrong-repeat hypothesis forms a diagonal
that dies where surrounding content diverges; the true one accumulates
collinear points across section boundaries. Keeping the raw cloud preserves
that long-range accumulation for the Hough + segment-cover DP.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from alignment.landmark_fp import (
    FPS as LFPS,
)
from alignment.landmark_fp import (
    NFFT,
    SR,
    constellation,
)
from alignment.looptrace.config import LM_V1, LandmarkConfig

_CACHE = Path(__file__).parent / ".lm_cache"


def hash_points(
    tf: np.ndarray, fb: np.ndarray, cfg: LandmarkConfig = LM_V1
) -> dict[tuple[int, int, int], np.ndarray]:
    """{key: anchor_time_frames}. Peaks must be time-sorted pairs within a
    fan-out window, like the legacy hasher, but keyed tolerance-first."""
    order = np.argsort(tf, kind="stable")
    tf, fb = tf[order], fb[order]
    hz = SR / NFFT  # Hz per STFT bin
    out: dict[tuple[int, int, int], list[int]] = {}
    n = len(tf)
    for i in range(n):
        f1 = fb[i] * hz
        if f1 < cfg.min_hz:
            continue
        for j in range(i + 1, min(i + 1 + cfg.fan, n)):
            dt = int(tf[j] - tf[i])
            if dt < 1 or dt > cfg.dt_max_frames:
                continue
            f2 = fb[j] * hz
            if f2 < cfg.min_hz:
                continue
            key = (
                int(np.log2(f1) * cfg.f1_bins_per_oct),
                int(round(np.log2(f2 / f1) * cfg.ratio_bins_per_oct)),
                int(round(np.log2(dt) * cfg.dt_bins_per_oct)),
            )
            out.setdefault(key, []).append(int(tf[i]))
    return {k: np.asarray(v, dtype=np.int32) for k, v in out.items()}


def audio_points(y: np.ndarray, cfg: LandmarkConfig = LM_V1):
    return hash_points(*constellation(y), cfg)


def ref_points_cached(
    ref_path: str, cfg: LandmarkConfig = LM_V1
) -> dict[tuple[int, int, int], np.ndarray]:
    """Hash a full reference once, cached to disk keyed by (path, config)."""
    from alignment.looptrace import selfsim

    tag = hashlib.md5(f"{ref_path}|{cfg.version}".encode()).hexdigest()[:16]
    f = _CACHE / f"{tag}.json"
    if f.is_file():
        raw = json.loads(f.read_text())
        return {
            tuple(json.loads(k)): np.asarray(v, dtype=np.int32) for k, v in raw.items()
        }
    pts = audio_points(selfsim.load_audio(ref_path), cfg)
    _CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({json.dumps(list(k)): v.tolist() for k, v in pts.items()}))
    return pts


def match_points(mix_h: dict, ref_h: dict, cfg: LandmarkConfig = LM_V1) -> np.ndarray:
    """(N, 2) array of matched anchor times (t_mix_s, t_song_s). Keys whose
    pairing explodes (very common patterns) are skipped — they carry no
    localization information and only add noise to the Hough."""
    tm: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    for k, mts in mix_h.items():
        rts = ref_h.get(k)
        if rts is None or len(mts) * len(rts) > cfg.pair_cap:
            continue
        g = np.meshgrid(mts, rts, indexing="ij")
        tm.append(g[0].ravel())
        ts.append(g[1].ravel())
    if not tm:
        return np.zeros((0, 2), dtype=np.float32)
    out = np.stack(
        [np.concatenate(tm) / LFPS, np.concatenate(ts) / LFPS], axis=1
    ).astype(np.float32)
    return out
