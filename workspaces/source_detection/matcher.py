"""The localization core: a reversed, pitch- and stretch-aware chroma matched
filter. Slides each source template over the mix (the inverse of
refine_ref_offsets, which slides a mix window over a ref), across 12 semitone
rotations and a BPM-derived stretch band, and peak-picks raw detections.

Stretch handling follows the domain rule: the host instrumental's BPM is fixed
within a span and overlaid acapellas are beat-synced to it, so the time-stretch
is ~ mix_local_bpm / source_bpm and we search a tight band around that estimate
rather than a blind seconds grid."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from . import config, features
from .records import Detection, MixInput, SourceTrack

log = logging.getLogger(__name__)


def _stretch_set(source_bpm: float | None, mix_bpm: float | None) -> list[float]:
    """Multiplicative stretch factors to try (mix sec per source sec)."""
    if source_bpm and mix_bpm:
        est = mix_bpm / source_bpm
        # acapellas are bar-synced: a 2x / 0.5x BPM read is the same groove
        while est > config.STRETCH_BAND[1]:
            est /= 2.0
        while est < config.STRETCH_BAND[0]:
            est *= 2.0
        lo, hi = config.STRETCH_BAND
        span = (hi - lo) * 0.5
        cand = np.linspace(max(lo, est - span / 2), min(hi, est + span / 2),
                           config.STRETCH_STEPS)
        return sorted({round(float(s), 4) for s in cand} | {1.0})
    return list(config.FALLBACK_STRETCHES)


def _match_curve(tmpl: np.ndarray, mixc: np.ndarray) -> np.ndarray:
    """Normalized matched-filter score of `tmpl` (12, m) at every start frame of
    `mixc` (12, M). Cosine-like in [-1, 1]. Returns (M - m + 1,)."""
    from scipy.signal import fftconvolve

    m = tmpl.shape[1]
    if mixc.shape[1] <= m:
        return np.array([], dtype=np.float32)
    w = tmpl / (np.linalg.norm(tmpl) + 1e-9)
    num = fftconvolve(mixc, w[:, ::-1], mode="valid", axes=1).sum(axis=0)
    e = np.concatenate([[0.0], np.cumsum((mixc ** 2).sum(axis=0))])
    den = np.sqrt(np.maximum(e[m:] - e[:-m], 1e-9))
    return (num / den).astype(np.float32)


def _resample_cols(tmpl: np.ndarray, stretch: float) -> np.ndarray:
    """Stretch a (12, n) template to (12, round(n*stretch)) by nearest-frame
    resampling — the source played at 1/stretch speed occupies stretch x frames
    in the mix."""
    n = tmpl.shape[1]
    m = max(1, int(round(n * stretch)))
    idx = np.clip((np.arange(m) / stretch).astype(int), 0, n - 1)
    return tmpl[:, idx]


def _peak_pick(curve: np.ndarray, min_score: float, min_dist_frames: int):
    from scipy.signal import find_peaks
    if curve.size == 0:
        return np.array([], dtype=int)
    peaks, _ = find_peaks(curve, height=min_score, distance=max(1, min_dist_frames))
    return peaks


def detect_source(
    source: SourceTrack, mix: MixInput, cfg: config.Config,
) -> list[Detection]:
    """All raw detections of `source` anywhere in `mix`, over both stem channels."""
    dets: list[Detection] = []
    mix_bpm = (features.bpm_of(mix.instrumental) if mix.instrumental
               else (features.bpm_of(mix.full) if mix.full else None))

    for channel, src_path in source.channels():
        mix_path = _mix_channel_for(channel, mix)
        if mix_path is None or not mix_path.is_file():
            continue
        mixc = features.chroma_of(mix_path)
        srcc = features.chroma_of(src_path)
        if srcc.shape[1] < 4 or mixc.shape[1] < 4:
            continue

        src_bpm = source.bpm or features.bpm_of(src_path)
        stretches = _stretch_set(src_bpm, mix_bpm)
        tmpl_len = int(round(cfg.template_s / config.FRAME_S))
        stride = int(round(cfg.template_stride_s / config.FRAME_S))
        n_frames = srcc.shape[1]
        offsets = list(range(0, max(1, n_frames - tmpl_len // 2), stride))[:8]

        for off in offsets:
            base = srcc[:, off:off + tmpl_len]
            if base.shape[1] < tmpl_len // 2:
                continue
            best = _best_over_rot_stretch(base, mixc, stretches, cfg)
            if best is None:
                continue
            curve, rot_at, str_at, m_at = best
            min_dist = int(round(config.PEAK_MIN_DISTANCE_S / config.FRAME_S))
            for k in _peak_pick(curve, cfg.peak_min_score, min_dist):
                s = float(str_at[k])
                m = int(m_at[k])
                r = int(rot_at[k])
                mix_start = k * config.FRAME_S
                ref_start = off * config.FRAME_S
                dets.append(Detection(
                    sid=source.sid, name=source.name, channel=channel,
                    mix_start_s=mix_start, mix_end_s=mix_start + m * config.FRAME_S,
                    ref_start_s=ref_start, ref_end_s=ref_start + cfg.template_s,
                    pitch_shift_semi=r if r <= 6 else r - 12,
                    time_stretch=s, confidence=float(curve[k]),
                    extra={"template_off_s": round(ref_start, 2)},
                ))
    return dets


def _best_over_rot_stretch(base, mixc, stretches, cfg):
    """Best (score, rotation, stretch, mix-length) per mix start frame, reduced
    over all 12 rotations and the stretch set. Curves of differing length are
    aligned at frame 0 and truncated to the common prefix."""
    M = mixc.shape[1]
    longest_m = max(int(round(base.shape[1] * s)) for s in stretches)
    L = M - longest_m + 1
    if L <= 0:
        return None
    best_score = np.full(L, -2.0, dtype=np.float32)
    best_rot = np.zeros(L, dtype=np.int16)
    best_str = np.ones(L, dtype=np.float32)
    best_m = np.full(L, longest_m, dtype=np.int32)
    for s in stretches:
        stretched = _resample_cols(base, s)
        m = stretched.shape[1]
        for r in range(cfg.n_rotations):
            rolled = np.roll(stretched, r, axis=0)
            curve = _match_curve(rolled, mixc)[:L]
            if curve.size < L:
                continue
            better = curve > best_score
            best_score[better] = curve[better]
            best_rot[better] = r
            best_str[better] = s
            best_m[better] = m
    return best_score, best_rot, best_str, best_m


def _mix_channel_for(source_channel: str, mix: MixInput) -> Path | None:
    """A source's vocal template is matched against the mix vocal stem; its
    instrumental against the mix instrumental stem; 'full' falls back to the mix
    instrumental (the host bed carries the harmony)."""
    if source_channel == "vocals":
        return mix.vocals or mix.full
    if source_channel == "instrumental":
        return mix.instrumental or mix.full
    return mix.instrumental or mix.full
