"""Render BB12-realistic windows: handoffs, loops, stacked vocals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from workspaces.alignment_prototype.refine_ref_offsets import SR

from .render import _add_with_gain, _load_mono, write_flac
from .timeline import AcappellaSpan, InstrumentalBlock, MashupWindowV2, RegularSpan


def _gain_envelope(
    n: int, start_s: float, curve: tuple[tuple[float, float], ...], sr: int = SR
) -> np.ndarray:
    gain = np.ones(n, dtype=np.float32)
    if not curve:
        return gain
    times = np.array([t for t, _ in curve], dtype=np.float64)
    vals = np.array([v for _, v in curve], dtype=np.float32)
    rel = np.arange(n, dtype=np.float64) / sr + start_s
    gain = np.interp(rel, times, vals).astype(np.float32)
    return gain


def _render_bed_slice(
    bed_path,
    sl: MixSlice,
    *,
    sr: int = SR,
) -> tuple[np.ndarray, int]:
    y = _load_mono(bed_path, sr=sr)
    ref_i0 = int(sl.ref_start_s * sr)
    ref_i1 = int(sl.ref_end_s * sr)
    seg = y[ref_i0:ref_i1]
    target_len = int((sl.mix_end_s - sl.mix_start_s) * sr)
    if len(seg) > target_len:
        seg = seg[:target_len]
    elif len(seg) < target_len:
        seg = np.pad(seg, (0, target_len - len(seg)))
    start = int(sl.mix_start_s * sr)
    return seg, start


def _render_instrumental_block(
    block: InstrumentalBlock,
    mix_instr: np.ndarray,
    mix_full: np.ndarray,
    *,
    crossfade_s: float,
) -> None:
    for sl in block.slices:
        seg, start = _render_bed_slice(block.bed.path, sl)
        if seg.size == 0:
            continue
        gain = np.ones(len(seg), dtype=np.float32)
        rel = np.arange(len(seg), dtype=np.float64) / SR + sl.mix_start_s
        # Block-level handoff envelope (avoids double-loud beds in overlap).
        if crossfade_s > 0:
            if block.mix_start_s > 0.01:
                gain *= np.clip(
                    (rel - block.mix_start_s) / crossfade_s, 0.0, 1.0
                ).astype(np.float32)
            if block.mix_end_s < len(mix_instr) / SR - 0.01:
                gain *= np.clip(
                    (block.mix_end_s - rel) / crossfade_s, 0.0, 1.0
                ).astype(np.float32)
        _add_with_gain(mix_instr, seg, start, gain)
        _add_with_gain(mix_full, seg, start, gain)


def _stretch_pitch_vocal(
    seg: np.ndarray,
    *,
    host_bpm: float,
    payload_bpm: float,
    pitch_semi: int,
) -> np.ndarray:
    import librosa

    out = seg
    rate = payload_bpm / host_bpm if host_bpm > 0 else 1.0
    if abs(rate - 1.0) > 0.005:
        out = librosa.effects.time_stretch(out, rate=rate)
    if pitch_semi:
        out = librosa.effects.pitch_shift(out, sr=SR, n_steps=pitch_semi)
    return out


def _render_acap_linear(
    span: AcappellaSpan,
    mix_vocals: np.ndarray,
    mix_full: np.ndarray,
) -> None:
    y = _load_mono(span.payload.path)
    ref_i0 = int(span.ref_start_s * SR)
    ref_i1 = int(min(span.ref_end_s, len(y) / SR - 0.01) * SR)
    seg = y[ref_i0:ref_i1]
    seg = _stretch_pitch_vocal(
        seg,
        host_bpm=span.host_bpm,
        payload_bpm=span.payload.bpm,
        pitch_semi=span.pitch_shift_semi,
    )
    target_len = int((span.mix_end_s - span.mix_start_s) * SR)
    if len(seg) > target_len:
        seg = seg[:target_len]
    elif len(seg) < target_len:
        seg = np.pad(seg, (0, target_len - len(seg)))
    start = int(span.mix_start_s * SR)
    gain = _gain_envelope(len(seg), span.mix_start_s, span.gain_curve)
    _add_with_gain(mix_vocals, seg, start, gain)
    _add_with_gain(mix_full, seg, start, gain)


def _render_acap_loop(
    span: AcappellaSpan,
    mix_vocals: np.ndarray,
    mix_full: np.ndarray,
) -> None:
    y = _load_mono(span.payload.path)
    for sl in span.slices:
        ref_i0 = int(sl.ref_start_s * SR)
        ref_i1 = int(sl.ref_end_s * SR)
        seg = y[ref_i0:ref_i1]
        seg = _stretch_pitch_vocal(
            seg,
            host_bpm=span.host_bpm,
            payload_bpm=span.payload.bpm,
            pitch_semi=span.pitch_shift_semi,
        )
        target_len = int((sl.mix_end_s - sl.mix_start_s) * SR)
        if len(seg) > target_len:
            seg = seg[:target_len]
        elif len(seg) < target_len:
            seg = np.pad(seg, (0, target_len - len(seg)))
        start = int(sl.mix_start_s * SR)
        gain = _gain_envelope(len(seg), sl.mix_start_s, span.gain_curve)
        _add_with_gain(mix_vocals, seg, start, gain)
        _add_with_gain(mix_full, seg, start, gain)


def _render_regular(
    span: RegularSpan,
    mix_vocals: np.ndarray,
    mix_instr: np.ndarray,
    mix_full: np.ndarray,
) -> None:
    inst = _load_mono(span.regular.instrumental_path)
    voc = _load_mono(span.regular.vocals_path)
    m = min(len(inst), len(voc))
    full = inst[:m] + voc[:m]
    ref_i0 = int(span.ref_start_s * SR)
    ref_i1 = int(min(span.ref_end_s, m / SR - 0.01) * SR)
    seg = full[ref_i0:ref_i1]
    seg = _stretch_pitch_vocal(
        seg,
        host_bpm=span.host_bpm,
        payload_bpm=span.regular.bpm,
        pitch_semi=span.pitch_shift_semi,
    )
    target_len = int((span.mix_end_s - span.mix_start_s) * SR)
    if len(seg) > target_len:
        seg = seg[:target_len]
    elif len(seg) < target_len:
        seg = np.pad(seg, (0, target_len - len(seg)))
    start = int(span.mix_start_s * SR)
    gain = _gain_envelope(len(seg), span.mix_start_s, span.gain_curve)
    # A full song carries both channels; split for the routed-stem targets.
    _add_with_gain(mix_full, seg, start, gain)
    _add_with_gain(mix_vocals, seg, start, gain * 0.5)
    _add_with_gain(mix_instr, seg, start, gain * 0.5)


@dataclass(frozen=True)
class RenderedWindowV2:
    mix: np.ndarray
    mix_vocals: np.ndarray
    mix_instrumental: np.ndarray
    sr: int


def render_window_v2(
    window: MashupWindowV2,
    *,
    crossfade_s: float = 3.0,
) -> RenderedWindowV2:
    n = int(window.window_duration_s * SR)
    mix = np.zeros(n, dtype=np.float32)
    mix_vocals = np.zeros(n, dtype=np.float32)
    mix_instr = np.zeros(n, dtype=np.float32)

    for block in window.instrumentals:
        _render_instrumental_block(block, mix_instr, mix, crossfade_s=crossfade_s)

    for ac in window.acappellas:
        if ac.is_loop:
            _render_acap_loop(ac, mix_vocals, mix)
        else:
            _render_acap_linear(ac, mix_vocals, mix)

    for reg in window.regulars:
        _render_regular(reg, mix_vocals, mix_instr, mix)

    peak = max(float(np.max(np.abs(mix))), 1e-6)
    scale = 0.95 / peak
    return RenderedWindowV2(
        mix=mix * scale,
        mix_vocals=mix_vocals * scale,
        mix_instrumental=mix_instr * scale,
        sr=SR,
    )


__all__ = ["RenderedWindowV2", "render_window_v2", "write_flac"]
