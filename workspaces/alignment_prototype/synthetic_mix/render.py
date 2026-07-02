"""Audio renderer: beatmatch, pitch-shift, structured overlap with fades."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from workspaces.alignment_prototype.refine_ref_offsets import SR

from .scenario import MashupScenario, OverlaySpec

_FADE_S = 4.0


def _load_mono(path, sr: int = SR) -> np.ndarray:
    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


def _apply_fade(gain: np.ndarray, fade_in: int, fade_out: int) -> None:
    n = gain.shape[0]
    if fade_in > 0:
        gain[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
    if fade_out > 0:
        gain[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)


def _prepare_overlay(
    spec: OverlaySpec,
    bed_bpm: float,
    payload_bpm: float,
) -> np.ndarray:
    """Return beatmatched + pitch-shifted vocal segment for the mix window."""
    import librosa

    y = _load_mono(spec.payload.path)
    ref_end = spec.ref_start_s + (spec.set_end_s - spec.set_start_s) * spec.tempo_ratio
    ref_end = min(ref_end, len(y) / SR - 0.01)
    i0 = int(spec.ref_start_s * SR)
    i1 = int(ref_end * SR)
    if i1 - i0 < SR:
        return np.zeros(0, dtype=np.float32)
    seg = y[i0:i1]

    rate = payload_bpm / bed_bpm if bed_bpm > 0 else 1.0
    if abs(rate - 1.0) > 0.005:
        seg = librosa.effects.time_stretch(seg, rate=rate)
    if spec.pitch_shift_semi:
        seg = librosa.effects.pitch_shift(seg, sr=SR, n_steps=spec.pitch_shift_semi)

    target_len = int((spec.set_end_s - spec.set_start_s) * SR)
    if len(seg) > target_len:
        seg = seg[:target_len]
    elif len(seg) < target_len:
        seg = np.pad(seg, (0, target_len - len(seg)))
    return seg


def _add_with_gain(
    buf: np.ndarray, seg: np.ndarray, start: int, gain: np.ndarray
) -> None:
    n = min(len(seg), len(buf) - start, len(gain))
    if n <= 0 or start < 0:
        return
    buf[start : start + n] += seg[:n] * gain[:n]


@dataclass(frozen=True)
class RenderedMix:
    mix: np.ndarray
    mix_vocals: np.ndarray
    mix_instrumental: np.ndarray
    sr: int


def render_scenario(scenario: MashupScenario) -> RenderedMix:
    """Composite bed + overlays into mix channels."""
    n = int(scenario.mix_duration_s * SR)
    mix = np.zeros(n, dtype=np.float32)
    mix_vocals = np.zeros(n, dtype=np.float32)
    mix_instr = np.zeros(n, dtype=np.float32)

    bed_y = _load_mono(scenario.bed.path)
    bed_i0 = int(scenario.bed_ref_start_s * SR)
    bed_seg = bed_y[bed_i0 : bed_i0 + n]
    if len(bed_seg) < n:
        bed_seg = np.pad(bed_seg, (0, n - len(bed_seg)))
    mix_instr[:] = bed_seg
    mix[:] += bed_seg

    fade = int(_FADE_S * SR)
    for ov in scenario.overlays:
        seg = _prepare_overlay(ov, scenario.bed.bpm, ov.payload.bpm)
        if seg.size == 0:
            continue
        gain = np.ones(len(seg), dtype=np.float32)
        _apply_fade(gain, fade, fade)
        start = int(ov.set_start_s * SR)
        _add_with_gain(mix, seg, start, gain)
        _add_with_gain(mix_vocals, seg, start, gain)

    peak = max(float(np.max(np.abs(mix))), 1e-6)
    scale = 0.95 / peak
    mix *= scale
    mix_vocals *= scale
    mix_instr *= scale
    return RenderedMix(
        mix=mix, mix_vocals=mix_vocals, mix_instrumental=mix_instr, sr=SR
    )


def write_flac(path, y: np.ndarray, sr: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, format="FLAC", subtype="PCM_16")
