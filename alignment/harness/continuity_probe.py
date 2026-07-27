"""The continuity-stack probe, adapted to the harness contract.

Wraps continuity_refine._stack_best (math unchanged) — the REPEAT-ROBUST
placement signal. A single matched-filter window argmaxes onto any equivalent
repeat (the chorus that recurs 3x); stacking K probe windows across the span and
summing their offset curves reinforces the ONE diagonal the whole span agrees on
and washes out spurious repeats (BB12: 42% -> 59% exact-<2s on straight clips).
Complementary to the single-window chroma/HuBERT probes — it's the harmony-axis
answer to the fiber/repeat-ambiguity failures.

Confidence is the stack peak's z-score (peak in units of the channel's own score
noise — debiased so sparse vocal and dense instrumental stacks compare), squashed
to [0,1]; calibration is Phase C. Abstains below a provisional z floor.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..continuity_refine import _probe_offsets, _stack_best, _windows_from
from ..refine_ref_offsets import HOP, SR, STRETCHES, chroma
from .contract import AlignmentResult, CandidatePool, MixContext, Probe, RefContext

_WINDOW_S = 12.0
_K_MAX = 5
_Z_SCALE = 3.0  # z-score that maps to confidence 1.0
_MIN_Z = 1.0  # provisional abstain floor (peak must clear 1 sd of channel noise)


def _default_chroma_whole(path) -> np.ndarray:
    import librosa
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(str(path), sr=SR, mono=True)
    return chroma(y)


class ContinuityProbe(Probe):
    """Stacked-window matched-filter placement (repeat-robust, harmony axis)."""

    name = "continuity"

    def __init__(
        self,
        *,
        window_s: float = _WINDOW_S,
        k_max: int = _K_MAX,
        stretches: tuple[float, ...] = STRETCHES,
        band_s: float = 0.0,
        z_scale: float = _Z_SCALE,
        min_z: float = _MIN_Z,
        mix_chroma_whole: Callable[[MixContext], np.ndarray] | None = None,
        ref_chroma: Callable[[RefContext], np.ndarray] | None = None,
    ) -> None:
        self._window_s = window_s
        self._k_max = k_max
        self._stretches = stretches
        self._band_frames = int(band_s * SR / HOP)
        self._z_scale = z_scale
        self._min_z = min_z
        self._mix_chroma_whole = mix_chroma_whole or (
            lambda m: _default_chroma_whole(m.audio_path)
        )
        self._ref_chroma = ref_chroma or (lambda r: _default_chroma_whole(r.audio_path))

    def run(
        self, mix: MixContext, ref: RefContext, candidates: CandidatePool
    ) -> AlignmentResult:
        # the stack needs the span extent to place its K probe windows.
        if mix.span_start_s is None or mix.span_end_s is None:
            return AlignmentResult.abstained(
                source=self.name, recording_id=ref.recording_id
            )
        span_len = mix.span_end_s - mix.span_start_s
        n = int(self._window_s * SR / HOP)
        dts = _probe_offsets(span_len, self._window_s, self._k_max)
        mc = self._mix_chroma_whole(mix)
        win_list = _windows_from(mc, mix.span_start_s, dts, n)
        windows = [(int(dt), np.asarray(w, dtype=np.float32)) for dt, w in win_list]
        if not windows:
            return AlignmentResult.abstained(
                source=self.name, recording_id=ref.recording_id
            )
        ref_f = self._ref_chroma(ref)
        best = _stack_best(windows, ref_f, tuple(self._stretches), self._band_frames)
        if best is None:
            return AlignmentResult.abstained(
                source=self.name, recording_id=ref.recording_id
            )
        r0_s, _peak, _prom, stretch, z = best
        if z < self._min_z:
            return AlignmentResult.abstained(
                source=self.name, recording_id=ref.recording_id
            )
        confidence = max(0.0, min(1.0, z / self._z_scale))
        return AlignmentResult(
            recording_id=ref.recording_id,
            offset_s=r0_s,
            tempo_ratio=stretch,
            confidence=confidence,
            source=self.name,
        )
