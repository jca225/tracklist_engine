"""Phase 3.2: the chroma matched-filter probe, adapted to the harness contract.

Wraps refine_ref_offsets.detect_offset (math unchanged) — the harmonic placement
signal that's strong on long melodic sections and weak on short repetitive drops
(complementary to the fingerprint probe, which is the opposite). Normalizes its
(ref_start_s, peak, stretch) tuple into an AlignmentResult.

Confidence is the correlation peak, clamped to [0,1] (calibration is Phase 3.3).
Feature extraction is injectable so the adapter is testable without librosa.
"""

from __future__ import annotations

from typing import Callable

from ..refine_ref_offsets import STRETCHES, detect_offset
from .contract import AlignmentResult, CandidatePool, MixContext, Probe, RefContext

# Provisional abstain floor: chroma cosine is high on a true long-melodic match,
# lower at chance. Replaced by calibration (Phase 3.3).
_MIN_PEAK = 0.5


def chroma_result_to_alignment(
    ref_start_s: float,
    peak: float,
    stretch: float,
    *,
    recording_id: str | None,
    min_peak: float = _MIN_PEAK,
    source: str = "chroma",
) -> AlignmentResult:
    """Map a raw detect_offset tuple to a normalized AlignmentResult.

    Abstains on a weak correlation peak rather than committing to a chroma match
    that's likely chance (chroma is non-discriminative on short repetitive audio).
    """
    confidence = max(0.0, min(1.0, peak))
    if confidence < min_peak:
        return AlignmentResult.abstained(source=source, recording_id=recording_id)
    return AlignmentResult(
        recording_id=recording_id,
        offset_s=ref_start_s,
        tempo_ratio=stretch,
        confidence=confidence,
        source=source,
    )


def _default_chroma(path, *, start_s: float | None = None, end_s: float | None = None):
    import librosa

    from ..refine_ref_offsets import SR, chroma

    y, _ = librosa.load(str(path), sr=SR, mono=True)
    if start_s is not None:
        a = int(start_s * SR)
        b = int(end_s * SR) if end_s is not None else len(y)
        y = y[a:b]
    return chroma(y)


class ChromaProbe(Probe):
    """Chroma matched-filter placement, normalized to the contract."""

    name = "chroma"

    def __init__(
        self,
        *,
        stretches: tuple[float, ...] = STRETCHES,
        mix_chroma: Callable[[MixContext], object] | None = None,
        ref_chroma: Callable[[RefContext], object] | None = None,
    ) -> None:
        self._stretches = stretches
        self._mix_chroma = mix_chroma or (
            lambda m: _default_chroma(
                m.audio_path, start_s=m.span_start_s, end_s=m.span_end_s
            )
        )
        self._ref_chroma = ref_chroma or (lambda r: _default_chroma(r.audio_path))

    def run(
        self, mix: MixContext, ref: RefContext, candidates: CandidatePool
    ) -> AlignmentResult:
        win_f = self._mix_chroma(mix)
        ref_f = self._ref_chroma(ref)
        ref_start_s, peak, stretch = detect_offset(win_f, ref_f, self._stretches)
        return chroma_result_to_alignment(
            ref_start_s, peak, stretch, recording_id=ref.recording_id, source=self.name
        )
