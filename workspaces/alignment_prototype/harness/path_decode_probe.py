"""Phase 3.2: the piecewise-linear path-decode probe, adapted to the contract.

Wraps path_decode.decode_path (math unchanged) — the only probe that returns a
PIECEWISE map (loops, section-jumps, half/double-time), which lands directly in
AlignmentResult.segments. offset_s/ref_end_s summarize the first/last segment.

Confidence is the Viterbi path score divided by the number of mix frames — i.e.
the mean per-frame match quality, net of jump penalties — clamped to [0,1]. That
is a principled provisional signal (not a fabricated logistic), but it is the
probe whose confidence most needs calibration (Phase 3.3); the mapping is isolated
in `score_to_confidence` so calibration drops in there.
"""

from __future__ import annotations

from typing import Callable

from ..path_decode import decode_path
from ..refine_ref_offsets import STRETCHES
from .chroma_probe import _default_chroma
from .contract import (
    AlignmentResult,
    CandidatePool,
    MixContext,
    Probe,
    RefContext,
    RefSegment,
)

_MIN_CONFIDENCE = 0.3
_DEFAULT_LAM = 0.2  # jump penalty


def score_to_confidence(score: float, n_mix_frames: int) -> float:
    """Mean per-frame path score, clamped to [0,1]. Provisional (see docstring)."""
    return max(0.0, min(1.0, score / max(1, n_mix_frames)))


def path_result_to_alignment(
    segments: list[tuple[float, float, float]],
    score: float,
    *,
    n_mix_frames: int,
    recording_id: str | None,
    min_confidence: float = _MIN_CONFIDENCE,
    source: str = "path",
) -> AlignmentResult:
    """Map a decode_path (segments, score) into an AlignmentResult.

    Abstains when the decode found no segments or the mean path quality is weak.
    """
    if not segments:
        return AlignmentResult.abstained(source=source, recording_id=recording_id)
    confidence = score_to_confidence(score, n_mix_frames)
    if confidence < min_confidence:
        return AlignmentResult.abstained(source=source, recording_id=recording_id)
    refsegs = tuple(
        RefSegment(mix_start_s=ms, ref_start_s=rs, ref_end_s=re)
        for (ms, rs, re) in segments
    )
    return AlignmentResult(
        recording_id=recording_id,
        offset_s=refsegs[0].ref_start_s,
        ref_end_s=refsegs[-1].ref_end_s,
        segments=refsegs,
        confidence=confidence,
        source=source,
    )


class PathDecodeProbe(Probe):
    """Piecewise-linear Viterbi placement, normalized to the contract."""

    name = "path"

    def __init__(
        self,
        *,
        stretches: tuple[float, ...] = STRETCHES,
        lam: float = _DEFAULT_LAM,
        mix_features: Callable[[MixContext], object] | None = None,
        ref_features: Callable[[RefContext], object] | None = None,
    ) -> None:
        self._stretches = stretches
        self._lam = lam
        self._mix_features = mix_features or (
            lambda m: _default_chroma(
                m.audio_path, start_s=m.span_start_s, end_s=m.span_end_s
            )
        )
        self._ref_features = ref_features or (lambda r: _default_chroma(r.audio_path))

    def run(
        self, mix: MixContext, ref: RefContext, candidates: CandidatePool
    ) -> AlignmentResult:
        M = self._mix_features(mix)
        R = self._ref_features(ref)
        segments, score = decode_path(M, R, self._stretches, self._lam)
        return path_result_to_alignment(
            segments,
            score,
            n_mix_frames=M.shape[1],
            recording_id=ref.recording_id,
            source=self.name,
        )
