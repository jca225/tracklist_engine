"""The HuBERT phonetic matched-filter probe, adapted to the harness contract.

Wraps similarity_probe._hubert + refine_ref_offsets.detect_offset (math
unchanged) — the LANGUAGE-axis placement signal. HuBERT matches on *what is
sung*, so key changes / pitch shifts are irrelevant ("the lyrics don't
transpose"): it localizes vocals where chroma cannot. Measured on BB11 (36
acappella spans): HuBERT ref-offset median 2.1s vs chroma 39.6s, <5s 67% vs 36%.

This is the vocal-channel counterpart to ChromaProbe (the harmony axis); the two
fail on COMPLEMENTARY spans (chroma rescues a few HuBERT catastrophes), which is
why the fusion arbiter — not either alone — is the goal.

CONFIDENCE CAVEAT: the HuBERT correlation peak is NOT well-separated — the BB11
eval has correct hits at peak~0.30 and wrong matches at peak~0.40. So `confidence
= peak` here is provisional; real calibration + per-context weighting is the
learned arbiter (Phase C). The abstain floor is set low so it only refuses the
clearly-at-chance matches, deferring the real threshold to the arbiter.
"""

from __future__ import annotations

from typing import Callable

from ..refine_ref_offsets import STRETCHES, detect_offset
from .contract import AlignmentResult, CandidatePool, MixContext, Probe, RefContext

# HuBERT mid layer carries the most phonetic content (similarity_probe default).
_HUBERT_LAYER = 9
# Conservative floor: BB11 misses cluster at peak 0.19-0.40 but a correct span
# landed at 0.30, so a high floor would abstain real hits. Keep it low; the
# arbiter (Phase C) learns the true, context-dependent threshold.
_MIN_PEAK = 0.25


def hubert_result_to_alignment(
    ref_start_s: float,
    peak: float,
    stretch: float,
    *,
    recording_id: str | None,
    min_peak: float = _MIN_PEAK,
    source: str = "hubert",
) -> AlignmentResult:
    """Map a raw detect_offset tuple (HuBERT features) to an AlignmentResult."""
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


def _default_hubert(
    path,
    *,
    layer: int = _HUBERT_LAYER,
    start_s: float | None = None,
    end_s: float | None = None,
):
    import librosa

    from ..refine_ref_offsets import SR
    from workspaces.section_hsmm.similarity_probe import _hubert

    y, _ = librosa.load(str(path), sr=SR, mono=True)
    if start_s is not None:
        a = int(start_s * SR)
        b = int(end_s * SR) if end_s is not None else len(y)
        y = y[a:b]
    # _hubert resamples SR->16k internally and returns (768, T) on the SR/HOP
    # grid, so it is a drop-in for chroma in the same detect_offset harness.
    return _hubert(y, layer)


class HubertProbe(Probe):
    """HuBERT phonetic matched-filter placement (vocal/language axis)."""

    name = "hubert"

    def __init__(
        self,
        *,
        layer: int = _HUBERT_LAYER,
        stretches: tuple[float, ...] = STRETCHES,
        mix_feat: Callable[[MixContext], object] | None = None,
        ref_feat: Callable[[RefContext], object] | None = None,
    ) -> None:
        self._layer = layer
        self._stretches = stretches
        self._mix_feat = mix_feat or (
            lambda m: _default_hubert(
                m.audio_path, layer=layer, start_s=m.span_start_s, end_s=m.span_end_s
            )
        )
        self._ref_feat = ref_feat or (
            lambda r: _default_hubert(r.audio_path, layer=layer)
        )

    def run(
        self, mix: MixContext, ref: RefContext, candidates: CandidatePool
    ) -> AlignmentResult:
        win_f = self._mix_feat(mix)
        ref_f = self._ref_feat(ref)
        ref_start_s, peak, stretch = detect_offset(win_f, ref_f, self._stretches)
        return hubert_result_to_alignment(
            ref_start_s, peak, stretch, recording_id=ref.recording_id, source=self.name
        )
