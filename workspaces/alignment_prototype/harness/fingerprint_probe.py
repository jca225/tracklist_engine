"""Phase 3.2: the landmark-fingerprint probe, adapted to the harness contract.

Wraps landmark_fp.fp_offset (the proven constellation matcher — NOT reimplemented)
and normalizes its bespoke (ref_start_s, votes, stretch, sharpness) tuple into an
AlignmentResult. fingerprinting is the sharpest signal for the wrong-content error,
and its confidence is the most legible of any probe (votes ~ hundreds for a real
hit, ~0 for a miss).

Confidence here is a PROVISIONAL monotone squash of the vote count — good enough
for abstention + ranking. Replacing it with a calibrated [0,1] (Platt/isotonic vs
held-out GT) is Phase 3.3; the squash is isolated in `votes_to_confidence` so the
calibration drops in without touching the probe.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

from ..landmark_fp import fp_offset
from .contract import AlignmentResult, CandidatePool, MixContext, Probe, RefContext

# Provisional constants (pre-calibration). votes for a real hit run into the
# hundreds; a miss is ~0. sharpness is peak/second-peak.
_VOTE_SCALE = 80.0
_MIN_VOTES = 8
_MIN_SHARPNESS = 1.2


def votes_to_confidence(votes: int, *, vote_scale: float = _VOTE_SCALE) -> float:
    """Monotone squash of vote count into [0,1]: 0 votes -> 0, saturating to 1.

    Provisional (see module docstring). Strictly increasing in votes so it is a
    valid ranking signal even before calibration.
    """
    if votes <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - math.exp(-votes / vote_scale)))


def fp_result_to_alignment(
    ref_start_s: float,
    votes: int,
    stretch: float,
    sharpness: float,
    *,
    recording_id: str | None,
    min_votes: int = _MIN_VOTES,
    min_sharpness: float = _MIN_SHARPNESS,
    source: str = "fp",
) -> AlignmentResult:
    """Map a raw fp_offset tuple to a normalized AlignmentResult.

    Abstains (rather than guessing) when the constellation agreement is too weak:
    too few votes, or a peak that doesn't clearly beat the runner-up.
    """
    if votes < min_votes or sharpness < min_sharpness:
        return AlignmentResult.abstained(source=source, recording_id=recording_id)
    return AlignmentResult(
        recording_id=recording_id,
        offset_s=ref_start_s,
        tempo_ratio=stretch,
        confidence=votes_to_confidence(votes),
        source=source,
    )


def _default_loader(path: Path):
    import librosa

    from ..landmark_fp import SR

    y, _ = librosa.load(str(path), sr=SR, mono=True)
    return y


class FingerprintProbe(Probe):
    """Landmark-constellation placement, normalized to the contract."""

    name = "fp"

    def __init__(
        self,
        *,
        stretches: tuple[float, ...] = (0.98, 1.0, 1.02),
        loader: Callable[[Path], object] | None = None,
    ) -> None:
        self._stretches = stretches
        self._load = loader or _default_loader

    def run(
        self, mix: MixContext, ref: RefContext, candidates: CandidatePool
    ) -> AlignmentResult:
        mix_y = self._load(mix.audio_path)
        ref_y = self._load(ref.audio_path)
        ref_start_s, votes, stretch, sharpness = fp_offset(
            mix_y, ref_y, stretches=self._stretches
        )
        return fp_result_to_alignment(
            ref_start_s,
            votes,
            stretch,
            sharpness,
            recording_id=ref.recording_id,
            source=self.name,
        )
