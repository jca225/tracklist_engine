"""Categorical operation LFs (lane 2 of Phase-1b).

Unlike offsets (continuous — see continuous_model.py), operation-type labels
are genuinely categorical, so DS-style aggregation IS well-matched here.
First LF: the keystone key-lock vs varispeed discriminator —
  varispeed: pitch shift == 12*log2(tempo_ratio)  (pitch/tempo coupled)
  key-lock:  tempo_ratio != 1 but pitch preserved (Master Tempo, 2001+)
Detector signatures grounded in docs/dj_operation_ontology_research.md §2.5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import librosa
import numpy as np

from pws_aligner.core.votes import AbstainReason

_BINS_PER_OCTAVE = 36  # 1/3-semitone CQT resolution
_N_BINS = 6 * _BINS_PER_OCTAVE
_FMIN = 55.0
_TEMPO_EPS = 0.02  # |r-1| below this = no meaningful tempo change
_PITCH_TOL_ST = 0.5  # tolerance on pitch-shift match, semitones
_MAX_SHIFT_ST = 12.0


class TempoPitchLabel(Enum):
    KEYLOCK = "keylock"
    VARISPEED = "varispeed"
    NO_TEMPO_CHANGE = "no_tempo_change"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class OperationVote:
    """Vote from one labeling function on a single span.

    When ``abstained`` is True, ``label`` is ``TempoPitchLabel.UNKNOWN.value``
    and MUST NOT be read as a decision.
    """

    lf: str
    span_id: str
    label: str
    confidence: float
    abstained: bool
    reason: AbstainReason


def _cqt_profile(y: np.ndarray, sr: int) -> np.ndarray:
    c = np.abs(
        librosa.cqt(
            y, sr=sr, fmin=_FMIN, n_bins=_N_BINS, bins_per_octave=_BINS_PER_OCTAVE
        )
    )
    prof = np.log1p(c).mean(axis=1)
    prof -= prof.mean()
    return prof


def estimate_pitch_shift_semitones(mix: np.ndarray, ref: np.ndarray, sr: int) -> float:
    """Pitch shift of mix relative to ref via CQT-profile cross-correlation."""
    pm, pr = _cqt_profile(mix, sr), _cqt_profile(ref, sr)
    max_lag = int(_MAX_SHIFT_ST * _BINS_PER_OCTAVE / 12)
    best_lag, best_score = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = pm[lag:], pr[: len(pr) - lag]
        else:
            a, b = pm[:lag], pr[-lag:]
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        score = float(np.dot(a, b) / denom)
        if score > best_score:
            best_lag, best_score = lag, score
    return best_lag * 12.0 / _BINS_PER_OCTAVE


def keylock_vs_varispeed(
    span_id: str, mix: np.ndarray, ref: np.ndarray, sr: int, tempo_ratio: float
) -> OperationVote:
    lf = "keylock_vs_varispeed"
    if abs(tempo_ratio - 1.0) < _TEMPO_EPS:
        return OperationVote(
            lf,
            span_id,
            TempoPitchLabel.NO_TEMPO_CHANGE.value,
            0.9,  # TODO: calibrate against GT operation labels
            False,
            AbstainReason.NONE,
        )
    shift = estimate_pitch_shift_semitones(mix, ref, sr)
    expected = 12.0 * math.log2(tempo_ratio)
    # Keylock check first: if pitch is preserved (near zero) it IS keylock,
    # regardless of how large the tempo change is.  A preserved-pitch
    # observation is an unambiguous keylock signature.
    if abs(shift) < _PITCH_TOL_ST:
        return OperationVote(
            lf,
            span_id,
            TempoPitchLabel.KEYLOCK.value,
            0.8,
            False,
            AbstainReason.NONE,  # TODO: calibrate against GT operation labels
        )
    # Varispeed: only classify when the expected coupled pitch shift is itself
    # detectable (> tolerance).  When the tempo nudge is so small that the
    # expected pitch shift is sub-tolerance, varispeed and keylock are
    # indistinguishable by pitch alone — but shift≈0 was already handled above,
    # so we fall through to abstain.
    if abs(expected) > _PITCH_TOL_ST and abs(shift - expected) < max(
        _PITCH_TOL_ST, 0.25 * abs(expected)
    ):
        return OperationVote(
            lf, span_id, TempoPitchLabel.VARISPEED.value, 0.8, False, AbstainReason.NONE
        )
    # pitch moved but matches neither pattern (e.g. key-lock + key-shift):
    return OperationVote(
        lf, span_id, TempoPitchLabel.UNKNOWN.value, 0.0, True, AbstainReason.LOW_MARGIN
    )
