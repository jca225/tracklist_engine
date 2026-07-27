"""Audio-side identity verification for the three canonical axes.

The autonomous acquisition loop needs to confirm that an acquired file is
actually the claimed version / stem / variant before it is promoted to a GT
source. This module provides heuristic, feature-based verifiers that consume
structured signals already produced by the analysis pipeline (duration, beats,
cue points, Essentia features, MERT embeddings). It is pure: callers load the
features from the DB or audio files; the verifiers only judge.

Verdicts are intentionally ternary:
- SAME / SAME_VERSION / SAME_VARIANT: the acquired audio matches the reference.
- DIFFERENT_*: the acquired audio is clearly a different song/version/variant.
- AMBIGUOUS: the signal is insufficient — the cascade must fall back to
  REVIEW_MANUAL or another tier (e.g. separation), never guess.

The thresholds are heuristic and will be calibrated / replaced by a learned
MERT head once the decision ledger (Phase 1c) accumulates enough cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np


class AxisVerdict(str, Enum):
    """Ternary outcome for one axis of identity verification."""

    SAME = "same"  # same song, same value on this axis
    DIFFERENT = "different"  # different song or different value on this axis
    AMBIGUOUS = "ambiguous"  # insufficient signal; refuse to guess


class VersionVerdict(str, Enum):
    SAME_VERSION = "same_version"
    DIFFERENT_VERSION = "different_version"
    AMBIGUOUS = "ambiguous"


class VariantVerdict(str, Enum):
    SAME_VARIANT = "same_variant"
    DIFFERENT_VARIANT = "different_variant"
    AMBIGUOUS = "ambiguous"


# -----------------------------------------------------------------------------
# Structured inputs (pure, no DB coupling)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class EssentiaFeatures:
    """Subset of track_audio_features used for version comparison."""

    bpm: float | None = None
    key_pc: int | None = None  # 0..11 pitch class
    key_mode: str | None = None  # 'major' | 'minor'
    energy: float | None = None
    danceability: float | None = None
    instrumentalness: float | None = None
    lufs: float | None = None


@dataclass(frozen=True)
class StructureFeatures:
    """Cue / measure structure derived from track_analysis and cue_points."""

    duration_s: float
    cue_points_s: Sequence[float] = ()
    measure_times_s: Sequence[float] = ()
    beat_times_s: Sequence[float] = ()


@dataclass(frozen=True)
class VersionVerifyInputs:
    """Everything the version verifier needs to compare two recordings."""

    reference: EssentiaFeatures
    candidate: EssentiaFeatures
    reference_structure: StructureFeatures | None = None
    candidate_structure: StructureFeatures | None = None
    mert_cosine: float | None = None  # pre-computed mean-pooled MERT similarity


@dataclass(frozen=True)
class VariantVerifyInputs:
    """Everything the variant verifier needs to compare regular vs candidate."""

    reference_structure: StructureFeatures
    candidate_structure: StructureFeatures
    # chromaprint similarity vs reference, if available (advisory)
    chromaprint_sim: float | None = None


# -----------------------------------------------------------------------------
# Variant verifier (extended vs regular, etc.)
# -----------------------------------------------------------------------------

# Extended tracks are longer than their regular/radio counterparts, typically by
# an extended intro and/or outro. The fingerprint module's band [0.90, 1.15] is
# too tight because a 5:30 extended vs 3:45 radio gives ratio ~1.47.
# Use a looser ratio + structure deltas.
_DUR_RATIO_EXTENDED_MIN = 1.05  # extended is at least 5% longer
_DUR_RATIO_EXTENDED_MAX = 1.80  # sanity ceiling; longer is a different arrangement
_DUR_RATIO_SAME_MAX = 1.05  # within 5% -> same variant (or very close edit)
_DUR_RATIO_TOO_SHORT = 0.95  # shorter than reference is not "extended"


def _measure_count(structure: StructureFeatures) -> int:
    return len(structure.measure_times_s)


def _intro_length_s(structure: StructureFeatures) -> float:
    """First significant cue point, or first measure after start, or 0."""
    cues = [c for c in structure.cue_points_s if c > 0.5]
    if cues:
        return float(cues[0])
    measures = structure.measure_times_s
    if len(measures) >= 2:
        return float(measures[1])
    return 0.0


def _outro_start_s(structure: StructureFeatures) -> float:
    """Last cue point, or last measure, or duration."""
    if structure.cue_points_s:
        return float(structure.cue_points_s[-1])
    if structure.measure_times_s:
        return float(structure.measure_times_s[-1])
    return structure.duration_s


def verify_variant(inputs: VariantVerifyInputs) -> VariantVerdict:
    """Heuristic: is the candidate a different variant (extended vs regular)?

    Returns SAME_VARIANT when duration and structure are too close to be a
    different variant. DIFFERENT_VARIANT when the candidate is substantially
    longer / has a longer intro/outro. AMBIGUOUS when the signal is inconsistent
    (e.g. duration says extended but cue structure says identical).
    """
    ref = inputs.reference_structure
    cand = inputs.candidate_structure

    if ref.duration_s <= 0:
        return VariantVerdict.AMBIGUOUS

    ratio = cand.duration_s / ref.duration_s
    if ratio < _DUR_RATIO_TOO_SHORT:
        # candidate is shorter than reference — not an "extended" variant;
        # could be a radio edit or a different cut. Ambiguous without more signal.
        return VariantVerdict.AMBIGUOUS

    if ratio <= _DUR_RATIO_SAME_MAX:
        # close enough in duration; check cue count to avoid mislabeling a
        # same-length remix as "same variant"
        ref_cues = len(ref.cue_points_s)
        cand_cues = len(cand.cue_points_s)
        if ref_cues and cand_cues and abs(cand_cues - ref_cues) > 2:
            return VariantVerdict.AMBIGUOUS
        return VariantVerdict.SAME_VARIANT

    if ratio >= _DUR_RATIO_EXTENDED_MIN and ratio <= _DUR_RATIO_EXTENDED_MAX:
        # candidate is longer — likely extended. Cross-check with intro/outro
        # growth: the extra length should be in the intro and/or outro.
        intro_growth = _intro_length_s(cand) - _intro_length_s(ref)
        outro_growth = (cand.duration_s - _outro_start_s(cand)) - (
            ref.duration_s - _outro_start_s(ref)
        )
        total_extra = cand.duration_s - ref.duration_s
        # require at least 25% of the extra length to be in intro+outro
        if intro_growth + outro_growth >= 0.25 * total_extra:
            return VariantVerdict.DIFFERENT_VARIANT
        return VariantVerdict.AMBIGUOUS

    # ratio outside both bands: dramatically different length → likely a
    # different arrangement or wrong song, not a clean variant.
    return VariantVerdict.AMBIGUOUS


# -----------------------------------------------------------------------------
# Version verifier (remix vs original, etc.)
# -----------------------------------------------------------------------------

_BPM_DIFF_SAME_MAX = 0.5  # same version: BPM within 0.5
_BPM_DIFF_DIFFERENT_MIN = 4.0  # different version: BPM differs by > 4 BPM
_KEY_PC_DIFF_SAME_MAX = 0  # exact same key
_KEY_PC_DIFF_DIFFERENT_MIN = 3  # at least 3 semitones different
_ENERGY_DIFF_DIFFERENT_MIN = 0.25  # big energy shift suggests a different version
_MERT_COSINE_SAME_MIN = 0.70  # same song (any version) has high timbre similarity
_MERT_COSINE_DIFFERENT_MAX = 0.50  # below this is likely a different song


def _key_distance(
    a_pc: int | None, a_mode: str | None, b_pc: int | None, b_mode: str | None
) -> int:
    """Semitone distance between keys, ignoring mode for distance (mod 12)."""
    if a_pc is None or b_pc is None:
        return -1  # unknown
    return min(abs(a_pc - b_pc), 12 - abs(a_pc - b_pc))


def _cue_pattern_diff(
    ref: StructureFeatures, cand: StructureFeatures, *, tolerance_s: float = 2.0
) -> float:
    """Fraction of cue points in the reference that are NOT matched within
    ``tolerance_s`` in the candidate, normalized to [0, 1]. A same-version
    recording should share most cue points (after tempo alignment). A remix
    will differ more."""
    if not ref.cue_points_s:
        return 0.0
    matched = 0
    for rc in ref.cue_points_s:
        if any(abs(rc - cc) <= tolerance_s for cc in cand.cue_points_s):
            matched += 1
    return 1.0 - (matched / len(ref.cue_points_s))


def verify_version(inputs: VersionVerifyInputs) -> VersionVerdict:
    """Heuristic: is the candidate the SAME version as the reference?

    A remix usually shares the same vocals and harmonic content but may differ
    in BPM, energy, and arrangement. Strategy:
    - MERT cosine is the strongest same-song signal: high -> same song (could be
      same or different version); low -> different song.
    - If same song, use BPM / key / energy / cue-pattern deltas to decide if it
      is the SAME version or a DIFFERENT version.
    - AMBIGUOUS when signals conflict or MERT is unavailable and Essentia is
      weak.
    """
    ref = inputs.reference
    cand = inputs.candidate
    signals: list[str] = []

    # 1. Same-song gate via MERT (if available). MERT cosine is the strongest
    # signal that the two recordings are the same song at all.
    if inputs.mert_cosine is not None:
        if inputs.mert_cosine < _MERT_COSINE_DIFFERENT_MAX:
            return VersionVerdict.DIFFERENT_VERSION
        if inputs.mert_cosine >= _MERT_COSINE_SAME_MIN:
            signals.append("mert_same_song")
        else:
            signals.append("mert_weak")

    # 2. Version-axis deltas: same version should be very close on BPM and key,
    # different version (remix) can differ.
    bpm_diff = (
        abs(ref.bpm - cand.bpm)
        if ref.bpm is not None and cand.bpm is not None
        else None
    )
    key_diff = _key_distance(ref.key_pc, ref.key_mode, cand.key_pc, cand.key_mode)
    energy_diff = (
        abs(ref.energy - cand.energy)
        if ref.energy is not None and cand.energy is not None
        else None
    )

    if bpm_diff is not None:
        if bpm_diff <= _BPM_DIFF_SAME_MAX:
            signals.append("bpm_same")
        elif bpm_diff >= _BPM_DIFF_DIFFERENT_MIN:
            signals.append("bpm_different")
        else:
            signals.append("bpm_ambiguous")

    if key_diff >= 0:
        if key_diff == _KEY_PC_DIFF_SAME_MAX:
            signals.append("key_same")
        elif key_diff >= _KEY_PC_DIFF_DIFFERENT_MIN:
            signals.append("key_different")
        else:
            signals.append("key_ambiguous")

    if energy_diff is not None:
        if energy_diff >= _ENERGY_DIFF_DIFFERENT_MIN:
            signals.append("energy_different")
        else:
            signals.append("energy_same")

    # 3. Structural cue-pattern delta (only meaningful if same song).
    if inputs.reference_structure and inputs.candidate_structure:
        cue_diff = _cue_pattern_diff(
            inputs.reference_structure, inputs.candidate_structure
        )
        if cue_diff >= 0.50:
            signals.append("cues_different")
        elif cue_diff <= 0.20:
            signals.append("cues_same")
        else:
            signals.append("cues_ambiguous")

    # 4. Aggregate into a verdict.
    # Aggregate signals. Same-song evidence (MERT) lets us trust a single
    # version-axis delta as a remix; without MERT, we require at least two
    # Essentia/structure deltas to avoid misclassifying different songs as
    # remixes.
    different = {
        "bpm_different",
        "key_different",
        "energy_different",
        "cues_different",
    }
    same = {"bpm_same", "key_same", "energy_same", "cues_same"}
    n_different = sum(1 for s in signals if s in different)
    n_same = sum(1 for s in signals if s in same)
    mert_same = "mert_same_song" in signals

    if n_different >= 2 or (n_different >= 1 and mert_same):
        return VersionVerdict.DIFFERENT_VERSION
    if n_same >= 2 and n_different == 0:
        return VersionVerdict.SAME_VERSION
    return VersionVerdict.AMBIGUOUS


# -----------------------------------------------------------------------------
# Convenience: same-song check (used by the cascade before version/variant)
# -----------------------------------------------------------------------------


def same_song(inputs: VersionVerifyInputs) -> AxisVerdict:
    """High-level same-song gate. Returns SAME only when MERT or a strong
    Essentia consensus says the two recordings are the same song."""
    if inputs.mert_cosine is not None:
        if inputs.mert_cosine >= _MERT_COSINE_SAME_MIN:
            return AxisVerdict.SAME
        if inputs.mert_cosine < _MERT_COSINE_DIFFERENT_MAX:
            return AxisVerdict.DIFFERENT
        return AxisVerdict.AMBIGUOUS

    ref = inputs.reference
    cand = inputs.candidate
    bpm_ok = (
        ref.bpm is not None
        and cand.bpm is not None
        and abs(ref.bpm - cand.bpm) <= _BPM_DIFF_DIFFERENT_MIN
    )
    key_ok = _key_distance(ref.key_pc, ref.key_mode, cand.key_pc, cand.key_mode) in (
        0,
        1,
        2,
    )
    if bpm_ok and key_ok:
        return AxisVerdict.SAME
    return AxisVerdict.AMBIGUOUS
