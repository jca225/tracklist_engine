"""Tests for the heuristic audio-side identity verifiers."""

from __future__ import annotations

from analysis.identity_verify import (
    AxisVerdict,
    EssentiaFeatures,
    StructureFeatures,
    VariantVerdict,
    VariantVerifyInputs,
    VersionVerdict,
    VersionVerifyInputs,
    same_song,
    verify_variant,
    verify_version,
)


def _structure(
    *, duration: float, cues: tuple[float, ...] = (), measures: tuple[float, ...] = ()
) -> StructureFeatures:
    return StructureFeatures(
        duration_s=duration,
        cue_points_s=cues,
        measure_times_s=measures,
    )


# ── variant verifier ─────────────────────────────────────────────────────────


def test_variant_extended_longer_with_intro_outro_growth():
    """A 5:30 extended vs 3:45 radio, with the extra length in intro/outro, is
    DIFFERENT_VARIANT."""
    radio = _structure(
        duration=225.0,
        cues=(0.0, 30.0, 195.0),
        measures=(0.0, 0.5, 1.0),
    )
    extended = _structure(
        duration=330.0,
        cues=(0.0, 60.0, 300.0),
        measures=(0.0, 0.5, 1.0),
    )
    assert (
        verify_variant(VariantVerifyInputs(radio, extended))
        is VariantVerdict.DIFFERENT_VARIANT
    )


def test_variant_same_length_same_cues_is_same_variant():
    """Two files with identical duration and cue count are SAME_VARIANT."""
    a = _structure(duration=225.0, cues=(0.0, 30.0, 195.0))
    b = _structure(duration=225.0, cues=(0.0, 30.0, 195.0))
    assert verify_variant(VariantVerifyInputs(a, b)) is VariantVerdict.SAME_VARIANT


def test_variant_shorter_than_reference_is_ambiguous():
    """A candidate shorter than the reference is not an extended variant; it
    could be a radio edit or a different cut, so AMBIGUOUS."""
    ref = _structure(duration=225.0)
    cand = _structure(duration=200.0)
    assert verify_variant(VariantVerifyInputs(ref, cand)) is VariantVerdict.AMBIGUOUS


def test_variant_dramatically_different_length_is_ambiguous():
    """A 2× length difference is not a clean extended variant; AMBIGUOUS."""
    ref = _structure(duration=225.0)
    cand = _structure(duration=480.0)
    assert verify_variant(VariantVerifyInputs(ref, cand)) is VariantVerdict.AMBIGUOUS


# ── version verifier ─────────────────────────────────────────────────────────


def test_version_same_bpm_key_energy_is_same_version():
    """Identical BPM, key, energy, and high MERT cosine -> SAME_VERSION."""
    ref = EssentiaFeatures(bpm=128.0, key_pc=7, key_mode="minor", energy=0.8)
    cand = EssentiaFeatures(bpm=128.0, key_pc=7, key_mode="minor", energy=0.81)
    inputs = VersionVerifyInputs(
        reference=ref,
        candidate=cand,
        mert_cosine=0.85,
    )
    assert verify_version(inputs) is VersionVerdict.SAME_VERSION


def test_version_remix_differs_in_bpm_and_energy():
    """A remix: same song (high MERT) but different BPM and energy -> DIFFERENT_VERSION."""
    ref = EssentiaFeatures(bpm=128.0, key_pc=7, key_mode="minor", energy=0.8)
    cand = EssentiaFeatures(bpm=140.0, key_pc=7, key_mode="minor", energy=0.55)
    inputs = VersionVerifyInputs(
        reference=ref,
        candidate=cand,
        mert_cosine=0.82,
    )
    assert verify_version(inputs) is VersionVerdict.DIFFERENT_VERSION


def test_version_different_song_low_mert_is_different_version():
    """Low MERT cosine means different song -> DIFFERENT_VERSION immediately."""
    ref = EssentiaFeatures(bpm=128.0, key_pc=7, key_mode="minor")
    cand = EssentiaFeatures(bpm=120.0, key_pc=2, key_mode="major")
    inputs = VersionVerifyInputs(
        reference=ref,
        candidate=cand,
        mert_cosine=0.40,
    )
    assert verify_version(inputs) is VersionVerdict.DIFFERENT_VERSION


def test_version_mert_weak_but_same_bpm_key_is_same_version():
    """Weak MERT but Essentia consensus says same -> SAME_VERSION."""
    ref = EssentiaFeatures(bpm=128.0, key_pc=7, key_mode="minor", energy=0.8)
    cand = EssentiaFeatures(bpm=128.0, key_pc=7, key_mode="minor", energy=0.79)
    inputs = VersionVerifyInputs(
        reference=ref,
        candidate=cand,
        mert_cosine=0.55,
    )
    assert verify_version(inputs) is VersionVerdict.SAME_VERSION


def test_version_conflicting_signals_are_ambiguous():
    """BPM says same, key says different, MERT weak -> AMBIGUOUS."""
    ref = EssentiaFeatures(bpm=128.0, key_pc=7, key_mode="minor", energy=0.8)
    cand = EssentiaFeatures(bpm=128.1, key_pc=10, key_mode="major", energy=0.8)
    inputs = VersionVerifyInputs(
        reference=ref,
        candidate=cand,
        mert_cosine=0.60,
    )
    assert verify_version(inputs) is VersionVerdict.AMBIGUOUS


def test_version_no_mert_same_bpm_key_is_same_version():
    """When MERT is unavailable, strong Essentia consensus (same BPM + key)
    is enough for SAME_VERSION."""
    ref = EssentiaFeatures(bpm=128.0, key_pc=7, key_mode="minor")
    cand = EssentiaFeatures(bpm=128.0, key_pc=7, key_mode="minor")
    inputs = VersionVerifyInputs(reference=ref, candidate=cand)
    assert verify_version(inputs) is VersionVerdict.SAME_VERSION


def test_version_no_mert_single_different_signal_is_ambiguous():
    """When MERT is unavailable, a single Essentia delta is not enough to
    declare a different version — that could just be a different song."""
    ref = EssentiaFeatures(bpm=128.0, key_pc=7, key_mode="minor")
    cand = EssentiaFeatures(bpm=150.0, key_pc=7, key_mode="minor")
    inputs = VersionVerifyInputs(reference=ref, candidate=cand)
    assert verify_version(inputs) is VersionVerdict.AMBIGUOUS


def test_version_no_mert_two_different_signals_is_different_version():
    """When MERT is unavailable, two independent Essentia deltas (BPM + key) are
    enough for DIFFERENT_VERSION."""
    ref = EssentiaFeatures(bpm=128.0, key_pc=7, key_mode="minor")
    cand = EssentiaFeatures(bpm=150.0, key_pc=10, key_mode="major")
    inputs = VersionVerifyInputs(reference=ref, candidate=cand)
    assert verify_version(inputs) is VersionVerdict.DIFFERENT_VERSION


# ── same-song gate ───────────────────────────────────────────────────────────


def test_same_song_high_mert_returns_same():
    ref = EssentiaFeatures(bpm=128.0, key_pc=7, key_mode="minor")
    cand = EssentiaFeatures(bpm=140.0, key_pc=7, key_mode="minor")
    inputs = VersionVerifyInputs(reference=ref, candidate=cand, mert_cosine=0.80)
    assert same_song(inputs) is AxisVerdict.SAME


def test_same_song_low_mert_returns_different():
    ref = EssentiaFeatures(bpm=128.0, key_pc=7, key_mode="minor")
    cand = EssentiaFeatures(bpm=128.0, key_pc=7, key_mode="minor")
    inputs = VersionVerifyInputs(reference=ref, candidate=cand, mert_cosine=0.40)
    assert same_song(inputs) is AxisVerdict.DIFFERENT


def test_same_song_no_mert_close_bpm_key_returns_same():
    ref = EssentiaFeatures(bpm=128.0, key_pc=7, key_mode="minor")
    cand = EssentiaFeatures(bpm=129.0, key_pc=8, key_mode="minor")
    inputs = VersionVerifyInputs(reference=ref, candidate=cand)
    assert same_song(inputs) is AxisVerdict.SAME


def test_same_song_no_mert_far_bpm_returns_ambiguous():
    ref = EssentiaFeatures(bpm=128.0, key_pc=7, key_mode="minor")
    cand = EssentiaFeatures(bpm=150.0, key_pc=2, key_mode="major")
    inputs = VersionVerifyInputs(reference=ref, candidate=cand)
    assert same_song(inputs) is AxisVerdict.AMBIGUOUS
