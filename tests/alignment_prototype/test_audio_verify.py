from __future__ import annotations

import numpy as np

from workspaces.alignment_prototype.candidate_arbiter.audio_verify import (
    verify_alignment,
)


def _normalized(seed: int, rows: int = 120, dims: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(rows, dims)).astype(np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True) + 1e-9
    return values


def test_audio_pair_verifier_prefers_pinned_matching_windows() -> None:
    ref = _normalized(0)
    mix = _normalized(1)
    mix[40:64] = ref[10:34]

    correct = verify_alignment(
        mix,
        ref,
        mix_start_bin=40,
        ref_start_bin=10,
        window_bins=24,
    )
    wrong = verify_alignment(
        mix,
        ref,
        mix_start_bin=70,
        ref_start_bin=10,
        window_bins=24,
    )

    assert correct is not None and wrong is not None
    assert correct.match_mean > 0.99
    assert correct.match_mean > wrong.match_mean
    assert correct.margin > wrong.margin
    assert correct.continuity > 0.95


def test_audio_pair_verifier_reports_support_and_background() -> None:
    ref = _normalized(2)
    mix = ref.copy()

    result = verify_alignment(
        mix,
        ref,
        mix_start_bin=30,
        ref_start_bin=30,
        window_bins=20,
        negative_shift_bins=(-20, 20),
    )

    assert result is not None
    assert result.support_bins == 20
    assert result.background_count == 2
    assert result.match_p10 > 0.99


def test_audio_pair_verifier_abstains_on_short_or_out_of_bounds_window() -> None:
    values = _normalized(3, rows=20)
    assert (
        verify_alignment(
            values,
            values,
            mix_start_bin=15,
            ref_start_bin=0,
            window_bins=10,
        )
        is None
    )
