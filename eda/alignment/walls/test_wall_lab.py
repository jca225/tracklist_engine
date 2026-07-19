"""TDD for the pure distinguishability engine (the structure-wall research tool).

The audio/IO wrappers are thin and smoke-tested against real sets in the
notebook; the *logic* that decides "is this repeat physically winnable?" is
here and must be correct: a similarity curve, its top-two peaks, and the
best-vs-runner-up margin that separates a recoverable miss from the ceiling.
"""

from __future__ import annotations

import numpy as np

from eda.alignment.walls.wall_lab import (
    classify_distinguishability,
    feature_for_stem,
    frames_to_s,
    similarity_curve,
    top_two_peaks,
)


def test_feature_for_stem_routes_vocals_to_hubert():
    # chroma breaks on re-pitched acappellas → auto must pick HuBERT for vocals
    assert feature_for_stem("acappella") == "hubert"
    assert feature_for_stem("instrumental") == "chroma"
    assert feature_for_stem("regular") == "chroma"
    # explicit override wins over auto-routing
    assert feature_for_stem("acappella", "chroma") == "chroma"


def test_similarity_curve_peaks_where_window_matches():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((12, 100)).astype(np.float32)
    win = ref[:, 30:40]  # the window is literally a slice of the ref
    curve = similarity_curve(win, ref)
    assert curve.shape == (100 - 10 + 1,)
    assert int(np.argmax(curve)) == 30
    assert curve[30] > 0.99  # near-perfect cosine match at the true position


def test_similarity_curve_ambiguous_ref_has_two_equal_peaks():
    rng = np.random.default_rng(1)
    block = rng.standard_normal((12, 10)).astype(np.float32)
    ref = rng.standard_normal((12, 100)).astype(np.float32)
    ref[:, 20:30] = block  # same content appears twice → a true "repeat"
    ref[:, 70:80] = block
    curve = similarity_curve(block, ref)
    # both copies score near 1.0 → the choice is physically ambiguous
    assert curve[20] > 0.99 and curve[70] > 0.99


def test_top_two_peaks_separates_by_min_sep():
    scores = np.zeros(100, dtype=np.float32)
    scores[20] = 0.9
    scores[22] = 0.85  # near the best — must NOT count as the runner-up
    scores[70] = 0.6  # far away — this is the real runner-up
    best_i, best_v, second_i, second_v = top_two_peaks(scores, min_sep=10)
    assert best_i == 20 and abs(best_v - 0.9) < 1e-6
    assert second_i == 70 and abs(second_v - 0.6) < 1e-6


def test_classify_distinguishability_margin_bands():
    # large margin: the true position stands out → recoverable
    assert (
        classify_distinguishability(best=0.9, runner_up=0.5, margin_thresh=0.1)
        == "recoverable"
    )
    # tiny margin: two near-equal matches → physically ambiguous (the ceiling)
    assert (
        classify_distinguishability(best=0.92, runner_up=0.90, margin_thresh=0.1)
        == "ambiguous"
    )
    # exactly at threshold counts as ambiguous (conservative — don't over-claim winnable)
    assert (
        classify_distinguishability(best=0.80, runner_up=0.70, margin_thresh=0.1)
        == "ambiguous"
    )


def test_frames_to_s_uses_sr_hop():
    # 43.066... frames per second at SR=22050, HOP=512
    assert abs(frames_to_s(43) - 43 * 512 / 22050) < 1e-9
    assert frames_to_s(0) == 0.0
