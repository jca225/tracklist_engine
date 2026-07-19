"""Unit tests for the oracle instance-selection separability harness (Task A).

Pure ranking / enumeration / metric logic — no audio. Real feature extraction is
exercised by the integration run, not here.
"""

from __future__ import annotations

import numpy as np

from workspaces.alignment_prototype.evals import instance_separability as ins


# --------------------------------------------------------------------------
# Unit 2 — candidate enumeration
# --------------------------------------------------------------------------


def test_enumerate_marks_gt_instance_within_its_interval():
    # fiber 5 has two instances: frames [0,10) and [20,30); GT played at 5.0s.
    labels = np.array([5] * 10 + [-1] * 10 + [5] * 10, dtype=int)
    cands = ins.enumerate_candidates(labels, label_hz=1.0, gt_ref_start_s=5.0)
    assert len(cands) == 2
    gt = [c for c in cands if c.is_gt_instance]
    assert len(gt) == 1
    assert gt[0].ref_start_s == 0.0 and gt[0].ref_end_s == 10.0


def test_enumerate_returns_empty_when_gt_frame_not_in_a_fiber():
    # GT ref_start falls on silence (-1): no repeat class, nothing to select.
    labels = np.array([5] * 10 + [-1] * 10, dtype=int)
    cands = ins.enumerate_candidates(labels, label_hz=1.0, gt_ref_start_s=15.0)
    assert cands == []


def test_enumerate_single_member_fiber_is_not_selectable():
    # fiber 5 occurs once; GT in it. One candidate -> not a selection problem.
    labels = np.array([5] * 10 + [-1] * 10, dtype=int)
    cands = ins.enumerate_candidates(labels, label_hz=1.0, gt_ref_start_s=5.0)
    assert len(cands) == 1
    assert not ins.is_selectable(cands)


# --------------------------------------------------------------------------
# Unit 4 — ranking + separability metrics
# --------------------------------------------------------------------------


def _cand(ref_start, is_gt, **feats):
    return ins.Candidate(
        ref_start_s=ref_start,
        ref_end_s=ref_start + 10.0,
        label=5,
        is_gt_instance=is_gt,
        features=dict(feats),
    )


def test_top1_score_one_when_gt_has_strict_highest_feature():
    cands = [_cand(0.0, True, hubert=0.9), _cand(20.0, False, hubert=0.3)]
    assert ins.top1_score(cands, "hubert") == 1.0


def test_top1_score_zero_when_rival_wins():
    cands = [_cand(0.0, True, hubert=0.3), _cand(20.0, False, hubert=0.9)]
    assert ins.top1_score(cands, "hubert") == 0.0


def test_top1_score_none_features_rank_last():
    # a candidate with a missing feature must never outrank a scored one.
    cands = [_cand(0.0, True, hubert=0.1), _cand(20.0, False, hubert=None)]
    assert ins.top1_score(cands, "hubert") == 1.0


def test_top1_score_tie_is_fair_not_order_dependent():
    # GT tied with one rival -> expected credit 0.5 under random tie-break, NOT 1.0
    # just because GT is listed first. This is the earliest-instance confound guard.
    tied = [_cand(0.0, True, hubert=0.8), _cand(20.0, False, hubert=0.8)]
    assert ins.top1_score(tied, "hubert") == 0.5
    # order must not matter
    tied_rev = [_cand(20.0, False, hubert=0.8), _cand(0.0, True, hubert=0.8)]
    assert ins.top1_score(tied_rev, "hubert") == 0.5


def test_gap_recovered_counts_only_selectable_rows():
    rows = [
        [_cand(0.0, True, hubert=0.9), _cand(20.0, False, hubert=0.1)],  # correct
        [_cand(0.0, True, hubert=0.1), _cand(20.0, False, hubert=0.9)],  # wrong
        [_cand(0.0, True, hubert=0.5)],  # single-member -> excluded
    ]
    n_correct, n_selectable, frac = ins.gap_recovered(rows, "hubert")
    assert (n_correct, n_selectable) == (1, 2)
    assert frac == 0.5


def test_position_baseline_earliest_and_latest():
    # rows ordered by ref_start; 'earliest' credits GT when it is the first instance.
    rows = [
        [_cand(0.0, True), _cand(20.0, False)],  # GT earliest
        [_cand(0.0, False), _cand(20.0, True)],  # GT latest
    ]
    assert ins.position_baseline(rows, "earliest") == 0.5
    assert ins.position_baseline(rows, "latest") == 0.5


def test_zscore_combo_is_majority_direction():
    # With 2 candidates every non-constant feature z-scores to +/-1, so the combo
    # is decided by MAJORITY direction, not magnitude. GT leads on 2 of 3 features
    # (hubert, fp); mu is tied (constant -> 0 contribution) -> GT wins the combo.
    cands = [
        _cand(0.0, True, hubert=1.0, mu=0.5, fp=0.9),
        _cand(20.0, False, hubert=0.0, mu=0.5, fp=0.1),
    ]
    ins.assign_combo(cands)
    assert ins.top1_score(cands, "combo") == 1.0


def test_transfer_learned_weights_rank_gt_top_on_separable_data():
    # train rows where hubert alone separates; a fitted ranker should pick gt top
    # on a held-out row with the same structure.
    train = [
        [
            _cand(0.0, True, hubert=0.9, mu=0.5, fp=0.5),
            _cand(20.0, False, hubert=0.2, mu=0.5, fp=0.5),
        ],
        [
            _cand(0.0, True, hubert=0.8, mu=0.4, fp=0.6),
            _cand(20.0, False, hubert=0.1, mu=0.6, fp=0.4),
        ],
    ]
    test = [
        [
            _cand(0.0, True, hubert=0.95, mu=0.5, fp=0.5),
            _cand(20.0, False, hubert=0.15, mu=0.5, fp=0.5),
        ],
    ]
    weights = ins.fit_ranker(train)
    n_correct, n_sel, frac = ins.transfer_gap_recovered(test, weights)
    assert (n_correct, n_sel, frac) == (1, 1, 1.0)


# --------------------------------------------------------------------------
# Unit 3 — per-candidate feature extraction (pure, over injected arrays)
# --------------------------------------------------------------------------


def test_candidate_features_hubert_peaks_at_true_instance():
    # A 4-frame mix window is embedded verbatim at ref frame 0 (GT instance) and a
    # content-DIFFERENT pattern at ref frame 20 (rival). The matched filter is
    # energy-normalized (scale-invariant), so the rival must differ in direction,
    # not amplitude, to score lower — mirroring the real risk that near-identical
    # fiber siblings are indistinguishable.
    fps = 5.0
    rng = np.arange(8.0).reshape(2, 4)  # (D=2, n=4) mix window
    ref = np.zeros((2, 40), dtype=float)
    ref[:, 0:4] = rng  # true instance, exact content
    ref[:, 20:24] = rng[:, ::-1]  # rival, same energy, different content
    ctx = ins.FeatureContext(
        mix_win=rng.astype(float),
        ref_feat=ref,
        stretch=1.0,
        labels=np.array([5] * 25 + [-1] * 15),
        label_hz=fps,
        mu=np.linspace(0.2, 0.9, 40),
        fp_votes={0: 10, 20: 2},
        fps=fps,
        fp_hz=fps,
    )
    gt = ins.Candidate(0.0, 0.8, 5, True)
    rival = ins.Candidate(4.0, 4.8, 5, False)
    fg = ins.candidate_features(gt, ctx)
    fr = ins.candidate_features(rival, ctx)
    assert fg["hubert"] > fr["hubert"]
    assert fg["mu"] < fr["mu"]  # mu is monotone up here — sanity that we index it
    assert fg["fp"] > fr["fp"]  # more vote mass at offset 0 than at offset 20


def test_candidate_features_out_of_range_offset_is_none():
    ctx = ins.FeatureContext(
        mix_win=np.ones((2, 4)),
        ref_feat=np.ones((2, 10)),
        stretch=1.0,
        labels=np.array([5] * 10),
        label_hz=5.0,
        mu=np.ones(10),
        fp_votes={},
        fps=5.0,
        fp_hz=5.0,
    )
    # candidate ref_start beyond the matched-filter curve -> hubert None
    far = ins.Candidate(100.0, 100.8, 5, False)
    f = ins.candidate_features(far, ctx)
    assert f["hubert"] is None
