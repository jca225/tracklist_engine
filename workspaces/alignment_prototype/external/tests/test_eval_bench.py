import math

import numpy as np
import pandas as pd

from workspaces.alignment_prototype.external.eval_bench import (
    HOP,
    SR,
    GTSpan,
    Pred,
    Sample,
    is_abstain,
    make_fused,
    method_dtw,
    score_sample,
    stratum,
    summary_by_stratum,
)


def test_stratum_parses_warp_and_effect():
    assert stratum("set042mix3-resample-bass-07") == ("resample", "bass")
    assert stratum("set001mix3-none-none-00") == ("none", "none")
    assert stratum("set099mix3-stretch-distortion-19") == ("stretch", "distortion")


def test_stratum_unknown_on_garbage():
    assert stratum("not-a-real-id") == ("unknown", "unknown")
    assert stratum("set042mix3-warpX-effectY-01") == ("unknown", "unknown")


def test_is_abstain():
    assert is_abstain(Pred(float("nan"), float("nan"), 3.0)) is True
    assert is_abstain(Pred(12.0, 1.0, 50.0)) is False


def test_score_sample_marks_abstain_and_excludes_from_error():
    mix = np.zeros((12, 400), dtype=np.float32)
    s = Sample("m", mix, {0: np.zeros((12, 40), np.float32)}, [GTSpan(0, 10.0, 1.0)])
    rows, _ = score_sample(s, {0: Pred(float("nan"), float("nan"), 1.0)})
    assert len(rows) == 1
    assert rows[0]["abstained"] is True
    assert math.isnan(rows[0]["set_start_err"])


def test_make_fused_is_a_method_factory():
    m0 = make_fused(0.0)  # André-mode
    m1 = make_fused(100.0)  # open-mode
    assert callable(m0) and callable(m1)
    # feature-only sample -> method_fused returns {} (no audio), both modes
    mix = np.zeros((12, 400), dtype=np.float32)
    s = Sample("m", mix, {0: np.zeros((12, 40), np.float32)}, [GTSpan(0, 1.0, 1.0)])
    assert m0(s) == {} and m1(s) == {}


def test_summary_by_stratum_groups_and_reports_abstain():
    df = pd.DataFrame(
        [
            dict(
                mix_id="set1mix3-none-none-00",
                track=0,
                set_start_err=1.0,
                tempo_err=0.01,
                tempo_pct=1.0,
                peak=50,
                abstained=False,
            ),
            dict(
                mix_id="set1mix3-none-none-00",
                track=1,
                set_start_err=float("nan"),
                tempo_err=float("nan"),
                tempo_pct=float("nan"),
                peak=2,
                abstained=True,
            ),
            dict(
                mix_id="set2mix3-resample-bass-01",
                track=0,
                set_start_err=3.0,
                tempo_err=0.02,
                tempo_pct=2.0,
                peak=40,
                abstained=False,
            ),
        ]
    )
    df.attrs["identity_acc"] = float("nan")
    out = summary_by_stratum({"fused": df})
    none = out[
        (out.method == "fused") & (out.warp == "none") & (out.effect == "none")
    ].iloc[0]
    assert none.n == 2
    assert none.abstain_pct == 50.0
    assert none.set_start_MAE_s == 1.0  # committed-only mean
    allrow = out[(out.method == "fused") & (out.warp == "ALL")].iloc[0]
    assert allrow.n == 3


def test_method_dtw_recovers_planted_offset():
    rng = np.random.default_rng(0)
    D, tlen = 12, 120
    tf = rng.random((D, tlen)).astype(np.float32)
    Tm = 800
    mix = rng.random((D, Tm)).astype(np.float32) * 0.05
    start_f = 300
    mix[:, start_f : start_f + tlen] += tf  # planted, tempo 1.0, no stretch
    s = Sample("synthDTW", mix, {0: tf}, [GTSpan(0, start_f * HOP / SR, 1.0)])
    preds = method_dtw(s)
    assert 0 in preds
    # within ~1s of the planted start
    assert abs(preds[0].set_start_s - start_f * HOP / SR) < 1.0
    assert abs(preds[0].tempo_ratio - 1.0) < 0.15


def test_method_dtw_abstains_on_tiny_track():
    mix = np.zeros((12, 400), dtype=np.float32)
    s = Sample("m", mix, {0: np.zeros((12, 4), np.float32)}, [GTSpan(0, 1.0, 1.0)])
    preds = method_dtw(s)
    assert math.isnan(preds[0].set_start_s)


def test_method_dtw_score_is_path_average_not_best_cell():
    # Discriminating test: track is UNRELATED to the mix except ONE perfect cell.
    # best-cell score (1 - C.min()) -> ~1.0; path-average score (the fix) -> the
    # background cosine cost, well below 0.9. Asserting < 0.9 FAILS on the old
    # best-cell formula and PASSES on the path-average formula.
    rng = np.random.default_rng(9)
    D, tlen, Tm = 12, 80, 700
    tf = rng.random((D, tlen)).astype(np.float32)
    mix = rng.random((D, Tm)).astype(np.float32)  # unrelated background
    mix[:, 400] = tf[:, 40]  # single perfect cell -> C.min() == 0
    s = Sample("s", mix, {0: tf}, [GTSpan(0, 400 * HOP / SR, 1.0)])
    preds = method_dtw(s)
    assert not math.isnan(preds[0].score)
    assert preds[0].score < 0.9  # best-cell (~1.0) would fail this


def test_method_fused_resample_returns_empty_without_audio():
    from workspaces.alignment_prototype.external.eval_bench import method_fused_resample

    mix = np.zeros((12, 400), dtype=np.float32)
    s = Sample("m", mix, {0: np.zeros((12, 40), np.float32)}, [GTSpan(0, 1.0, 1.0)])
    assert method_fused_resample(s) == {}  # no mix_path -> audio method no-ops


def test_identity_excludes_abstained_spans():
    # one committed correct span + one abstained span. With distractors that
    # out-score the abstained pred, identity should be 1.0 (1/1 committed), not
    # 0.5 (1/2 counting the abstention as a miss).
    # Key: track 1 is abstained (NaN set_start), so it should NOT be counted in
    # either hits or denominator.
    mix = np.ones((12, 400), dtype=np.float32)
    tfa = np.ones((12, 40), dtype=np.float32)
    tfb = np.ones((12, 40), dtype=np.float32)
    distractor = np.full((12, 40), 0.5, dtype=np.float32)
    s = Sample(
        "m",
        mix,
        {0: tfa, 1: tfb},
        [GTSpan(0, 1.0, 1.0), GTSpan(1, 2.0, 1.0)],
        distractor_feats={"d0": distractor},
    )
    preds = {
        0: Pred(1.0, 1.0, 100.0),  # committed, out-scores distractor (~1.0)
        1: Pred(float("nan"), float("nan"), 0.5),  # abstained, loses to distractor
    }
    _, id_ok = score_sample(s, preds)
    # Old code: hits = (100.0 > 1.0) + (0.5 > 1.0) = 1 + 0 = 1, id_ok = 1/2 = 0.5
    # New code: committed_gt = [GTSpan(0, ...)], hits = 1, id_ok = 1/1 = 1.0
    assert id_ok == 1.0
