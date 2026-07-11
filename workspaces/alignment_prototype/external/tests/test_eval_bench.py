from workspaces.alignment_prototype.external.eval_bench import stratum


def test_stratum_parses_warp_and_effect():
    assert stratum("set042mix3-resample-bass-07") == ("resample", "bass")
    assert stratum("set001mix3-none-none-00") == ("none", "none")
    assert stratum("set099mix3-stretch-distortion-19") == ("stretch", "distortion")


def test_stratum_unknown_on_garbage():
    assert stratum("not-a-real-id") == ("unknown", "unknown")
    assert stratum("set042mix3-warpX-effectY-01") == ("unknown", "unknown")


import math
import numpy as np
from workspaces.alignment_prototype.external.eval_bench import (
    GTSpan,
    Pred,
    Sample,
    is_abstain,
    make_fused,
    score_sample,
)


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
