from workspaces.alignment_prototype.external.eval_bench import stratum


def test_stratum_parses_warp_and_effect():
    assert stratum("set042mix3-resample-bass-07") == ("resample", "bass")
    assert stratum("set001mix3-none-none-00") == ("none", "none")
    assert stratum("set099mix3-stretch-distortion-19") == ("stretch", "distortion")


def test_stratum_unknown_on_garbage():
    assert stratum("not-a-real-id") == ("unknown", "unknown")
    assert stratum("set042mix3-warpX-effectY-01") == ("unknown", "unknown")
