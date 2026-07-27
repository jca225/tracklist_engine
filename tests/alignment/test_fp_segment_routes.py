import pytest

from alignment.fp_segments.routes import lane, validate_route


def test_instrumental_lane_is_strictly_like_for_like():
    route = lane("instrumental")
    assert route.mix_file == "mix_instrumental.flac"
    assert route.mix_channel == "instrumental"
    assert route.reference_stem == "instrumental"
    assert route.claimed_stems == frozenset({"instrumental"})


def test_vocal_lane_is_strictly_like_for_like():
    route = lane("vocal")
    assert route.mix_file == "mix_vocals.flac"
    assert route.mix_channel == "vocals"
    assert route.reference_stem == "vocals"
    assert route.claimed_stems == frozenset({"acappella"})


@pytest.mark.parametrize(
    ("mix_channel", "reference_stem"),
    [
        ("instrumental", "regular"),
        ("instrumental", "vocals"),
        ("vocals", "instrumental"),
        ("vocals", "regular"),
        ("full", "regular"),
    ],
)
def test_unlike_or_full_routes_are_forbidden(mix_channel, reference_stem):
    with pytest.raises(ValueError, match="forbidden"):
        validate_route(
            mix_channel=mix_channel,
            reference_stem=reference_stem,
        )
