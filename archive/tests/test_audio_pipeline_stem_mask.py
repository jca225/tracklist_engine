"""Stage-4 stem-mask classifier tests."""
from __future__ import annotations

import pytest

from archive.audio_pipeline.alignment.stem_mask import classify


# ---- category boundaries ---------------------------------------------------

def test_empty_rates_returns_none() -> None:
    assert classify({}) == "none"


def test_all_below_threshold_returns_none() -> None:
    assert classify({"full": 0.1, "vocals": 0.2, "drums": 0.1, "bass": 0.1, "other": 0.1}) == "none"


def test_only_vocals_aligns_is_acappella() -> None:
    assert classify({"vocals": 0.7, "drums": 0.1, "bass": 0.1, "other": 0.1, "full": 0.2}) == "acappella"


def test_vocals_missing_drums_strong_is_instrumental() -> None:
    assert classify({"vocals": 0.15, "drums": 0.6, "bass": 0.1, "other": 0.1, "full": 0.2}) == "instrumental"


def test_all_four_stems_aligned_is_full() -> None:
    assert classify({
        "vocals": 0.7, "drums": 0.6, "bass": 0.6, "other": 0.6, "full": 0.6,
    }) == "full"


def test_vocals_plus_full_stem_is_full() -> None:
    # Full stem high even if some individual instrument is below threshold
    assert classify({"vocals": 0.7, "drums": 0.3, "bass": 0.3, "other": 0.3, "full": 0.6}) == "full"


def test_vocals_plus_one_instrument_is_partial() -> None:
    # Not enough instruments for 'full', not empty of instruments for 'acappella'
    assert classify({"vocals": 0.7, "drums": 0.5, "bass": 0.1, "other": 0.1, "full": 0.2}) == "partial"


# ---- threshold parameterization --------------------------------------------

def test_threshold_can_be_tuned_down_to_catch_marginal_alignments() -> None:
    rates = {"vocals": 0.3, "drums": 0.1, "bass": 0.1, "other": 0.1, "full": 0.15}
    assert classify(rates, threshold=0.4) == "none"
    assert classify(rates, threshold=0.25) == "acappella"


# ---- regression: real samples from Vol. 11 ---------------------------------

def test_vol11_row83_vocals_win_classifies_acappella() -> None:
    # Actual row: full=0.09, vocals=0.63, drums=0.15, bass=0.30, other=0.24
    # → vocals cleared but nothing else did → acappella
    rates = {"full": 0.09, "vocals": 0.63, "drums": 0.15, "bass": 0.30, "other": 0.24}
    assert classify(rates) == "acappella"


def test_vol11_row150_drums_win_classifies_instrumental() -> None:
    # Actual row: full=0.06, vocals=0.30, drums=0.58, bass=0.19, other=0.08
    rates = {"full": 0.06, "vocals": 0.30, "drums": 0.58, "bass": 0.19, "other": 0.08}
    assert classify(rates) == "instrumental"


def test_vol11_row122_bass_only_classifies_instrumental() -> None:
    # full=0.20, vocals=0.38, drums=0.30, bass=0.63, other=0.14
    rates = {"full": 0.20, "vocals": 0.38, "drums": 0.30, "bass": 0.63, "other": 0.14}
    assert classify(rates) == "instrumental"


# ---- margin guard (fixes Good Grief misclassification) ---------------------

def test_margin_guard_reclassifies_borderline_acappella_as_partial() -> None:
    """Real Vol. 11 row 2 — 'Good Grief (Instrumental Mix)'. Vocals barely
    over threshold at 0.416, drums right behind at 0.355 (margin 0.061).
    Old classifier said 'acappella'; margin guard says 'partial' which is
    more honest — and the stem-energy classifier then corrects the call."""
    rates = {"full": 0.264, "vocals": 0.416, "drums": 0.355, "bass": 0.061, "other": 0.295}
    assert classify(rates, margin=0.08) == "partial"


def test_margin_guard_does_not_fire_on_clear_wins() -> None:
    # Vocals 0.7 vs next 0.3 — margin 0.4, well above the 0.08 guard.
    rates = {"full": 0.2, "vocals": 0.7, "drums": 0.1, "bass": 0.1, "other": 0.3}
    assert classify(rates) == "acappella"


# ---- tracklist version parser ----------------------------------------------

def test_parse_version_tag_instrumental_mix() -> None:
    from archive.audio_pipeline.alignment.stem_mask import parse_version_tag
    assert parse_version_tag(
        "Bastille - Good Grief (Don Diablo Remix) (Instrumental Mix) VIRGIN"
    ) == "instrumental"


def test_parse_version_tag_acappella() -> None:
    from archive.audio_pipeline.alignment.stem_mask import parse_version_tag
    assert parse_version_tag("Track Name (Acapella)")   == "acappella"
    assert parse_version_tag("Track Name (Acappella)")  == "acappella"
    assert parse_version_tag("Track Name - Vocal Mix")  == "acappella"


def test_parse_version_tag_dub_and_karaoke() -> None:
    from archive.audio_pipeline.alignment.stem_mask import parse_version_tag
    assert parse_version_tag("Track (Dub Mix)") == "instrumental"
    assert parse_version_tag("Track (Karaoke Version)") == "instrumental"


def test_parse_version_tag_none_for_regular_tracks() -> None:
    from archive.audio_pipeline.alignment.stem_mask import parse_version_tag
    assert parse_version_tag("Artist - Title (Extended Mix)") is None
    assert parse_version_tag("Artist - Title") is None
    assert parse_version_tag("") is None
    assert parse_version_tag(None) is None
