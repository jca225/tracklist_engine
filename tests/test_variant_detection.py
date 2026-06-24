"""Convention-gap test: extended-mix detection on the variant axis.

derive_claimed_variant was the one axis with no test (version + stem had them).
The rule: an "extended <mix|version|edit>" phrase => variant 'extended', else the
default 'regular'. The bare word "extended" is NOT enough — it needs the noun, so
"Extended Play"/"extended outro" don't false-positive into the variant axis.
"""

from __future__ import annotations

from tokenizer.identity_axes import derive_claimed_variant


def test_extended_mix_is_extended():
    assert derive_claimed_variant("Avicii - Levels (Extended Mix)") == "extended"


def test_extended_version_and_edit():
    assert derive_claimed_variant("Artist - Title (Extended Version)") == "extended"
    assert derive_claimed_variant("Artist - Title (Extended Edit)") == "extended"


def test_case_insensitive():
    assert derive_claimed_variant("artist - title (extended mix)") == "extended"


def test_detected_in_row_text_too():
    assert (
        derive_claimed_variant("Artist - Title", "played the extended mix")
        == "extended"
    )


def test_plain_track_is_regular():
    assert derive_claimed_variant("Daft Punk - Around The World") == "regular"


def test_bare_extended_word_does_not_trigger():
    # "extended" without mix/version/edit must not flip the variant axis.
    assert derive_claimed_variant("Artist - Extended Play") == "regular"
    assert (
        derive_claimed_variant("Artist - Title", "an extended outro here") == "regular"
    )


def test_none_inputs_default_regular():
    assert derive_claimed_variant(None) == "regular"
    assert derive_claimed_variant(None, None) == "regular"
