"""Tests for tokenizer/identity_axes.py — version vs stem split."""
from __future__ import annotations

from tokenizer.identity_axes import derive_claimed_stem, derive_version_flags


def test_acappella_is_stem_not_version() -> None:
    name = "Artist - Title (Syn Cole Remix) (Acappella)"
    assert derive_claimed_stem(name) == "acappella"  # stem axis
    is_remix, version = derive_version_flags(name + " remix", remix_flag=False)
    assert version == "Remix"
    assert is_remix is True


def test_instrumental_is_stem() -> None:
    name = "Martin Garrix - There For You (Madison Mars Remix) (Instrumental)"
    assert derive_claimed_stem(name) == "instrumental"
    _, version = derive_version_flags(name)
    assert version == "Remix"


def test_plain_track_regular_stem() -> None:
    assert derive_claimed_stem("Daft Punk - Around the World") == "regular"
    assert derive_version_flags("Daft Punk - Around the World")[1] is None
