"""Tests for mashup/bootleg version parsing."""

from __future__ import annotations

from tokenizer.identity_axes import derive_version_flags


def test_mashup_in_row_text():
    _, tag = derive_version_flags(
        "Quintino vs DJ Kool - Let Me Clear My Scorpion (Henry Fong Mashup)"
    )
    assert tag == "Mashup"


def test_bootleg():
    _, tag = derive_version_flags("Artist - Title (Someone Bootleg)")
    assert tag == "Bootleg"
