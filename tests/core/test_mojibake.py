"""Unit tests for the mojibake detector/repair primitive (core/mojibake.py).

Anchored on the issue-#74 forensic case: the en-dash `U+2013` (UTF-8
`e2 80 93`) mis-decoded as latin-1 becomes `U+00E2 U+0080 U+0093` (``â€"``).
"""

from __future__ import annotations

import pytest

from core.mojibake import is_double_encoded_utf8, repair_double_encoded_utf8

# The issue-#74 forensic pair: correct title fragment vs. its stored mojibake.
CORRECT = "Sign Of The Times – Studio Acapella"  # real en-dash U+2013
MOJIBAKE = CORRECT.encode("utf-8").decode("latin-1")  # â€" — how pi stored it


def test_forensic_case_detected_and_repaired() -> None:
    assert MOJIBAKE != CORRECT
    assert is_double_encoded_utf8(MOJIBAKE) is True
    assert repair_double_encoded_utf8(MOJIBAKE) == CORRECT


@pytest.mark.parametrize(
    "s",
    [
        "",
        "/mnt/storage/objects/abc/abc__youtube__vid.m4a",  # pure ASCII path
        "plain title.wav",
    ],
)
def test_ascii_and_empty_never_flagged(s: str) -> None:
    assert is_double_encoded_utf8(s) is False
    assert repair_double_encoded_utf8(s) == s


def test_genuine_single_accent_not_flagged() -> None:
    # A correctly-stored accented name: 'é' == U+00E9, which is a lone 0xE9 in
    # latin-1 and NOT valid UTF-8 -> must not be mistaken for mojibake.
    s = "café del mar.m4a"
    assert is_double_encoded_utf8(s) is False
    assert repair_double_encoded_utf8(s) == s


def test_correct_utf8_char_above_latin1_not_flagged() -> None:
    # A path already holding the real en-dash (post-repair) is never re-flagged,
    # because encode('latin-1') raises on any code point > U+00FF.
    assert is_double_encoded_utf8(CORRECT) is False
    assert repair_double_encoded_utf8(CORRECT) == CORRECT


def test_repair_is_idempotent() -> None:
    once = repair_double_encoded_utf8(MOJIBAKE)
    assert repair_double_encoded_utf8(once) == once == CORRECT


def test_repair_noop_on_clean_returns_same_object() -> None:
    assert repair_double_encoded_utf8(CORRECT) is CORRECT
