"""Tests for tokenizer/identity_axes.py — version vs stem split."""
from __future__ import annotations

import importlib.util
from pathlib import Path

# Load identity_axes directly — tokenizer/__init__.py pulls bs4/pydantic we
# don't need for these unit tests (and aren't installed in CI guardrails job).
_spec = importlib.util.spec_from_file_location(
    "tokenizer.identity_axes",
    Path(__file__).resolve().parents[1] / "tokenizer" / "identity_axes.py",
)
assert _spec and _spec.loader
_identity_axes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_identity_axes)
derive_claimed_stem = _identity_axes.derive_claimed_stem
derive_version_flags = _identity_axes.derive_version_flags


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
