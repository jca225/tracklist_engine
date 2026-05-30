"""Tests for core/identity.py — three-axis keys."""
from __future__ import annotations

from core.identity import RecordingAxes, parse_axes_key


def test_concatenated_key() -> None:
    axes = RecordingAxes(version="remix", stem="acappella", variant="extended")
    assert axes.key() == "remix__acappella__extended"


def test_parse_roundtrip() -> None:
    key = "original__regular__regular"
    assert parse_axes_key(key).key() == key


def test_legacy_stem_alias() -> None:
    assert parse_axes_key("remix__full__regular").stem == "regular"
