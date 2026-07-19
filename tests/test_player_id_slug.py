"""Tests for `_ascii_player_id` — the sanitizer that keeps raw unicode out of
canonical audio paths (root cause of the track_audio.path mojibake, bug A1).

A player_id that carries a non-ASCII byte survives into
`<tid>__<platform>__<player_id>.<ext>`; that path string then mojibake-corrupts
crossing the Mac(UTF-8)->pi(Latin-1) SSH/sqlite boundary. The fix is to force
player_id to pure ASCII at the path chokepoint.
"""

from __future__ import annotations

from scripts.replace_track_audio import _ascii_player_id as slug


def test_ascii_passthrough_is_identity():
    # Real platform ids (youtube/soundcloud/spotify) are already ASCII and must
    # not be altered.
    for pid in ["dQw4w9WgXcQ", "a_b-c", "1234567890", "track.v2", "AbC_09-xy"]:
        assert slug(pid) == pid


def test_idempotent():
    once = slug("Some (Official Karaoke Instrumental) ｜ SongJam")
    assert slug(once) == once


def test_fullwidth_pipe_stripped():
    # U+FF5C (the exact byte that caused the reported mojibake) must not survive.
    out = slug("Song ｜ SongJam")
    assert "｜" not in out
    assert out.isascii()


def test_accented_latin_transliterated_not_dropped():
    # RÜFÜS-style names: keep the letters, transliterated, don't nuke to empty.
    out = slug("RÜFÜS")
    assert out.isascii()
    assert out.lower() == "rufus"


def test_output_always_ascii():
    for pid in ["ｎｉｃｅ", "café", "naïve", "Ω_test", "emoji😀name"]:
        assert slug(pid).isascii()


def test_all_nonascii_falls_back_deterministically():
    # A name with no ASCII survivors still yields a valid, stable, non-empty id.
    a = slug("日本語のタイトル")
    b = slug("日本語のタイトル")
    assert a == b
    assert a.isascii()
    assert a  # non-empty
    # different input -> different fallback
    assert slug("日本語のタイトル") != slug("другое имя")


def test_no_path_separators_or_spaces_survive():
    out = slug("a/b c\\d:e")
    for bad in ["/", "\\", " ", ":"]:
        assert bad not in out


def test_empty_input_returns_empty():
    assert slug("") == ""


def test_length_bounded():
    out = slug("x" * 500)
    assert len(out) <= 96
