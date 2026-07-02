"""Filename parsing for the Discord stem-library matcher.

The staging corpus (`/mnt/storage/staging/discord_stems`) carries filenames whose
percent-encoding was sanitized to underscores (`%20` -> `_20`). The decoder must
reverse real escapes without touching look-alikes (`_feat.`, `_ACAPELLA`, `_12`).
"""

from __future__ import annotations

from pathlib import Path

from scripts.match_stem_library import decode_escaped, parse_filename


class TestDecodeEscaped:
    def test_space(self) -> None:
        assert decode_escaped("Don_20Diablo") == "Don Diablo"

    def test_ampersand_run(self) -> None:
        assert decode_escaped("Martin_20Garrix_20_26_20MOti") == "Martin Garrix & MOti"

    def test_parens(self) -> None:
        assert decode_escaped("Virus_20_28Instrumental_29") == "Virus (Instrumental)"

    def test_utf8_multibyte(self) -> None:
        assert decode_escaped("Isab_C3_A8l_20Usher") == "Isabèl Usher"

    def test_lowercase_hex_word_untouched(self) -> None:
        # "fe" is valid hex but percent-encoding emits uppercase — "_feat." is text
        assert decode_escaped("Someday_20_feat._20Usher") == "Someday _feat. Usher"

    def test_mixed_case_word_untouched(self) -> None:
        assert decode_escaped("X_Acapella") == "X_Acapella"

    def test_lone_high_byte_untouched(self) -> None:
        # 0xAC alone is invalid UTF-8 — "_ACAPELLA" is text, not an escape
        assert decode_escaped("X_ACAPELLA") == "X_ACAPELLA"

    def test_control_char_untouched(self) -> None:
        # 0x12 is a control char — "_12" is a track/volume number
        assert decode_escaped("Vol_12") == "Vol_12"

    def test_alnum_decode_rejected(self) -> None:
        # 0x72 = "r": alnum is never percent-encoded — "_7258" is an id
        assert decode_escaped("AUDIO_7258_20x") == "AUDIO_7258 x"

    def test_camelot_key_untouched(self) -> None:
        # no "_20" anywhere -> name is not percent-encoded; "_2A" is a Camelot key
        assert (
            decode_escaped("Heroes_Instrumental_128_2A") == "Heroes_Instrumental_128_2A"
        )

    def test_dj_handle_untouched(self) -> None:
        assert decode_escaped("DIY_Acapella_by_3Dee") == "DIY_Acapella_by_3Dee"

    def test_plain_name_passthrough(self) -> None:
        assert (
            decode_escaped("Cartinez-Luv-Me-Extended-Instrumental")
            == "Cartinez-Luv-Me-Extended-Instrumental"
        )


class TestParseFilenameEscaped:
    def test_escaped_artist_title_axes(self) -> None:
        p = parse_filename(
            Path("Martin_20Garrix_20_26_20MOti_20-_20Virus_20_28Instrumental_29.wav")
        )
        assert p.artist == "Martin Garrix & MOti"
        assert p.title == "Virus"
        assert p.stem == "instrumental"

    def test_escaped_extended_variant(self) -> None:
        p = parse_filename(
            Path("Cartinez_20-_20Luv_20Me_20_28Extended_20Instrumental_29.mp3")
        )
        assert p.artist == "Cartinez"
        assert p.title == "Luv Me"
        assert p.stem == "instrumental"
        assert p.variant == "extended"
