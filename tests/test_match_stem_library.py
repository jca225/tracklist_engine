"""Filename parsing for the Discord stem-library matcher.

The staging corpus (`/mnt/storage/staging/discord_stems`) carries filenames whose
percent-encoding was sanitized to underscores (`%20` -> `_20`). The decoder must
reverse real escapes without touching look-alikes (`_feat.`, `_ACAPELLA`, `_12`).
"""

from __future__ import annotations

from pathlib import Path

from scripts.match_stem_library import (
    Recording,
    candidates,
    decide,
    decode_escaped,
    parse_filename,
)


def _rec(rid: str, work: str, title: str, version: str = "original") -> Recording:
    return Recording(
        recording_id=rid,
        work_id=work,
        version=version,
        version_artist="",
        stem="regular",
        variant="regular",
        artist="Technotronic",
        title=title,
    )


class TestCrossWorkMargin:
    def test_sibling_recordings_do_not_depress_margin(self) -> None:
        # original + remix of the same work tie at the top; the margin must be
        # measured against the best *different* work, not the sibling
        recs = [
            _rec("r1", "w1", "Pump Up The Jam"),
            _rec("r2", "w1", "Pump Up The Jam", version="remix"),
            _rec("r3", "w2", "Get Up"),
        ]
        p = parse_filename(Path("Technotronic_20-_20Pump_20Up_20The_20Jam.wav"))
        cands, other_s = candidates(p, recs, k=3)
        top_s = cands[0][1]
        assert cands[0][0].work_id == "w1"
        assert top_s - other_s > 0.2  # decisive vs w2, despite the w1 sibling tie

    def test_other_work_found_beyond_topk(self) -> None:
        # k=1 truncates the display list, but the cross-work runner-up must
        # still be found in the full scored list
        recs = [
            _rec("r1", "w1", "Pump Up The Jam"),
            _rec("r2", "w1", "Pump Up The Jam", version="remix"),
            _rec("r3", "w2", "Pump Up The Jams"),
        ]
        p = parse_filename(Path("Technotronic_20-_20Pump_20Up_20The_20Jam.wav"))
        cands, other_s = candidates(p, recs, k=1)
        assert len(cands) == 1
        assert other_s > 0.8  # w2's near-identical title contests the match


def _decide(top_s: float, margin: float, audio: float | None, **kw: object) -> str:
    defaults: dict = dict(
        stem="acappella",
        accept=0.80,
        min_margin=0.08,
        verified=True,
        hubert_floor=0.55,
        chroma_floor=0.35,
    )
    defaults.update(kw)
    return decide(top_s, margin, audio, defaults.pop("stem"), **defaults)


class TestDecide:
    def test_double_confirm_is_auto_accept(self) -> None:
        assert _decide(0.9, 0.2, 0.7) == "auto_accept"

    def test_audio_disagreement_demotes_to_review(self) -> None:
        assert _decide(0.9, 0.2, 0.4) == "review"

    def test_no_audio_signal_demotes_to_review(self) -> None:
        # a metadata accept without waveform confirmation is not auto-appliable
        assert _decide(0.9, 0.2, None) == "review"

    def test_unverified_run_keeps_plain_accept(self) -> None:
        assert _decide(0.9, 0.2, None, verified=False) == "accept"

    def test_instrumental_uses_chroma_floor(self) -> None:
        # 0.4 fails the hubert floor (0.55) but clears the chroma floor (0.35)
        assert _decide(0.9, 0.2, 0.4, stem="instrumental") == "auto_accept"

    def test_borderline_metadata_is_review(self) -> None:
        assert _decide(0.7, 0.2, 0.9) == "review"

    def test_weak_metadata_abstains_despite_audio(self) -> None:
        assert _decide(0.3, 0.2, 0.9) == "abstain"

    def test_low_margin_blocks_accept(self) -> None:
        assert _decide(0.9, 0.01, 0.9) == "review"


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
