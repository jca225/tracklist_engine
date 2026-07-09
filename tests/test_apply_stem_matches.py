"""Row selection for apply_stem_matches --auto (D.3 double-confirm routing).

Old verified CSVs carry decision='accept' with the audio_score column ignored;
new matcher CSVs carry decision='auto_accept'. --auto must apply both shapes
only when the waveform confirmed, and queue the rest for human review.
"""

from __future__ import annotations

from scripts.apply_stem_matches import audio_confirmed, row_action

FLOORS = dict(hubert_floor=0.55, chroma_floor=0.35)


def _action(row: dict[str, str], auto: bool = True) -> str:
    return row_action(row, accept="accept", auto=auto, **FLOORS)


class TestRowAction:
    def test_auto_accept_decision_applies(self) -> None:
        assert _action({"decision": "auto_accept", "stem": "acappella"}) == "apply"

    def test_legacy_accept_with_confirming_audio_applies(self) -> None:
        row = {"decision": "accept", "stem": "acappella", "audio_score": "0.71"}
        assert _action(row) == "apply"

    def test_legacy_accept_without_audio_goes_to_review(self) -> None:
        row = {"decision": "accept", "stem": "acappella", "audio_score": ""}
        assert _action(row) == "review"

    def test_legacy_accept_with_refuting_audio_goes_to_review(self) -> None:
        row = {"decision": "accept", "stem": "acappella", "audio_score": "0.31"}
        assert _action(row) == "review"

    def test_abstain_is_skipped(self) -> None:
        assert _action({"decision": "abstain", "stem": "acappella"}) == "skip"

    def test_manual_mode_ignores_audio(self) -> None:
        # a human-reviewed CSV is trusted as-is
        row = {"decision": "accept", "stem": "acappella", "audio_score": ""}
        assert _action(row, auto=False) == "apply"

    def test_manual_mode_never_applies_review_rows(self) -> None:
        assert (
            _action({"decision": "review", "stem": "acappella"}, auto=False) == "skip"
        )


class TestAudioConfirmed:
    def test_instrumental_uses_chroma_floor(self) -> None:
        row = {"stem": "instrumental", "audio_score": "0.40"}
        assert audio_confirmed(row, **FLOORS)

    def test_acappella_needs_hubert_floor(self) -> None:
        row = {"stem": "acappella", "audio_score": "0.40"}
        assert not audio_confirmed(row, **FLOORS)

    def test_garbage_score_is_unconfirmed(self) -> None:
        row = {"stem": "acappella", "audio_score": "n/a"}
        assert not audio_confirmed(row, **FLOORS)
