from __future__ import annotations

"""Tests for stem_routing.py (Workstream F).

TDD: written BEFORE the implementation, verified RED, then GREEN.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from workspaces.pws_aligner.stem_routing import (
    LF_COMPETENCE,
    Stem,
    RoutedVote,
    competent_stems,
    route_audio_lf,
)
from workspaces.pws_aligner.operations import OperationVote
from workspaces.pws_aligner.votes import AbstainReason


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_vote(span_id: str = "span0") -> OperationVote:
    """Return a minimal non-abstaining OperationVote for use as a stub."""
    return OperationVote(
        lf="stub",
        span_id=span_id,
        label="test_label",
        confidence=0.9,
        abstained=False,
        reason=AbstainReason.NONE,
    )


def _make_audio(sr: int = 22050, duration: float = 0.1) -> np.ndarray:
    """Tiny silent audio array for routing tests (no DSP needed)."""
    return np.zeros(int(sr * duration), dtype=np.float32)


# ---------------------------------------------------------------------------
# Stem enum
# ---------------------------------------------------------------------------


class TestStemEnum:
    def test_three_variants(self) -> None:
        assert Stem.FULL.value == "full"
        assert Stem.VOCAL.value == "vocal"
        assert Stem.INSTRUMENTAL.value == "instrumental"

    def test_distinct(self) -> None:
        assert Stem.FULL != Stem.VOCAL != Stem.INSTRUMENTAL


# ---------------------------------------------------------------------------
# RoutedVote dataclass
# ---------------------------------------------------------------------------


class TestRoutedVote:
    def test_fields_present(self) -> None:
        v = _stub_vote()
        rv = RoutedVote(stem=Stem.FULL, vote=v)
        assert rv.stem == Stem.FULL
        assert rv.vote is v

    def test_frozen(self) -> None:
        rv = RoutedVote(stem=Stem.VOCAL, vote=_stub_vote())
        with pytest.raises((AttributeError, TypeError)):
            rv.stem = Stem.INSTRUMENTAL  # type: ignore[misc]

    def test_equality(self) -> None:
        v = _stub_vote()
        rv1 = RoutedVote(stem=Stem.FULL, vote=v)
        rv2 = RoutedVote(stem=Stem.FULL, vote=v)
        assert rv1 == rv2

    def test_inequality_different_stem(self) -> None:
        v = _stub_vote()
        rv1 = RoutedVote(stem=Stem.FULL, vote=v)
        rv2 = RoutedVote(stem=Stem.VOCAL, vote=v)
        assert rv1 != rv2


# ---------------------------------------------------------------------------
# LF_COMPETENCE table
# ---------------------------------------------------------------------------


class TestLFCompetenceTable:
    def test_known_lfs_present(self) -> None:
        expected = {
            "echo_tail",
            "noise_sweep",
            "filter_sweep",
            "sidechain",
            "loop",
            "keylock_vs_varispeed",
        }
        assert expected.issubset(LF_COMPETENCE.keys())

    def test_echo_tail_competence(self) -> None:
        stems = LF_COMPETENCE["echo_tail"]
        assert Stem.FULL in stems
        assert Stem.VOCAL in stems

    def test_noise_sweep_competence(self) -> None:
        stems = LF_COMPETENCE["noise_sweep"]
        assert Stem.FULL in stems
        assert Stem.VOCAL not in stems
        assert Stem.INSTRUMENTAL not in stems

    def test_filter_sweep_competence(self) -> None:
        stems = LF_COMPETENCE["filter_sweep"]
        assert Stem.FULL in stems
        assert Stem.INSTRUMENTAL in stems

    def test_sidechain_competence(self) -> None:
        stems = LF_COMPETENCE["sidechain"]
        assert Stem.INSTRUMENTAL in stems
        assert Stem.FULL in stems

    def test_loop_competence(self) -> None:
        stems = LF_COMPETENCE["loop"]
        assert Stem.FULL in stems
        assert Stem.VOCAL in stems
        assert Stem.INSTRUMENTAL in stems

    def test_keylock_vs_varispeed_competence(self) -> None:
        stems = LF_COMPETENCE["keylock_vs_varispeed"]
        assert Stem.INSTRUMENTAL in stems
        assert Stem.VOCAL not in stems


# ---------------------------------------------------------------------------
# competent_stems()
# ---------------------------------------------------------------------------


class TestCompetentStems:
    def test_known_lf_returns_table_value(self) -> None:
        result = competent_stems("echo_tail")
        assert result == LF_COMPETENCE["echo_tail"]

    def test_unknown_lf_defaults_to_full(self) -> None:
        result = competent_stems("completely_unknown_lf_xyz")
        assert result == (Stem.FULL,)

    def test_stem_agnostic_lfs_default_to_full(self) -> None:
        # cue_time and claimed_axes_identity are not in the table;
        # they are stem-agnostic and should get the (FULL,) default.
        assert competent_stems("cue_time") == (Stem.FULL,)
        assert competent_stems("claimed_axes_identity") == (Stem.FULL,)

    def test_returns_tuple(self) -> None:
        result = competent_stems("loop")
        assert isinstance(result, tuple)


# ---------------------------------------------------------------------------
# route_audio_lf()
# ---------------------------------------------------------------------------


class TestRouteAudioLF:
    """Tests for the dispatcher that calls call_fn once per competent stem."""

    def _tracking_call_fn(self, calls: list[Stem]):
        """Return a call_fn that records which stems it was called with."""

        def call_fn(stem: Stem, y: np.ndarray, sr: int) -> OperationVote:
            calls.append(stem)
            return _stub_vote()

        return call_fn

    def test_calls_fn_for_each_competent_stem_present(self) -> None:
        """echo_tail is competent on FULL + VOCAL; both present → 2 calls."""
        sr = 22050
        y = _make_audio(sr)
        stem_audios = {
            Stem.FULL: (y, sr),
            Stem.VOCAL: (y, sr),
            Stem.INSTRUMENTAL: (y, sr),
        }
        calls: list[Stem] = []
        results = route_audio_lf(
            "echo_tail", "span0", stem_audios, self._tracking_call_fn(calls)
        )
        assert Stem.FULL in calls
        assert Stem.VOCAL in calls
        # INSTRUMENTAL not in echo_tail competence — should NOT be called
        assert Stem.INSTRUMENTAL not in calls
        assert len(results) == 2

    def test_skips_absent_stems(self) -> None:
        """echo_tail competent on FULL+VOCAL, but only FULL present → 1 call."""
        sr = 22050
        y = _make_audio(sr)
        stem_audios = {
            Stem.FULL: (y, sr),
            # VOCAL absent
        }
        calls: list[Stem] = []
        results = route_audio_lf(
            "echo_tail", "span0", stem_audios, self._tracking_call_fn(calls)
        )
        assert Stem.FULL in calls
        assert Stem.VOCAL not in calls
        assert len(results) == 1

    def test_routed_votes_carry_correct_stem_tags(self) -> None:
        """Each RoutedVote is tagged with the stem it was run on."""
        sr = 22050
        y = _make_audio(sr)
        stem_audios = {
            Stem.FULL: (y, sr),
            Stem.VOCAL: (y, sr),
        }
        results = route_audio_lf(
            "echo_tail", "span0", stem_audios, lambda stem, y, sr: _stub_vote()
        )
        result_stems = {rv.stem for rv in results}
        assert result_stems == {Stem.FULL, Stem.VOCAL}

    def test_routed_vote_carries_underlying_vote(self) -> None:
        """RoutedVote.vote is the vote returned by call_fn."""
        sr = 22050
        y = _make_audio(sr)
        expected_vote = _stub_vote("span_abc")
        stem_audios = {Stem.FULL: (y, sr)}
        results = route_audio_lf(
            "noise_sweep", "span_abc", stem_audios, lambda stem, y, sr: expected_vote
        )
        assert len(results) == 1
        assert results[0].vote is expected_vote

    def test_empty_stem_audios_returns_empty(self) -> None:
        """No stems present → no calls, empty result."""
        calls: list[Stem] = []
        results = route_audio_lf(
            "echo_tail", "span0", {}, self._tracking_call_fn(calls)
        )
        assert results == []
        assert calls == []

    def test_unknown_lf_uses_full_default(self) -> None:
        """Unknown LF defaults to competence on FULL only."""
        sr = 22050
        y = _make_audio(sr)
        stem_audios = {
            Stem.FULL: (y, sr),
            Stem.VOCAL: (y, sr),
        }
        calls: list[Stem] = []
        results = route_audio_lf(
            "unknown_lf_xyz", "span0", stem_audios, self._tracking_call_fn(calls)
        )
        # Unknown → default (FULL,) → only FULL called
        assert calls == [Stem.FULL]
        assert len(results) == 1
        assert results[0].stem == Stem.FULL

    def test_loop_runs_on_all_three_stems(self) -> None:
        """loop is competent on all three stems → 3 calls when all present."""
        sr = 22050
        y = _make_audio(sr)
        stem_audios = {
            Stem.FULL: (y, sr),
            Stem.VOCAL: (y, sr),
            Stem.INSTRUMENTAL: (y, sr),
        }
        calls: list[Stem] = []
        results = route_audio_lf(
            "loop", "span0", stem_audios, self._tracking_call_fn(calls)
        )
        assert set(calls) == {Stem.FULL, Stem.VOCAL, Stem.INSTRUMENTAL}
        assert len(results) == 3

    def test_keylock_vs_varispeed_only_instrumental(self) -> None:
        """keylock_vs_varispeed competent on INSTRUMENTAL only."""
        sr = 22050
        y = _make_audio(sr)
        stem_audios = {
            Stem.FULL: (y, sr),
            Stem.VOCAL: (y, sr),
            Stem.INSTRUMENTAL: (y, sr),
        }
        calls: list[Stem] = []
        results = route_audio_lf(
            "keylock_vs_varispeed", "span0", stem_audios, self._tracking_call_fn(calls)
        )
        assert calls == [Stem.INSTRUMENTAL]
        assert len(results) == 1
        assert results[0].stem == Stem.INSTRUMENTAL
