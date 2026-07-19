"""HuBERT↔lyrics corroboration for G2 clustering (E1 pseudo mass)."""

from __future__ import annotations

from workspaces.alignment_prototype.agentic.live_runners import (
    LYRICS_HUBERT_CORROBORATE_S,
    corroborate_hubert_with_lyrics,
)


def test_corroborate_snaps_near_lyrics_peak():
    ss, rs, tag = corroborate_hubert_with_lyrics(
        120.0,
        40.0,
        prior_src="lyrics",
        lyrics_ss=100.0,
        tol_s=30.0,
    )
    assert ss == 100.0
    assert rs == 40.0
    assert "corroborate_lyrics" in tag


def test_corroborate_noop_when_far_or_wrong_prior():
    ss, rs, tag = corroborate_hubert_with_lyrics(
        200.0,
        40.0,
        prior_src="lyrics",
        lyrics_ss=100.0,
        tol_s=30.0,
    )
    assert (ss, rs, tag) == (200.0, 40.0, "")

    ss2, rs2, tag2 = corroborate_hubert_with_lyrics(
        105.0,
        40.0,
        prior_src="timeline",
        lyrics_ss=100.0,
    )
    assert (ss2, rs2, tag2) == (105.0, 40.0, "")


def test_default_tol_matches_constant():
    assert LYRICS_HUBERT_CORROBORATE_S == 30.0
    ss, _, tag = corroborate_hubert_with_lyrics(
        129.0, 10.0, prior_src="lyrics", lyrics_ss=100.0
    )
    assert ss == 100.0 and tag
    ss2, _, tag2 = corroborate_hubert_with_lyrics(
        131.0, 10.0, prior_src="lyrics", lyrics_ss=100.0
    )
    assert ss2 == 131.0 and not tag2
