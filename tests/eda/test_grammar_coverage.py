"""Unit tests for the grammar-coverage fingerprint pure logic."""

from __future__ import annotations

from eda.alignment.generalization.grammar_coverage import (
    SetFingerprint,
    build_coverage,
    frac_bin,
    frac_bin3,
    parse_play_time,
    quantile_edges,
)


def _fp(sid, *, w=0.0, ver=0.0, stem=0.0, tracks=20, pt=60.0, audio=False, fetch=True):
    return SetFingerprint(
        set_id=sid,
        title=sid,
        n_slots=tracks,
        total_tracks=tracks,
        play_time_min=pt,
        w_frac=w,
        version_frac=ver,
        stem_frac=stem,
        id_frac=0.0,
        mean_cue_gap=None,
        styles="",
        has_audio=audio,
        fetchable=fetch and not audio,
    )


def test_parse_play_time():
    assert parse_play_time("1h 28m") == 88.0
    assert parse_play_time("59m") == 59.0
    assert parse_play_time("1h") == 60.0
    assert parse_play_time("2h 0m") == 120.0
    assert parse_play_time("") is None
    assert parse_play_time("garbage") is None


def test_frac_bins():
    assert frac_bin(0.0) == "none"
    assert frac_bin(0.05) == "low"
    assert frac_bin(0.2) == "mid"
    assert frac_bin(0.9) == "high"
    assert frac_bin3(0.0) == "none"
    assert frac_bin3(0.1) == "low"
    assert frac_bin3(0.5) == "high"


def test_quantile_edges():
    assert quantile_edges([]) == []
    edges = quantile_edges([1, 2, 3, 4, 5, 6, 7, 8], n=4)
    assert len(edges) == 3
    assert edges == sorted(edges)


def test_density():
    assert _fp("a", tracks=30, pt=60.0).density == 0.5
    assert _fp("b", tracks=30, pt=None).density is None


def test_fill_weight_prefers_starved_corner():
    # Corpus: 3 mashup-heavy sets (all downloaded) + 3 straight sets (none downloaded).
    rows = [
        _fp("m1", w=0.8, ver=0.5, stem=0.4, audio=True),
        _fp("m2", w=0.7, ver=0.5, stem=0.4, audio=True),
        _fp("m3", w=0.9, ver=0.5, stem=0.4, audio=True),
        _fp("s1", w=0.0, ver=0.0, stem=0.0, audio=False),
        _fp("s2", w=0.0, ver=0.0, stem=0.0, audio=False),
        _fp("s3", w=0.0, ver=0.0, stem=0.0, audio=False),
    ]
    cov = build_coverage(rows)
    # A straight set (starved corner: 0% downloaded) must outrank a mashup clone
    # (saturated corner: 100% downloaded).
    straight = _fp("cand_straight", w=0.0, ver=0.0, stem=0.0)
    mashup = _fp("cand_mashup", w=0.8, ver=0.5, stem=0.4)
    assert cov.fill_weight(straight) > cov.fill_weight(mashup)
    # And the saturated corner reports high bin_coverage, starved reports ~0.
    assert cov.bin_coverage(mashup) > cov.bin_coverage(straight)


def test_coverage_shares_are_fractions():
    rows = [_fp(f"x{i}", w=0.0, audio=(i % 2 == 0)) for i in range(10)]
    cov = build_coverage(rows)
    for axis, bins in cov.per_axis.items():
        for b, c in bins.items():
            assert 0.0 <= c <= 1.0, (axis, b, c)
