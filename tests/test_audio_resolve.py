"""Tests for per-slot audio resolution (resolve-at-read)."""

from __future__ import annotations

from core.audio_resolve import (
    TIER_EXACT,
    TIER_STEM_FALLBACK,
    TIER_VARIANT_FALLBACK,
    resolve_slot_audio,
)


def _c(
    taid,
    stem="regular",
    variant="regular",
    platform="youtube",
    is_reference=0,
    downloaded_at="2026-01-01",
):
    return dict(
        track_audio_id=taid,
        stem=stem,
        variant=variant,
        platform=platform,
        is_reference=is_reference,
        downloaded_at=downloaded_at,
    )


def test_no_candidates():
    assert resolve_slot_audio("regular", "regular", []) == (None, None)


def test_exact_variant_wins_over_reference():
    # slot claims extended; the regular row is is_reference but the extended
    # row is the right one for this slot.
    cands = [
        _c(1, variant="regular", is_reference=1),
        _c(2, variant="extended", is_reference=0),
    ]
    chosen, tier = resolve_slot_audio("regular", "extended", cands)
    assert chosen["track_audio_id"] == 2 and tier == TIER_EXACT


def test_two_sets_one_recording_diverge():
    cands = [_c(1, variant="regular", is_reference=1), _c(2, variant="extended")]
    a, _ = resolve_slot_audio("regular", "extended", cands)
    b, _ = resolve_slot_audio("regular", "regular", cands)
    assert a["track_audio_id"] == 2 and b["track_audio_id"] == 1


def test_variant_fallback_when_extended_missing():
    # claimed extended, only regular exists -> regular, flagged as fallback.
    cands = [_c(1, variant="regular", is_reference=1)]
    chosen, tier = resolve_slot_audio("regular", "extended", cands)
    assert chosen["track_audio_id"] == 1 and tier == TIER_VARIANT_FALLBACK


def test_stem_fallback_baby_rule():
    # claimed acappella, only the regular full track exists -> regular (the
    # pull then serves vocals.flac from its Demucs stems). tier flags it.
    cands = [_c(1, stem="regular", variant="regular", is_reference=1)]
    chosen, tier = resolve_slot_audio("acappella", "regular", cands)
    assert chosen["track_audio_id"] == 1 and tier == TIER_STEM_FALLBACK


def test_prefers_explicit_acappella_over_regular():
    cands = [_c(1, stem="regular", is_reference=1), _c(2, stem="acappella")]
    chosen, tier = resolve_slot_audio("acappella", "regular", cands)
    assert chosen["track_audio_id"] == 2 and tier == TIER_EXACT


def test_platform_then_recency_tiebreak():
    # same tier+ref: manual beats youtube; among same platform, newest wins.
    cands = [
        _c(1, platform="youtube", downloaded_at="2026-06-01"),
        _c(2, platform="manual", downloaded_at="2026-01-01"),
        _c(3, platform="manual", downloaded_at="2026-06-01"),
    ]
    chosen, tier = resolve_slot_audio("regular", "regular", cands)
    assert chosen["track_audio_id"] == 3 and tier == TIER_EXACT
