"""Tier-1 verification: EXHAUSTIVE proofs over the finite identity domains.

These are not samples. The identity vocabulary is finite and tiny
(version x stem x variant = 7 x 3 x 2 = 42), and the satisfaction tier lattice
is finite too, so we enumerate the *entire* input space and assert the invariant
for every element — a decision-procedure-level proof, not a property-test guess.

Vocabularies are read from ``typing.get_args`` of the ``Literal`` types so the
proof automatically tracks the source of truth; ``test_axis_space_size`` is the
tripwire that fires (forcing a conscious doc + test review) if the space changes.
"""

from __future__ import annotations

import itertools
from typing import get_args

from core.audio_resolve import (
    TIER_EXACT,
    TIER_REGULAR,
    TIER_STEM_FALLBACK,
    TIER_UNRELATED,
    TIER_VARIANT_FALLBACK,
    _match_tier,
)
from core.identity import (
    _STEM_ALIASES,
    _VERSION_FROM_SCRAPE,
    RecordingAxes,
    Stem,
    Variant,
    Version,
    normalize_stem,
    normalize_variant,
    normalize_version,
    parse_axes_key,
    scrape_version_to_db,
)
from core.slot_inventory import LayerRole, derive_layer_role

VERSIONS = get_args(Version)
STEMS = get_args(Stem)
VARIANTS = get_args(Variant)
LAYER_ROLES = set(get_args(LayerRole))
ALL_TIERS = {
    TIER_EXACT,
    TIER_VARIANT_FALLBACK,
    TIER_STEM_FALLBACK,
    TIER_REGULAR,
    TIER_UNRELATED,
}

ALL_AXES = [
    RecordingAxes(version=v, stem=s, variant=va)
    for v, s, va in itertools.product(VERSIONS, STEMS, VARIANTS)
]


# ── identity vocabulary ───────────────────────────────────────────────────────


def test_axis_space_size():
    # Tripwire: the documented 7x3x2=42 space. If a Literal grows, update the
    # docs (CLAUDE.md, identity.py) and the migration story, then this assert.
    assert (len(VERSIONS), len(STEMS), len(VARIANTS)) == (7, 3, 2)
    assert len(ALL_AXES) == 42


def test_key_roundtrip_is_identity_over_whole_space():
    # parse ∘ print = id, for every one of the 42 axes (version_artist is not
    # part of the key, so it round-trips as None).
    for axes in ALL_AXES:
        assert parse_axes_key(axes.key()) == axes


def test_key_is_injective_over_whole_space():
    keys = [axes.key() for axes in ALL_AXES]
    assert len(set(keys)) == len(ALL_AXES)  # no collisions: the key is faithful


def test_key_shape_over_whole_space():
    for axes in ALL_AXES:
        parts = axes.key().split("__")
        assert len(parts) == 3
        assert parts[0] in VERSIONS and parts[1] in STEMS and parts[2] in VARIANTS
        assert axes.key() == axes.key().lower()


def test_display_suffix_total_over_whole_space():
    # Never raises, always a str — incl. with a version_artist present.
    for axes in ALL_AXES:
        assert isinstance(axes.display_suffix(), str)
        assert isinstance(
            RecordingAxes(
                axes.version, axes.stem, axes.variant, "Some Remixer"
            ).display_suffix(),
            str,
        )


# ── normalizers: canonical closure + idempotence ──────────────────────────────


def test_normalizers_fix_canonical_values():
    for v in VERSIONS:
        assert normalize_version(v) == v
    for s in STEMS:
        assert normalize_stem(s) == s
    for va in VARIANTS:
        assert normalize_variant(va) == va


def test_normalizers_idempotent_and_in_range_over_canonical():
    for v in VERSIONS:
        assert normalize_version(normalize_version(v)) == normalize_version(v)
    for s in STEMS:
        assert normalize_stem(s) in STEMS
    for va in VARIANTS:
        assert normalize_variant(va) in VARIANTS


def test_stem_aliases_close_into_vocabulary():
    # Every known alias resolves to a canonical stem (full/original -> regular).
    for raw, expected in _STEM_ALIASES.items():
        assert normalize_stem(raw) == expected
        assert normalize_stem(raw) in STEMS


# ── scrape → DB version map: total, in-range, axis-orthogonal ──────────────────


def test_scrape_version_map_total_and_in_range():
    for tag in _VERSION_FROM_SCRAPE:
        assert scrape_version_to_db(tag) in VERSIONS


def test_vocal_tags_never_become_a_version():
    # The orthogonality guarantee: a mis-tagged "Acappella" scrape is a *stem*
    # concern, never a version. It must collapse to original, and the vocal
    # words must not be members of the version vocabulary at all.
    assert scrape_version_to_db("Acappella") == "original"
    assert "acappella" not in VERSIONS and "instrumental" not in VERSIONS
    assert normalize_stem("acappella") == "acappella"
    assert normalize_stem("instrumental") == "instrumental"


# ── satisfaction tier lattice: exhaustive over all 36 (claim x candidate) ──────

_CLAIM_CAND = list(itertools.product(STEMS, VARIANTS, STEMS, VARIANTS))


def test_match_tier_total_and_in_range():
    for cs, cv, ws, wv in _CLAIM_CAND:
        assert _match_tier(cs, cv, ws, wv) in ALL_TIERS


def test_match_tier_exact_iff_both_axes_match():
    # TIER_EXACT (0) is reachable only by an exact stem+variant match — the
    # minimum of the order is the unique exact case.
    for cs, cv, ws, wv in _CLAIM_CAND:
        tier = _match_tier(cs, cv, ws, wv)
        assert (tier == TIER_EXACT) == (cs == ws and cv == wv)


def test_match_tier_is_deterministic():
    for cs, cv, ws, wv in _CLAIM_CAND:
        assert _match_tier(cs, cv, ws, wv) == _match_tier(cs, cv, ws, wv)


def test_match_tier_ordering_is_total():
    # The five tiers are a strict total order 0<1<2<3<4 (ints, all distinct).
    tiers = sorted(ALL_TIERS)
    assert tiers == [0, 1, 2, 3, 4]


# ── layer-role derivation: finite axes x slot-label equivalence classes ────────

# slot_label is an unbounded string, so we cover its equivalence classes
# (primary / first w-slot / later w-slot / non-matching) crossed exhaustively
# with the finite stem and concurrency axes.
_SLOT_LABELS = ["013", "081", "013w1", "013w2", "013w3", "", "garbage"]
_STEM_INPUTS = list(STEMS) + [None]


def test_layer_role_total_and_deterministic():
    for label in _SLOT_LABELS:
        for stem in _STEM_INPUTS:
            for conc in (True, False):
                role = derive_layer_role(label, is_concurrent=conc, claimed_stem=stem)
                assert role in LAYER_ROLES
                # idempotent under re-derivation (pure fn — re-materialization safe)
                assert role == derive_layer_role(
                    label, is_concurrent=conc, claimed_stem=stem
                )


def test_layer_role_primary_slot_rules():
    for stem in _STEM_INPUTS:
        assert derive_layer_role("081", is_concurrent=True, claimed_stem=stem) == "bed"
        assert (
            derive_layer_role("081", is_concurrent=False, claimed_stem=stem) == "solo"
        )


def test_layer_role_payload_rules():
    # acappella w-slot -> payload regardless of index; first w-slot regular -> payload.
    assert derive_layer_role("013w2", claimed_stem="acappella") == "payload"
    assert derive_layer_role("013w1", claimed_stem="regular") == "payload"
    assert derive_layer_role("013w2", claimed_stem="regular") == "constituent"
