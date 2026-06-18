"""Per-slot audio resolution (resolve-at-read).

A DJ-set slot claims what was played on the three identity axes
(``claimed_version`` is baked into ``recording_id``; ``claimed_stem`` and
``claimed_variant`` are sibling ``track_audio`` rows under that recording).
A single recording can have several ``track_audio`` rows (regular / extended /
acappella / instrumental), and two different sets may legitimately need
different ones — so a single per-recording ``is_reference`` flag cannot answer
"which file for THIS slot".

``resolve_slot_audio`` picks the ``track_audio`` row that best matches a slot's
claimed stem+variant, with graceful fallback (and reports which tier matched so
callers can flag substitutions, e.g. claimed extended but only regular exists).

Pure over plain dict rows — no DB/SSH — so it is unit-testable in isolation.
``recording_id`` (the version axis) must already be narrowed by the caller:
candidates passed in should belong to the slot's recording (or track).
"""

from __future__ import annotations

from typing import Any, Iterable

# Cleanest-source ordering, mirrors pull_set_for_alignment's platform pref.
_PLATFORM_RANK = {
    "manual": 0,
    "youtube_music": 1,
    "spotify": 2,
    "soundcloud": 3,
    "youtube": 4,
}

# Match tiers (lower = better). Names exported for logging/telemetry.
TIER_EXACT = 0  # stem AND variant both match the claim
TIER_VARIANT_FALLBACK = 1  # stem matches, claimed variant missing -> regular
TIER_STEM_FALLBACK = 2  # variant matches, claimed stem missing -> regular full
TIER_REGULAR = 3  # plain regular/regular (baby-rule default)
TIER_UNRELATED = 4  # some other realization of the recording
_TIER_NAME = {
    0: "exact",
    1: "variant-fallback",
    2: "stem-fallback",
    3: "regular",
    4: "unrelated",
}


def tier_name(tier: int) -> str:
    return _TIER_NAME.get(tier, str(tier))


def _match_tier(
    cand_stem: str, cand_variant: str, want_stem: str, want_variant: str
) -> int:
    if cand_stem == want_stem and cand_variant == want_variant:
        return TIER_EXACT
    if cand_stem == want_stem and cand_variant == "regular":
        return TIER_VARIANT_FALLBACK
    if cand_stem == "regular" and cand_variant == want_variant:
        return TIER_STEM_FALLBACK
    if cand_stem == "regular" and cand_variant == "regular":
        return TIER_REGULAR
    return TIER_UNRELATED


def resolve_slot_audio(
    claimed_stem: str | None,
    claimed_variant: str | None,
    candidates: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any] | None, int | None]:
    """Pick the best ``track_audio`` row for a slot's claim.

    ``candidates`` are dict rows for the slot's recording/track, each with keys
    ``stem``, ``variant``, ``platform``, ``is_reference``, ``downloaded_at``
    (others ignored, passed through). Returns ``(chosen_row, tier)`` or
    ``(None, None)`` if there are no candidates.

    Ordering: match tier, then ``is_reference``, then platform cleanliness,
    then newest ``downloaded_at``.
    """
    cands = list(candidates)
    if not cands:
        return None, None
    want_stem = (claimed_stem or "regular").lower()
    want_variant = (claimed_variant or "regular").lower()

    # Pre-sort newest-first so that, among equally-ranked candidates, min()
    # (which keeps the first occurrence of the minimum) returns the newest.
    cands.sort(key=lambda c: str(c.get("downloaded_at") or ""), reverse=True)

    def rank(c: dict[str, Any]) -> tuple[int, int, int]:
        return (
            _match_tier(
                (c.get("stem") or "regular").lower(),
                (c.get("variant") or "regular").lower(),
                want_stem,
                want_variant,
            ),
            0 if c.get("is_reference") else 1,
            _PLATFORM_RANK.get((c.get("platform") or "").lower(), 5),
        )

    best = min(cands, key=rank)
    return best, rank(best)[0]
