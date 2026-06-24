"""Tiered stem acquisition cascade driver (Phase 2 alignment plan)."""

from __future__ import annotations

from dataclasses import dataclass

from core.slot_inventory import InventoryAction, SlotSatisfaction, suggest_actions
from ingest.solvability import SolvabilityTier, classify_metadata


@dataclass(frozen=True)
class CascadePlan:
    actions: tuple[InventoryAction, ...]
    tier: SolvabilityTier
    note: str


def plan_for_slot(
    satisfaction: SlotSatisfaction,
    *,
    full_name: str | None = None,
    version: str | None = None,
) -> CascadePlan:
    """Propose acquisition steps ordered by cascade priority."""
    meta_tier = classify_metadata(
        full_name=full_name or satisfaction.claim.display_name,
        version=version,
        claimed_stem=satisfaction.claim.claimed_stem,
    )
    actions = suggest_actions(satisfaction)
    if satisfaction.status.value == "satisfied":
        return CascadePlan((), meta_tier, "already satisfied")
    if meta_tier.tier >= SolvabilityTier.SEPARATION_ONLY:
        return CascadePlan(actions, meta_tier, "fall back to separation floor")
    return CascadePlan(actions, meta_tier, "try community/official before separation")
