"""Slot satisfaction model — claim vs inventory vs reference choice.

Pure functions over plain dicts/dataclasses. No DB/SSH imports (substrate rule).
Consumers: pull pre-check, ingest search, GT reconcile.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Literal

from core.audio_resolve import (
    TIER_EXACT,
    TIER_STEM_FALLBACK,
    TIER_UNRELATED,
    TIER_VARIANT_FALLBACK,
    resolve_slot_audio,
    tier_name,
)
from core.identity import Stem, Variant, normalize_stem, normalize_variant

LayerRole = Literal["bed", "payload", "constituent", "solo"]

_W_SLOT_RE = re.compile(r"^(\d+)w(\d+)$")
_PRIMARY_SLOT_RE = re.compile(r"^(\d+)$")


class SatisfactionStatus(str, Enum):
    SATISFIED = "satisfied"
    MISSING = "missing"
    WRONG_STEM = "wrong_stem"
    WRONG_RECORDING = "wrong_recording"
    FALLBACK = "fallback"


class ActionKind(str, Enum):
    ACQUIRE_ACAPPELLA = "acquire_acappella"
    ACQUIRE_INSTRUMENTAL = "acquire_instrumental"
    REPLACE_VERSION = "replace_version"
    PROMOTE_CANDIDATE = "promote_candidate"
    USE_DEMUCS_VOCALS = "use_demucs_vocals"
    MINT_RECORDING = "mint_recording"
    REVIEW_MANUAL = "review_manual"


@dataclass(frozen=True)
class SlotClaim:
    set_id: str
    slot_label: str
    recording_id: str
    claimed_stem: Stem = "regular"
    claimed_variant: Variant = "regular"
    layer_role: LayerRole = "solo"
    display_name: str = ""
    constituent_ids: tuple[str, ...] = ()
    bed_slot: str | None = None
    row_index: int | None = None
    is_concurrent: bool = False


@dataclass(frozen=True)
class AssetCandidate:
    track_audio_id: int
    recording_id: str
    stem: Stem
    variant: Variant
    platform: str
    path: str | None = None
    is_reference: bool = False

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> AssetCandidate:
        return cls(
            track_audio_id=int(row["track_audio_id"]),
            recording_id=str(row.get("recording_id") or row.get("track_id") or ""),
            stem=normalize_stem(row.get("stem")),
            variant=normalize_variant(row.get("variant")),
            platform=str(row.get("platform") or ""),
            path=row.get("path"),
            is_reference=bool(row.get("is_reference")),
        )


@dataclass(frozen=True)
class ReferenceChoice:
    path: str
    claimed_stem: Stem
    ref_source: str


@dataclass(frozen=True)
class SlotSatisfaction:
    claim: SlotClaim
    status: SatisfactionStatus
    chosen: AssetCandidate | None = None
    resolve_tier: int | None = None
    detail: str = ""


@dataclass(frozen=True)
class InventoryAction:
    kind: ActionKind
    set_id: str
    slot_label: str
    recording_id: str
    detail: str = ""
    track_audio_id: int | None = None


def derive_layer_role(
    slot_label: str,
    *,
    is_concurrent: bool = False,
    claimed_stem: str | None = None,
) -> LayerRole:
    """Infer mashup layer role from slot label + stem claim."""
    stem = normalize_stem(claimed_stem)
    if _PRIMARY_SLOT_RE.match(slot_label or ""):
        return "bed" if is_concurrent else "solo"
    m = _W_SLOT_RE.match(slot_label or "")
    if not m:
        return "solo"
    w_idx = int(m.group(2))
    if stem == "acappella":
        return "payload"
    if w_idx == 1 and stem == "regular":
        return "payload"
    return "constituent"


def primary_slot_for(slot_label: str) -> str | None:
    m = _W_SLOT_RE.match(slot_label or "")
    if m:
        return f"{int(m.group(1)):03d}"
    return None


def slot_claim_from_row(row: dict[str, Any]) -> SlotClaim:
    """Build SlotClaim from a set_track_slots SQL row dict."""
    slot_label = str(row.get("slot_label") or "")
    is_concurrent = bool(int(row.get("is_concurrent") or 0))
    claimed_stem = normalize_stem(row.get("claimed_stem"))
    role = row.get("layer_role")
    if role in ("bed", "payload", "constituent", "solo"):
        layer_role: LayerRole = role
    else:
        layer_role = derive_layer_role(
            slot_label,
            is_concurrent=is_concurrent,
            claimed_stem=claimed_stem,
        )
    constituents: tuple[str, ...] = ()
    raw_c = row.get("constituents_json")
    if raw_c:
        import json

        try:
            parsed = json.loads(raw_c)
            if isinstance(parsed, list):
                constituents = tuple(str(x) for x in parsed)
        except json.JSONDecodeError:
            pass
    return SlotClaim(
        set_id=str(row.get("set_id") or ""),
        slot_label=slot_label,
        recording_id=str(row.get("recording_id") or row.get("track_id") or ""),
        claimed_stem=claimed_stem,
        claimed_variant=normalize_variant(row.get("claimed_variant")),
        layer_role=layer_role,
        display_name=str(row.get("full_name") or ""),
        constituent_ids=constituents,
        bed_slot=primary_slot_for(slot_label),
        row_index=int(row["row_index"]) if row.get("row_index") is not None else None,
        is_concurrent=is_concurrent,
    )


def evaluate_slot(
    claim: SlotClaim,
    candidates: Iterable[dict[str, Any] | AssetCandidate],
) -> SlotSatisfaction:
    """Evaluate whether inventory satisfies a slot claim."""
    rows: list[dict[str, Any]] = []
    for c in candidates:
        if isinstance(c, AssetCandidate):
            rows.append(
                {
                    "track_audio_id": c.track_audio_id,
                    "recording_id": c.recording_id,
                    "track_id": c.recording_id,
                    "stem": c.stem,
                    "variant": c.variant,
                    "platform": c.platform,
                    "path": c.path,
                    "is_reference": int(c.is_reference),
                }
            )
        else:
            rows.append(c)

    if not rows:
        return SlotSatisfaction(
            claim=claim,
            status=SatisfactionStatus.MISSING,
            detail="no track_audio rows for recording",
        )

    chosen_row, tier = resolve_slot_audio(
        claim.claimed_stem, claim.claimed_variant, rows
    )
    if chosen_row is None or tier is None:
        return SlotSatisfaction(
            claim=claim,
            status=SatisfactionStatus.MISSING,
            detail="resolve_slot_audio returned no candidate",
        )

    chosen = AssetCandidate.from_row(chosen_row)

    if tier == TIER_UNRELATED:
        return SlotSatisfaction(
            claim=claim,
            status=SatisfactionStatus.WRONG_STEM,
            chosen=chosen,
            resolve_tier=tier,
            detail=f"no matching stem/variant (best={chosen.stem}/{chosen.variant})",
        )

    if tier in (TIER_VARIANT_FALLBACK, TIER_STEM_FALLBACK):
        if claim.layer_role == "payload" and claim.claimed_stem == "acappella":
            if tier == TIER_STEM_FALLBACK:
                return SlotSatisfaction(
                    claim=claim,
                    status=SatisfactionStatus.WRONG_STEM,
                    chosen=chosen,
                    resolve_tier=tier,
                    detail="payload slot needs acappella; only regular full track",
                )
        return SlotSatisfaction(
            claim=claim,
            status=SatisfactionStatus.FALLBACK,
            chosen=chosen,
            resolve_tier=tier,
            detail=f"tier={tier_name(tier)}",
        )

    if tier == TIER_EXACT:
        if (
            claim.layer_role == "payload"
            and chosen.stem == "regular"
            and claim.claimed_stem == "regular"
        ):
            return SlotSatisfaction(
                claim=claim,
                status=SatisfactionStatus.WRONG_STEM,
                chosen=chosen,
                resolve_tier=tier,
                detail="payload slot using full regular track; prefer acapella or vocal stem",
            )
        return SlotSatisfaction(
            claim=claim,
            status=SatisfactionStatus.SATISFIED,
            chosen=chosen,
            resolve_tier=tier,
        )

    return SlotSatisfaction(
        claim=claim,
        status=SatisfactionStatus.FALLBACK,
        chosen=chosen,
        resolve_tier=tier,
        detail=f"tier={tier_name(tier)}",
    )


def suggest_actions(satisfaction: SlotSatisfaction) -> tuple[InventoryAction, ...]:
    """Suggest inventory actions for an unsatisfied or fallback slot."""
    c = satisfaction.claim
    base = dict(set_id=c.set_id, slot_label=c.slot_label, recording_id=c.recording_id)
    out: list[InventoryAction] = []

    if satisfaction.status == SatisfactionStatus.MISSING:
        if c.layer_role == "payload" or c.claimed_stem == "acappella":
            out.append(
                InventoryAction(
                    kind=ActionKind.ACQUIRE_ACAPPELLA,
                    detail="missing acappella asset",
                    **base,
                )
            )
        elif c.claimed_stem == "instrumental":
            out.append(
                InventoryAction(
                    kind=ActionKind.ACQUIRE_INSTRUMENTAL,
                    detail="missing instrumental asset",
                    **base,
                )
            )
        else:
            out.append(
                InventoryAction(
                    kind=ActionKind.REPLACE_VERSION,
                    detail="missing reference audio",
                    **base,
                )
            )
        return tuple(out)

    if satisfaction.status == SatisfactionStatus.WRONG_STEM:
        if c.layer_role == "payload":
            out.append(
                InventoryAction(
                    kind=ActionKind.ACQUIRE_ACAPPELLA,
                    detail=satisfaction.detail,
                    **base,
                )
            )
            out.append(
                InventoryAction(
                    kind=ActionKind.USE_DEMUCS_VOCALS,
                    detail="interim: use Demucs vocals off bed or regular reference",
                    **base,
                )
            )
        elif c.claimed_stem == "instrumental":
            out.append(
                InventoryAction(
                    kind=ActionKind.ACQUIRE_INSTRUMENTAL,
                    detail=satisfaction.detail,
                    **base,
                )
            )
        else:
            out.append(
                InventoryAction(
                    kind=ActionKind.REPLACE_VERSION, detail=satisfaction.detail, **base
                )
            )
        return tuple(out)

    if satisfaction.status == SatisfactionStatus.WRONG_RECORDING:
        out.append(
            InventoryAction(
                kind=ActionKind.REPLACE_VERSION,
                detail=satisfaction.detail,
                **base,
            )
        )
        return tuple(out)

    if satisfaction.status == SatisfactionStatus.FALLBACK:
        if c.layer_role == "payload":
            out.append(
                InventoryAction(
                    kind=ActionKind.ACQUIRE_ACAPPELLA,
                    detail=f"fallback tier: {satisfaction.detail}",
                    **base,
                )
            )
        else:
            out.append(
                InventoryAction(
                    kind=ActionKind.REVIEW_MANUAL,
                    detail=f"fallback tier: {satisfaction.detail}",
                    **base,
                )
            )
        return tuple(out)

    return ()


def format_satisfaction_report(
    results: Iterable[SlotSatisfaction],
) -> str:
    lines: list[str] = []
    for s in results:
        chosen = ""
        if s.chosen:
            chosen = (
                f" taid={s.chosen.track_audio_id} ({s.chosen.stem}/{s.chosen.variant})"
            )
        lines.append(
            f"  {s.claim.slot_label:6s} [{s.claim.layer_role:11s}] "
            f"{s.status.value:16s}{chosen}  {s.detail}"
        )
        for act in suggest_actions(s):
            lines.append(f"           -> {act.kind.value}: {act.detail}")
    return "\n".join(lines)


def is_blocking(satisfaction: SlotSatisfaction) -> bool:
    return satisfaction.status in (
        SatisfactionStatus.MISSING,
        SatisfactionStatus.WRONG_RECORDING,
        SatisfactionStatus.WRONG_STEM,
    )


def is_warning(satisfaction: SlotSatisfaction) -> bool:
    return satisfaction.status == SatisfactionStatus.FALLBACK
