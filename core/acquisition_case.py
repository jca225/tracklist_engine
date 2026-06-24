"""Acquisition Case Record — the decision trace for getting one slot's audio.

A *case* is opened per ``(set_id, slot_label, layer_role)`` and records the whole
curatorial story of acquiring the right audio for that slot: the claim (what the
tracklist says we need), the problems found with default inventory, every attempt
(search / fetch / separate / mix-extract) and its gate verdicts, and the final
resolution (which file won, why).

GT ``ref_source`` is the *outcome*; the case is the *trace* that produced it. GT
and ``track_audio_correction`` stay authoritative — a case never overrides them.

Storage is line-delimited JSON (one case per line), kept under
``data/acquisition_cases/{set_id}.jsonl`` while the schema is still soft. No DB,
no migration: append fields as real cases demand them.

Substrate rule: this module imports only stdlib + ``core`` (identity / inventory
vocab). It performs file IO at the edges (``load_cases`` / ``save_cases``); the
case values themselves are immutable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.identity import (
    Stem,
    Variant,
    Version,
    normalize_stem,
    normalize_variant,
    normalize_version,
)
from core.slot_inventory import LayerRole

SCHEMA_VERSION = 1


class CaseStatus(str, Enum):
    """Lifecycle of a case."""

    OPEN = "open"  # unresolved; inventory not yet satisfied
    RESOLVED = "resolved"  # a source won and was promoted
    UNRESOLVABLE = "unresolvable"  # no acquirable asset exists (phantom / mix-only)
    HUMAN_REVIEW = "human_review"  # gates ambiguous; needs ears


class ProblemClass(str, Enum):
    """What was wrong with the default inventory (multi-tag)."""

    WRONG_SONG = "wrong_song"  # link is a cover / different track
    WRONG_VERSION = "wrong_version"  # original instead of named remix, etc.
    WRONG_STEM = "wrong_stem"  # got full mix, claim wants acap/instrumental
    WRONG_VARIANT = "wrong_variant"  # radio vs extended
    SUBOPTIMAL_STEM = "suboptimal_stem"  # separated stem where official exists
    MISSING_ASSET = "missing_asset"  # no usable download for the claim
    PHANTOM_TRACK = "phantom_track"  # not online at all; mix-extract host
    COMMUNITY_TAIL = "community_tail"  # SC bootleg / Discord pack only
    STRUCTURE = "structure"  # mashup layering / loop / section-jump
    UNRESOLVED_MANIFEST = "unresolved_manifest"  # ALS path didn't join inventory


class AttemptAction(str, Enum):
    """A step taken to acquire audio for the slot."""

    YTMUSIC_SEARCH = "ytmusic_search"
    YT_SEARCH = "yt_search"
    SC_SEARCH = "sc_search"
    DISCORD_SCRAPE = "discord_scrape"
    SEPARATE = "separate"  # Demucs / Roformer stem split
    MIX_EXTRACT = "mix_extract"  # carve the part out of the set itself
    INGEST_URL = "ingest_url"  # direct URL → canonical store
    MANUAL = "manual"  # human pick, source unrecorded


class AttemptVerdict(str, Enum):
    PENDING = "pending"
    ACCEPT = "accept"
    REJECT = "reject"
    PROMOTE = "promote"  # accepted AND made the reference


class Actor(str, Enum):
    HUMAN = "human"
    AGENT = "agent"
    INGEST = "ingest"  # automated pipeline (ingest.main)
    BACKFILL = "backfill"  # reconstructed after the fact


@dataclass(frozen=True)
class CaseClaim:
    """What the tracklist says this slot needs."""

    recording_id: str
    display_name: str = ""
    version: Version = "original"
    stem: Stem = "regular"
    variant: Variant = "regular"
    scrape_urls: tuple[str, ...] = ()


@dataclass(frozen=True)
class Attempt:
    """One acquisition step and its gate verdicts.

    ``checks`` is intentionally an open mapping (e.g. ``version_gate``,
    ``fingerprint``, ``duration_ratio``, ``bleed_residual_db``) until the gate
    battery freezes its vocabulary — see the acquisition-case plan.
    """

    action: AttemptAction
    actor: Actor = Actor.HUMAN
    at: str | None = None  # ISO-8601; None when unknown (backfill)
    query: str | None = None
    url: str | None = None
    platform: str | None = None
    verdict: AttemptVerdict = AttemptVerdict.PENDING
    checks: Mapping[str, Any] = field(default_factory=dict)
    reject_reason: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class Resolution:
    """The winning asset for the slot."""

    track_audio_id: int | None = None
    ref_source: str | None = None  # online_candidate / demucs / reference / mix
    resolve_tier: int | None = None
    source_path: str | None = None
    promoted_at: str | None = None
    gt_confirmed: bool = False  # a human used this asset in Ableton GT


@dataclass(frozen=True)
class TrainingSignal:
    """Preference data harvested from the case for a future stem ranker."""

    negatives: tuple[str, ...] = ()  # rejected source urls/paths
    preference_pairs: tuple[tuple[str, str], ...] = ()  # (winner, loser)


@dataclass(frozen=True)
class AcquisitionCase:
    set_id: str
    slot_label: str
    layer_role: LayerRole
    claim: CaseClaim
    status: CaseStatus = CaseStatus.OPEN
    problem_classes: tuple[ProblemClass, ...] = ()
    attempts: tuple[Attempt, ...] = ()
    resolution: Resolution | None = None
    training: TrainingSignal = field(default_factory=TrainingSignal)
    notes: str = ""

    @property
    def case_id(self) -> str:
        return f"{self.set_id}:{self.slot_label}:{self.layer_role}"


# ── immutable updates ────────────────────────────────────────────────────────


def append_attempt(case: AcquisitionCase, attempt: Attempt) -> AcquisitionCase:
    """Return a new case with ``attempt`` appended (input unchanged)."""
    return replace(case, attempts=case.attempts + (attempt,))


def with_problem_classes(
    case: AcquisitionCase, *classes: ProblemClass
) -> AcquisitionCase:
    """Return a new case with ``classes`` merged in (order-preserving, deduped)."""
    seen = list(case.problem_classes)
    for c in classes:
        if c not in seen:
            seen.append(c)
    return replace(case, problem_classes=tuple(seen))


# ── serialization (lenient: unknown enum strings fall back, not raise) ────────


def _coerce_enum[E: Enum](enum_cls: type[E], raw: Any, default: E) -> E:
    try:
        return enum_cls(raw)
    except (ValueError, KeyError):
        return default


def attempt_to_dict(a: Attempt) -> dict[str, Any]:
    return {
        "action": a.action.value,
        "actor": a.actor.value,
        "at": a.at,
        "query": a.query,
        "url": a.url,
        "platform": a.platform,
        "verdict": a.verdict.value,
        "checks": dict(a.checks),
        "reject_reason": a.reject_reason,
        "notes": a.notes,
    }


def attempt_from_dict(d: Mapping[str, Any]) -> Attempt:
    return Attempt(
        action=_coerce_enum(AttemptAction, d.get("action"), AttemptAction.MANUAL),
        actor=_coerce_enum(Actor, d.get("actor"), Actor.HUMAN),
        at=d.get("at"),
        query=d.get("query"),
        url=d.get("url"),
        platform=d.get("platform"),
        verdict=_coerce_enum(AttemptVerdict, d.get("verdict"), AttemptVerdict.PENDING),
        checks=dict(d.get("checks") or {}),
        reject_reason=d.get("reject_reason"),
        notes=d.get("notes"),
    )


def case_to_dict(case: AcquisitionCase) -> dict[str, Any]:
    res = case.resolution
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case.case_id,
        "set_id": case.set_id,
        "slot_label": case.slot_label,
        "layer_role": case.layer_role,
        "status": case.status.value,
        "problem_classes": [p.value for p in case.problem_classes],
        "claim": {
            "recording_id": case.claim.recording_id,
            "display_name": case.claim.display_name,
            "version": case.claim.version,
            "stem": case.claim.stem,
            "variant": case.claim.variant,
            "scrape_urls": list(case.claim.scrape_urls),
        },
        "attempts": [attempt_to_dict(a) for a in case.attempts],
        "resolution": (
            None
            if res is None
            else {
                "track_audio_id": res.track_audio_id,
                "ref_source": res.ref_source,
                "resolve_tier": res.resolve_tier,
                "source_path": res.source_path,
                "promoted_at": res.promoted_at,
                "gt_confirmed": res.gt_confirmed,
            }
        ),
        "training": {
            "negatives": list(case.training.negatives),
            "preference_pairs": [list(p) for p in case.training.preference_pairs],
        },
        "notes": case.notes,
    }


def case_from_dict(d: Mapping[str, Any]) -> AcquisitionCase:
    claim_d = d.get("claim") or {}
    claim = CaseClaim(
        recording_id=str(claim_d.get("recording_id") or ""),
        display_name=str(claim_d.get("display_name") or ""),
        version=normalize_version(claim_d.get("version")),
        stem=normalize_stem(claim_d.get("stem")),
        variant=normalize_variant(claim_d.get("variant")),
        scrape_urls=tuple(claim_d.get("scrape_urls") or ()),
    )
    res_d = d.get("resolution")
    resolution = (
        None
        if not res_d
        else Resolution(
            track_audio_id=res_d.get("track_audio_id"),
            ref_source=res_d.get("ref_source"),
            resolve_tier=res_d.get("resolve_tier"),
            source_path=res_d.get("source_path"),
            promoted_at=res_d.get("promoted_at"),
            gt_confirmed=bool(res_d.get("gt_confirmed")),
        )
    )
    train_d = d.get("training") or {}
    training = TrainingSignal(
        negatives=tuple(train_d.get("negatives") or ()),
        preference_pairs=tuple(
            (str(p[0]), str(p[1]))
            for p in (train_d.get("preference_pairs") or ())
            if len(p) == 2
        ),
    )
    problems = tuple(
        _coerce_enum(ProblemClass, p, ProblemClass.MISSING_ASSET)
        for p in (d.get("problem_classes") or [])
    )
    role = d.get("layer_role")
    layer_role: LayerRole = (
        role if role in ("bed", "payload", "constituent", "solo") else "solo"
    )
    return AcquisitionCase(
        set_id=str(d.get("set_id") or ""),
        slot_label=str(d.get("slot_label") or ""),
        layer_role=layer_role,
        claim=claim,
        status=_coerce_enum(CaseStatus, d.get("status"), CaseStatus.OPEN),
        problem_classes=problems,
        attempts=tuple(attempt_from_dict(a) for a in (d.get("attempts") or [])),
        resolution=resolution,
        training=training,
        notes=str(d.get("notes") or ""),
    )


# ── JSONL file IO (edges) ─────────────────────────────────────────────────────


def load_cases(path: str | Path) -> list[AcquisitionCase]:
    """Read a ``{set_id}.jsonl`` file → cases. Missing file → empty list."""
    p = Path(path)
    if not p.exists():
        return []
    out: list[AcquisitionCase] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(case_from_dict(json.loads(line)))
    return out


def save_cases(path: str | Path, cases: Iterable[AcquisitionCase]) -> None:
    """Write cases as line-delimited JSON (one per line), creating parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(case_to_dict(c), ensure_ascii=False, sort_keys=True) for c in cases
    ]
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def default_path(set_id: str, *, root: str | Path = "data/acquisition_cases") -> Path:
    return Path(root) / f"{set_id}.jsonl"


# ── find-or-create mutation (the live-logging entry point) ────────────────────


def find_case_index(
    cases: list[AcquisitionCase], slot_label: str, recording_id: str
) -> int | None:
    """Index of the case matching ``(slot_label, recording_id)``, or None."""
    for i, c in enumerate(cases):
        if c.slot_label == slot_label and c.claim.recording_id == recording_id:
            return i
    return None


def record_attempt(
    *,
    set_id: str,
    slot_label: str,
    recording_id: str,
    attempt: Attempt,
    claim: CaseClaim | None = None,
    layer_role: LayerRole | None = None,
    resolution: Resolution | None = None,
    status: CaseStatus | None = None,
    add_problems: Iterable[ProblemClass] = (),
    root: str | Path = "data/acquisition_cases",
) -> AcquisitionCase:
    """Append ``attempt`` to the ``(set_id, slot_label, recording_id)`` case.

    Loads the set's JSONL, finds-or-creates the case, appends the attempt and
    any resolution/status/problem updates, writes the file back, and returns the
    updated case. This is the seam acquisition tools call so a manual fix logs
    its own decision trace. ``claim`` / ``layer_role`` seed a freshly created
    case (derived from the slot when omitted).
    """
    from core.slot_inventory import derive_layer_role

    path = default_path(set_id, root=root)
    cases = load_cases(path)
    idx = find_case_index(cases, slot_label, recording_id)
    if idx is None:
        seed_claim = claim or CaseClaim(recording_id=recording_id)
        role = layer_role or derive_layer_role(slot_label, claimed_stem=seed_claim.stem)
        cases.append(
            AcquisitionCase(
                set_id=set_id,
                slot_label=slot_label,
                layer_role=role,
                claim=seed_claim,
                status=CaseStatus.OPEN,
            )
        )
        idx = len(cases) - 1

    case = append_attempt(cases[idx], attempt)
    if add_problems:
        case = with_problem_classes(case, *add_problems)
    if resolution is not None:
        case = replace(case, resolution=resolution)
    if status is not None:
        case = replace(case, status=status)
    cases[idx] = case
    save_cases(path, cases)
    return case
