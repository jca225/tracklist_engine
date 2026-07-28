from __future__ import annotations

"""Actuation channel — the ACTION arm of the PWS aligner (active inference).

The labeling functions are *perception*: they observe a span and emit a noisy
vote. Active inference (the Solms-Friston spine the reframe rests on) has a second
arm — minimize free energy not only by updating beliefs but by **acting on the
world to change what you sense**. This module is that arm: when an LF abstains
because the *ingredient is bad* (the reference audio is missing, or is the wrong
version / a preview clip / the wrong master), instead of a dead NULL vote the
system emits an **actuation** — a request to re-acquire the reference — then
re-perceives once the corpus is fixed.

This is the decision/queue layer only. It decides WHAT to re-acquire and WHY and
emits a queue; the ingest pipeline is the *effector* that drains the queue and
performs downloads. Keeping the two separate means this channel adds no downloads
and does not collide with the ingest pipeline.

The gate is the typed abstention already in the LF contract (``AbstainReason``):
- ``NO_DATA`` / ``OUT_OF_DOMAIN`` — the abstention is an *ingredient* problem → act.
- ``LOW_MARGIN`` — the abstention is a genuine *ambiguity* → do NOT touch the
  corpus (this is the "detect-then-correct, never blanket-redownload" principle:
  the reason, not mere uncertainty, licenses a re-acquire).
A per-recording attempt budget prevents infinite redownload loops.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from pws_aligner.core.votes import AbstainReason

_MAX_ATTEMPTS = 2


class ActuationKind(Enum):
    FETCH_MISSING = "fetch_missing"
    REACQUIRE_WRONG_VERSION = "reacquire_wrong_version"


@dataclass(frozen=True)
class AbstentionEvent:
    """One LF's abstention on a span, with enough context to act on it.

    ``recording_id`` is the candidate the LF was evaluating (the thing to
    re-acquire). ``reason`` is the typed abstention; ``detail`` is a free-text
    corrective hint (e.g. "46s preview clip", "radio edit vs extended").
    """

    set_id: str
    slot_label: str
    recording_id: str | None
    lf: str
    reason: AbstainReason
    detail: str = ""


@dataclass(frozen=True)
class ActuationRequest:
    """A re-acquire action emitted to the ingest effector queue."""

    recording_id: str
    set_id: str
    slot_label: str
    kind: str  # ActuationKind value
    reason: str
    lf: str
    attempt: int


def route_abstention(
    event: AbstentionEvent,
    attempt_count: int = 0,
    *,
    max_attempts: int = _MAX_ATTEMPTS,
) -> ActuationRequest | None:
    """Map one abstention to a re-acquire request, or None if it should not act.

    Actuates only when (a) we know which recording to fetch, (b) the abstention
    reason is an ingredient problem (NO_DATA / OUT_OF_DOMAIN), and (c) the
    per-recording attempt budget is not exhausted. LOW_MARGIN (real ambiguity)
    and NONE (the LF fired) never actuate.
    """
    if not event.recording_id:
        return None  # no known target to re-acquire
    if attempt_count >= max_attempts:
        return None  # budget exhausted — stop the redownload loop
    if event.reason == AbstainReason.NO_DATA:
        kind = ActuationKind.FETCH_MISSING
        reason = event.detail or "reference audio missing/unreadable"
    elif event.reason == AbstainReason.OUT_OF_DOMAIN:
        kind = ActuationKind.REACQUIRE_WRONG_VERSION
        reason = event.detail or "reference looks like the wrong version/preview/master"
    else:
        return None  # NONE (fired) or LOW_MARGIN (ambiguity) — do not act
    return ActuationRequest(
        recording_id=event.recording_id,
        set_id=event.set_id,
        slot_label=event.slot_label,
        kind=kind.value,
        reason=reason,
        lf=event.lf,
        attempt=attempt_count,
    )


def collect_actuations(
    events: Sequence[AbstentionEvent],
    attempts: Mapping[str, int] | None = None,
    *,
    max_attempts: int = _MAX_ATTEMPTS,
) -> list[ActuationRequest]:
    """Reduce many abstention events to a deduped, budget-gated re-acquire queue.

    A recording missing across many spans yields ONE request. ``attempts`` carries
    prior re-acquire counts per recording (from the acquisition ledger) so an
    already-retried recording is not re-queued past its budget.
    """
    attempts = attempts or {}
    out: list[ActuationRequest] = []
    produced: set[str] = set()
    for event in events:
        rid = event.recording_id
        if rid is None or rid in produced:
            continue
        req = route_abstention(event, attempts.get(rid, 0), max_attempts=max_attempts)
        if req is not None:
            out.append(req)
            produced.add(rid)
    return out


def actuation_queue_dict(requests: Sequence[ActuationRequest]) -> list[dict]:
    """Serialize the re-acquire queue for the ingest effector to drain."""
    return [
        {
            "recording_id": r.recording_id,
            "set_id": r.set_id,
            "slot_label": r.slot_label,
            "kind": r.kind,
            "reason": r.reason,
            "lf": r.lf,
            "attempt": r.attempt,
        }
        for r in requests
    ]
