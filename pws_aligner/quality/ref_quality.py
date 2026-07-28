from __future__ import annotations

"""Ref-quality detectors → typed AbstentionEvents → re-acquire queue (B2).

Wires the existing corpus-health signals into the actuation channel:

  ingest/identity_gate.py verdicts   NO_REFERENCE / MISSING_FILE → NO_DATA;
                                     WRONG_SONG / DURATION_MISMATCH /
                                     LIVE_SUSPECT / FALLBACK_TO_ORIGINAL →
                                     OUT_OF_DOMAIN; WEAK_SIGNAL → LOW_MARGIN
                                     (traceable but NEVER actuates — real
                                     ambiguity is not an ingredient problem).
  duration/preview check             a file far shorter than the scraped
                                     duration is a preview-clip suspect
                                     (the 46s-preview → wrong-version class).
  external suspect lists             corpus_integrity / triage output rows
                                     ({set_id, slot_label, recording_id,
                                     kind, detail}) become events directly.

This module is decision-layer only: it emits a deduped, budget-gated queue
JSON via ``actuation.collect_actuations``. The ingest pipeline is the
effector that drains it — nothing here downloads (detect-then-correct,
never blanket-redownload).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

from ingest.identity_gate import Verdict, VerifyResult, verify_reference_exists

from pws_aligner.quality.actuation import AbstentionEvent, ActuationRequest, actuation_queue_dict
from pws_aligner.quality.actuation import collect_actuations
from pws_aligner.core.votes import AbstainReason

_LF_NAME = "ref_quality"

# A file this much shorter than the scraped duration is a preview suspect.
_PREVIEW_RATIO = 0.6
# ...but only when it is also absolutely short (a 3min radio edit of a 5min
# extended mix is a VARIANT question, not a preview clip).
_PREVIEW_ABS_S = 90.0

_VERDICT_REASON: dict[Verdict, AbstainReason] = {
    Verdict.NO_REFERENCE: AbstainReason.NO_DATA,
    Verdict.MISSING_FILE: AbstainReason.NO_DATA,
    Verdict.WRONG_SONG: AbstainReason.OUT_OF_DOMAIN,
    Verdict.DURATION_MISMATCH: AbstainReason.OUT_OF_DOMAIN,
    Verdict.LIVE_SUSPECT: AbstainReason.OUT_OF_DOMAIN,
    Verdict.FALLBACK_TO_ORIGINAL: AbstainReason.OUT_OF_DOMAIN,
    Verdict.WEAK_SIGNAL: AbstainReason.LOW_MARGIN,
}

_SUSPECT_KIND_REASON: dict[str, AbstainReason] = {
    "missing": AbstainReason.NO_DATA,
    "wrong_version": AbstainReason.OUT_OF_DOMAIN,
    "preview": AbstainReason.OUT_OF_DOMAIN,
    "wrong_master": AbstainReason.OUT_OF_DOMAIN,
}


def event_from_verify(
    set_id: str,
    slot_label: str,
    recording_id: str | None,
    result: VerifyResult,
) -> AbstentionEvent | None:
    """Map an identity_gate VerifyResult to a typed AbstentionEvent.

    OK / SKIPPED verdicts yield None (healthy or unjudgeable). WEAK_SIGNAL
    maps to LOW_MARGIN, which route_abstention refuses to actuate — the event
    exists for traceability, not action.
    """
    reason = _VERDICT_REASON.get(result.verdict)
    if reason is None:
        return None
    return AbstentionEvent(
        set_id=set_id,
        slot_label=slot_label,
        recording_id=recording_id,
        lf=_LF_NAME,
        reason=reason,
        detail=result.detail or result.verdict.value,
    )


def preview_event(
    set_id: str,
    slot_label: str,
    recording_id: str | None,
    *,
    actual_s: float,
    expected_s: float | None,
) -> AbstentionEvent | None:
    """Flag a reference whose duration screams preview clip.

    Requires BOTH a scraped/expected duration to compare against and an
    absolutely-short file: without the expected duration a short file could
    be a genuine radio edit (a variant question for acquisition, not a
    corrupt ingredient — see the short-edit-vs-scraped-duration policy).
    """
    if expected_s is None or expected_s <= 0.0:
        return None
    if actual_s >= _PREVIEW_ABS_S:
        return None
    if actual_s / expected_s >= _PREVIEW_RATIO:
        return None
    return AbstentionEvent(
        set_id=set_id,
        slot_label=slot_label,
        recording_id=recording_id,
        lf=_LF_NAME,
        reason=AbstainReason.OUT_OF_DOMAIN,
        detail=(
            f"preview-clip suspect: {actual_s:.0f}s file vs "
            f"{expected_s:.0f}s scraped duration"
        ),
    )


def suspects_to_events(suspects: Sequence[Mapping]) -> list[AbstentionEvent]:
    """Convert external suspect rows (corpus_integrity / triage lists) to events.

    Each row: ``{set_id, slot_label, recording_id, kind, detail}`` where kind
    is one of ``missing`` / ``wrong_version`` / ``preview`` / ``wrong_master``.
    Unknown kinds are skipped (never guess an actuation from an unrecognized
    signal).
    """
    out: list[AbstentionEvent] = []
    for row in suspects:
        reason = _SUSPECT_KIND_REASON.get(str(row.get("kind", "")))
        if reason is None:
            continue
        out.append(
            AbstentionEvent(
                set_id=str(row.get("set_id", "")),
                slot_label=str(row.get("slot_label", "")),
                recording_id=row.get("recording_id"),
                lf=_LF_NAME,
                reason=reason,
                detail=str(row.get("detail", "")),
            )
        )
    return out


def events_for_slots(
    db_path: str | Path, slots: Sequence[Mapping]
) -> list[AbstentionEvent]:
    """Run the reference-exists gate over slot rows, emitting events for the sick.

    Each slot row: ``{set_id, slot_label, recording_id}``. Slots whose
    reference row exists on disk produce no event.
    """
    db_path = Path(db_path)
    events: list[AbstentionEvent] = []
    for slot in slots:
        rid = slot.get("recording_id")
        if not rid:
            continue
        result = verify_reference_exists(db_path, str(rid))
        ev = event_from_verify(
            set_id=str(slot.get("set_id", "")),
            slot_label=str(slot.get("slot_label", "")),
            recording_id=str(rid),
            result=result,
        )
        if ev is not None:
            events.append(ev)
    return events


def write_reacquire_queue(
    events: Sequence[AbstentionEvent],
    *,
    attempts: Mapping[str, int] | None = None,
    out_path: str | Path,
) -> list[ActuationRequest]:
    """Reduce events to the deduped, budget-gated queue and write it as JSON."""
    requests = collect_actuations(events, attempts)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(actuation_queue_dict(requests), indent=2))
    return requests


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Ref-quality pass: emit a re-acquire queue JSON (no downloads)."
    )
    p.add_argument("--db", type=Path, required=True, help="music_database.db path")
    p.add_argument(
        "--slots",
        type=Path,
        required=True,
        help="JSON array of {set_id, slot_label, recording_id} rows to check",
    )
    p.add_argument(
        "--suspects",
        type=Path,
        default=None,
        help="optional JSON array of external suspect rows (kind/detail)",
    )
    p.add_argument(
        "--attempts",
        type=Path,
        default=None,
        help="optional JSON {recording_id: prior_attempt_count} ledger",
    )
    p.add_argument("--out", type=Path, required=True, help="queue JSON output path")
    args = p.parse_args(argv)

    slots = json.loads(args.slots.read_text())
    events = events_for_slots(args.db, slots)
    if args.suspects is not None:
        events += suspects_to_events(json.loads(args.suspects.read_text()))
    attempts = (
        json.loads(args.attempts.read_text()) if args.attempts is not None else {}
    )
    requests = write_reacquire_queue(events, attempts=attempts, out_path=args.out)
    print(f"wrote {args.out}  ({len(requests)} requests from {len(events)} events)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
