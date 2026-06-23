#!/usr/bin/env python
"""Mac-side: append an acquisition-case attempt after an audio fix.

The case corpus lives on the Mac (``data/acquisition_cases/{set_id}.jsonl``).
Executors like ``replace_track_audio.py`` run on **pi-storage** and cannot write
it, so they *emit* a one-line ``ACQUISITION_CASE\\t<json>`` record on success;
this tool persists it into the single Mac-side corpus. It is also usable
standalone with flags for any other fix path (raw SQL, ``apply_stem_matches``,
manual).

This is the same logging seam as the stem driver
([ingest_stem_url.py](ingest_stem_url.py)), just decoupled from the executor so
the corpus stays single-homed.

Examples::

    # pipe the pi-side executor's emit straight into the log
    ssh pi-storage 'cd ~/tracklist_engine && venvs/audio/bin/python \\
        -m scripts.replace_track_audio --track-audio-id 4011 --url URL \\
        --set-id 1fsnxchk --position 081 --axis version --reason "wrong remix"' \\
      | venvs/audio/bin/python scripts/log_acquisition.py --from-stdin

    # or log directly with flags
    venvs/audio/bin/python scripts/log_acquisition.py \\
        --set-id 1fsnxchk --position 081 --recording-id 281u6p4x \\
        --problem wrong_version --url URL --reason "original, not the Syn Cole remix"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.acquisition_case import (
    Actor,
    Attempt,
    AttemptAction,
    AttemptVerdict,
    CaseClaim,
    CaseStatus,
    ProblemClass,
    Resolution,
    record_attempt,
)
from core.identity import normalize_stem

EMIT_PREFIX = "ACQUISITION_CASE\t"


def _coerce(enum_cls, raw, default):
    try:
        return enum_cls(raw)
    except (ValueError, KeyError):
        return default


def _record_from_payload(
    payload: dict, *, root: str | Path, dry_run: bool
) -> str | None:
    """Persist one case record from a dict of fields. Returns case_id or None."""
    set_id = payload.get("set_id")
    slot = payload.get("slot_label") or payload.get("position")
    recording_id = payload.get("recording_id") or payload.get("track_id")
    if not (set_id and slot and recording_id):
        print(f"skip: missing set_id/slot/recording_id in {payload!r}", file=sys.stderr)
        return None

    verdict = _coerce(AttemptVerdict, payload.get("verdict"), AttemptVerdict.PROMOTE)
    action = _coerce(AttemptAction, payload.get("action"), AttemptAction.INGEST_URL)
    actor = _coerce(Actor, payload.get("actor"), Actor.HUMAN)
    problems = tuple(
        p
        for p in (
            _coerce(ProblemClass, x, None) for x in (payload.get("problems") or [])
        )
        if p is not None
    )
    source = payload.get("url")
    checks = payload.get("checks") or {}
    if payload.get("identity"):
        checks = {**checks, "identity": payload["identity"]}

    attempt = Attempt(
        action=action,
        actor=actor,
        url=source,
        platform=payload.get("ref_source"),
        verdict=verdict,
        checks=checks,
        notes=payload.get("reason"),
    )
    promoted = verdict is AttemptVerdict.PROMOTE
    resolution = (
        Resolution(
            track_audio_id=payload.get("track_audio_id"),
            ref_source=payload.get("ref_source"),
            source_path=source,
        )
        if promoted
        else None
    )
    status = (
        _coerce(CaseStatus, payload.get("status"), None)
        if payload.get("status")
        else None
    )

    if dry_run:
        cid = f"{set_id}:{slot}:?"
        print(
            f"[dry-run] would log {cid}  problems={[p.value for p in problems]} "
            f"verdict={verdict.value}"
        )
        return cid

    case = record_attempt(
        set_id=str(set_id),
        slot_label=str(slot),
        recording_id=str(recording_id),
        attempt=attempt,
        claim=CaseClaim(
            recording_id=str(recording_id),
            stem=normalize_stem(payload.get("stem")),
        ),
        resolution=resolution,
        status=status,
        add_problems=problems,
        root=root,
    )
    print(
        f"case-log: {case.case_id} <- {action.value}/{verdict.value} "
        f"(+{len(problems)} problem(s))"
    )
    return case.case_id


def _read_stdin_payloads() -> list[dict]:
    out: list[dict] = []
    for line in sys.stdin:
        line = line.rstrip("\n")
        if line.startswith(EMIT_PREFIX):
            line = line[len(EMIT_PREFIX) :]
        else:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"skip: bad ACQUISITION_CASE json ({e})", file=sys.stderr)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--from-stdin",
        action="store_true",
        help="Consume ACQUISITION_CASE lines emitted by an executor.",
    )
    p.add_argument("--set-id")
    p.add_argument("--position", "--slot", dest="position")
    p.add_argument("--recording-id", "--track-id", dest="recording_id")
    p.add_argument("--url")
    p.add_argument("--reason")
    p.add_argument(
        "--problem",
        action="append",
        default=[],
        help="Problem class (repeatable): wrong_version, suboptimal_stem, ...",
    )
    p.add_argument("--ref-source", default="online_candidate")
    p.add_argument("--action", default="ingest_url")
    p.add_argument(
        "--verdict", default="promote", choices=("promote", "accept", "reject")
    )
    p.add_argument("--stem", default="regular")
    p.add_argument("--status", default=None)
    p.add_argument(
        "--identity", default=None, help="Identity-gate verdict to record in checks."
    )
    p.add_argument("--case-root", default="data/acquisition_cases")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    if args.from_stdin:
        payloads = _read_stdin_payloads()
        if not payloads:
            print("no ACQUISITION_CASE records on stdin", file=sys.stderr)
            return 1
        n = sum(
            1
            for pl in payloads
            if _record_from_payload(pl, root=args.case_root, dry_run=args.dry_run)
        )
        print(f"logged {n}/{len(payloads)} record(s)")
        return 0 if n else 1

    payload = {
        "set_id": args.set_id,
        "slot_label": args.position,
        "recording_id": args.recording_id,
        "url": args.url,
        "reason": args.reason,
        "problems": args.problem,
        "ref_source": args.ref_source,
        "action": args.action,
        "verdict": args.verdict,
        "stem": args.stem,
        "status": args.status,
        "identity": args.identity,
    }
    return (
        0
        if _record_from_payload(payload, root=args.case_root, dry_run=args.dry_run)
        else 2
    )


if __name__ == "__main__":
    sys.exit(main())
