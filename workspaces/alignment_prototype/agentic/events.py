"""Append-only event log — the audit trail; belief state = replay of the log.

Every action and decision appends one JSON line. Nothing is ever mutated or
deleted, so any belief state is reproducible by replaying the prefix, every
pseudo-label carries honest provenance (which probe, what confidence, auto vs
human), and a crashed run resumes from its own log.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from workspaces.alignment_prototype.agentic.belief import Observation, SpanBelief

KINDS = ("observe", "commit", "review", "suggest", "escalate", "note")


@dataclass(frozen=True)
class Event:
    seq: int
    span_key: str  # slot_label, or "" for run-level notes
    kind: str  # one of KINDS
    payload: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> Event:
        d = json.loads(line)
        return cls(
            seq=d["seq"],
            span_key=d["span_key"],
            kind=d["kind"],
            payload=d.get("payload", {}),
        )


class EventLog:
    """Append-only JSONL log. ``path=None`` keeps it in memory (tests)."""

    def __init__(self, path: Path | None = None):
        self.path = path
        self._events: list[Event] = []
        if path is not None and path.exists():
            self._events = [
                Event.from_json(ln) for ln in path.read_text().splitlines() if ln
            ]

    def append(self, span_key: str, kind: str, payload: dict) -> Event:
        if kind not in KINDS:
            raise ValueError(f"unknown event kind {kind!r}")
        ev = Event(seq=len(self._events), span_key=span_key, kind=kind, payload=payload)
        self._events.append(ev)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(ev.to_json() + "\n")
        return ev

    @property
    def events(self) -> tuple[Event, ...]:
        return tuple(self._events)

    def observe(self, belief: SpanBelief, obs: Observation) -> SpanBelief:
        """Record an observation and return the updated belief."""
        self.append(
            belief.slot_label,
            "observe",
            {
                "probe": obs.probe,
                "set_start_s": obs.set_start_s,
                "ref_start_s": obs.ref_start_s,
                "confidence": obs.confidence,
                "precision": obs.precision,
                "cost": obs.cost,
                "detail": obs.detail,
            },
        )
        return belief.observe(obs)


def replay_beliefs(
    events: tuple[Event, ...],
    span_meta: dict[str, tuple[str, str]],  # slot_label -> (recording_id, claimed_stem)
) -> dict[str, SpanBelief]:
    """Reconstruct per-span beliefs from an event log (observe events only)."""
    beliefs: dict[str, SpanBelief] = {}
    for ev in events:
        if ev.kind != "observe":
            continue
        rec, stem = span_meta.get(ev.span_key, ("", "regular"))
        b = beliefs.get(ev.span_key) or SpanBelief(
            slot_label=ev.span_key, recording_id=rec, claimed_stem=stem
        )
        p = ev.payload
        beliefs[ev.span_key] = b.observe(
            Observation(
                probe=p["probe"],
                set_start_s=p["set_start_s"],
                confidence=p["confidence"],
                precision=p["precision"],
                cost=p.get("cost", 0.0),
                ref_start_s=p.get("ref_start_s"),
                detail=p.get("detail", ""),
            )
        )
    return beliefs
