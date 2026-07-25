"""Explain a provenanced value (Law 21) — read-only queries over the substrate.

The three write-side bricks persist claims / evidence / beliefs / observations /
derivations; this is the read side that makes them *useful* rather than
write-only. Two questions it answers:

- ``explain_subject`` — "why does the engine believe X about subject S?" The
  per-axis belief (decision, posterior, entropy), the claim + evidence behind it,
  and every observation on the subject with its diagnostic + producing run.
- ``lineage`` — "what produced this artifact, back to the roots?" Walks the
  derivation graph (the same parent edges the acyclic guard maintains).

Pure reads over a ``core.provenance`` SQLite connection; no writes, no I/O beyond
the DB. ``main()`` is a small CLI over the on-disk provenance DBs.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvidenceView:
    source_family: str
    direction: str
    native_score: dict[str, Any]


@dataclass(frozen=True)
class AxisView:
    """The engine's stance on one axis of one subject."""

    axis: str
    decision: str | None  # None if a claim exists but no belief was inferred
    chosen: str | None
    entropy: float | None
    posterior: dict[str, float]
    candidate: str | None  # the claim's proposed candidate
    evidence: list[EvidenceView]


@dataclass(frozen=True)
class ObservationView:
    predicate: str
    value: Any
    status: str
    diagnostic_code: str | None
    source_confidence: float | None
    source_sha256: str
    producer_process: str


@dataclass(frozen=True)
class HumanAssertionView:
    """One §7 human-label assertion on the subject (latest-first per field)."""

    field: str
    observed_value: Any
    uncertainty_model: dict[str, Any]
    annotator_id: str
    superseded: bool  # a newer assertion supersedes this one


@dataclass(frozen=True)
class SubjectExplanation:
    subject_id: str
    axes: list[AxisView]
    observations: list[ObservationView]
    human_assertions: list[HumanAssertionView] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.axes and not self.observations and not self.human_assertions


@dataclass(frozen=True)
class LineageNode:
    depth: int
    content_sha256: str
    kind: str
    relation: str | None  # edge relation to the child it was reached from
    producer_process: str | None  # the run that output this artifact, if any


def _loads(text: str | None) -> Any:
    return json.loads(text) if text else None


def explain_subject(conn: sqlite3.Connection, subject_id: str) -> SubjectExplanation:
    """Every axis-belief + observation the substrate holds for ``subject_id``."""
    axes: list[AxisView] = []
    # One AxisView per axis that has a claim OR a belief. Beliefs carry the
    # decision; claims carry the candidate + evidence chain.
    axis_names = [
        r["axis"]
        for r in conn.execute(
            "SELECT axis FROM claim WHERE subject_id=? "
            "UNION SELECT axis FROM belief WHERE subject_id=? ORDER BY axis",
            (subject_id, subject_id),
        ).fetchall()
    ]
    for axis in axis_names:
        belief = conn.execute(
            "SELECT decision, chosen, entropy, posterior_json FROM belief "
            "WHERE subject_id=? AND axis=? ORDER BY rowid DESC LIMIT 1",
            (subject_id, axis),
        ).fetchone()
        claim = conn.execute(
            "SELECT claim_id, candidate_id FROM claim "
            "WHERE subject_id=? AND axis=? ORDER BY rowid DESC LIMIT 1",
            (subject_id, axis),
        ).fetchone()
        evidence: list[EvidenceView] = []
        if claim is not None:
            for e in conn.execute(
                "SELECT source_family, direction, native_score_json FROM evidence "
                "WHERE claim_id=?",
                (claim["claim_id"],),
            ).fetchall():
                evidence.append(
                    EvidenceView(
                        source_family=e["source_family"],
                        direction=e["direction"],
                        native_score=_loads(e["native_score_json"]) or {},
                    )
                )
        axes.append(
            AxisView(
                axis=axis,
                decision=belief["decision"] if belief else None,
                chosen=belief["chosen"] if belief else None,
                entropy=belief["entropy"] if belief else None,
                posterior=_loads(belief["posterior_json"]) if belief else {},
                candidate=claim["candidate_id"] if claim else None,
                evidence=evidence,
            )
        )

    observations: list[ObservationView] = []
    for o in conn.execute(
        "SELECT o.predicate, o.value_json, o.status, o.diagnostic_code, "
        "o.source_confidence, o.source_sha256, r.process "
        "FROM observation o JOIN run r ON r.run_id=o.producer_run_id "
        "WHERE o.subject_id=? ORDER BY o.observed_at",
        (subject_id,),
    ).fetchall():
        observations.append(
            ObservationView(
                predicate=o["predicate"],
                value=_loads(o["value_json"]),
                status=o["status"],
                diagnostic_code=o["diagnostic_code"],
                source_confidence=o["source_confidence"],
                source_sha256=o["source_sha256"],
                producer_process=o["process"],
            )
        )

    human: list[HumanAssertionView] = []
    for h in conn.execute(
        "SELECT a.assertion_id, a.field, a.value_json, a.uncertainty_model_json, "
        "b.annotator_id, EXISTS (SELECT 1 FROM human_label_assertion n "
        "WHERE n.supersedes_assertion_id=a.assertion_id) AS superseded "
        "FROM human_label_assertion a "
        "JOIN human_label_bundle b ON b.bundle_id=a.bundle_id "
        "WHERE a.subject_id=? ORDER BY a.created_at DESC",
        (subject_id,),
    ).fetchall():
        human.append(
            HumanAssertionView(
                field=h["field"],
                observed_value=_loads(h["value_json"]),
                uncertainty_model=_loads(h["uncertainty_model_json"]) or {},
                annotator_id=h["annotator_id"],
                superseded=bool(h["superseded"]),
            )
        )
    return SubjectExplanation(subject_id, axes, observations, human)


def lineage(conn: sqlite3.Connection, content_sha256: str) -> list[LineageNode]:
    """The artifact and its transitive parents to the roots (breadth-first)."""
    nodes: list[LineageNode] = []
    seen: set[str] = set()
    # (sha, depth, relation-to-child)
    frontier: list[tuple[str, int, str | None]] = [(content_sha256, 0, None)]
    while frontier:
        sha, depth, relation = frontier.pop(0)
        if sha in seen:
            continue
        seen.add(sha)
        art = conn.execute(
            "SELECT kind FROM artifact WHERE content_sha256=?", (sha,)
        ).fetchone()
        producer = conn.execute(
            "SELECT r.process FROM run_output ro JOIN run r ON r.run_id=ro.run_id "
            "WHERE ro.content_sha256=? LIMIT 1",
            (sha,),
        ).fetchone()
        nodes.append(
            LineageNode(
                depth=depth,
                content_sha256=sha,
                kind=art["kind"] if art else "(unknown)",
                relation=relation,
                producer_process=producer["process"] if producer else None,
            )
        )
        for edge in conn.execute(
            "SELECT parent_sha256, relation FROM derivation WHERE child_sha256=?",
            (sha,),
        ).fetchall():
            frontier.append((edge["parent_sha256"], depth + 1, edge["relation"]))
    return nodes


# --- text rendering ---------------------------------------------------------
def format_subject(exp: SubjectExplanation) -> str:
    lines = [f"subject {exp.subject_id}"]
    if exp.is_empty:
        lines.append("  (nothing recorded)")
        return "\n".join(lines)
    for a in exp.axes:
        verdict = a.decision or "no-belief"
        chosen = f" → {a.chosen}" if a.chosen else ""
        ent = f" H={a.entropy:.2f}" if a.entropy is not None else ""
        lines.append(f"  [{a.axis}] {verdict}{chosen}{ent}")
        if a.candidate:
            lines.append(f"      claim candidate: {a.candidate}")
        for e in a.evidence:
            lines.append(
                f"      evidence: {e.source_family} {e.direction} {e.native_score}"
            )
    for o in exp.observations:
        diag = f"  ⚑{o.diagnostic_code}" if o.diagnostic_code else ""
        conf = (
            f" conf={o.source_confidence:.2f}"
            if o.source_confidence is not None
            else ""
        )
        lines.append(
            f"  <obs {o.predicate}> {o.status}{conf}{diag}  via {o.producer_process}"
        )
    for h in exp.human_assertions:
        old = "  (superseded)" if h.superseded else ""
        kind = h.uncertainty_model.get("kind", "?")
        lines.append(
            f"  <human {h.field}> = {h.observed_value!r} ±{kind} "
            f"by {h.annotator_id}{old}"
        )
    return "\n".join(lines)


def format_lineage(nodes: list[LineageNode]) -> str:
    if not nodes:
        return "(no such artifact)"
    lines = []
    for n in nodes:
        indent = "  " * n.depth
        rel = f" ({n.relation})" if n.relation else ""
        prod = f"  ←{n.producer_process}" if n.producer_process else ""
        lines.append(f"{indent}{n.content_sha256[:12]} [{n.kind}]{rel}{prod}")
    return "\n".join(lines)


def main() -> int:
    import argparse
    from pathlib import Path

    from .store import connect

    ap = argparse.ArgumentParser(description="Explain a provenanced value (Law 21).")
    ap.add_argument("--db", required=True, help="provenance DB root dir")
    ap.add_argument("--subject", help="subject id, e.g. 1fsnxchk::42w3")
    ap.add_argument("--artifact", help="artifact content_sha256 (prefix ok)")
    args = ap.parse_args()

    conn = connect(Path(args.db))
    if args.subject:
        print(format_subject(explain_subject(conn, args.subject)))
    if args.artifact:
        sha = args.artifact
        if len(sha) < 64:  # accept a prefix
            row = conn.execute(
                "SELECT content_sha256 FROM artifact WHERE content_sha256 LIKE ?",
                (sha + "%",),
            ).fetchone()
            if row is None:
                print(f"(no artifact matching {sha})")
                return 1
            sha = row["content_sha256"]
        print(format_lineage(lineage(conn, sha)))
    if not args.subject and not args.artifact:
        ap.error("give --subject or --artifact")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
