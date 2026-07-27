"""Decode calibrated axis beliefs into an off-path provenanced timeline.

This is the Phase-2 strangler seam.  It consumes only ``AxisBelief`` posteriors;
it never reads raw probe margins.  The source timeline supplies non-axis display
metadata and identity until the identity cutover, while accepted PLACEMENT and
STRUCTURE beliefs replace their respective coordinates.

Abstention is explicit in the canonical decoded document:

* unresolved placement -> ``set_start_s=None`` and ``set_end_s=None``;
* unresolved structure -> ``ref_start_s=None``, ``ref_end_s=None``, and
  ``ref_segments=None``.

The canonical scorer still expects its legacy non-null span schema.  Therefore
``materialize_evaluation_view`` is a separate compatibility view which drops
unresolved-placement spans and records coverage; it does not fabricate zeros.
This module is not imported by inference and changes no hot-path behavior.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.provenance import (
    Artifact,
    ArtifactKind,
    ArtifactStore,
    Axis,
    AxisBelief,
    Decision,
    ProvenanceRepository,
    Run,
    SubjectRef,
)

_BUNDLE_SCHEMA = "axis-belief-bundle-v1"
_DECODE_SCHEMA = "belief-timeline-v1"
_PROCESS = "timeline.decode_from_beliefs"
_VERSION = "1"


@dataclass(frozen=True)
class DecodedTimeline:
    document: dict[str, Any]
    artifact: Artifact
    run: Run
    n_spans: int
    n_placement_abstained: int
    n_structure_abstained: int


def _belief_record(belief: AxisBelief) -> dict[str, Any]:
    return {
        "belief_id": belief.belief_id,
        "subject": {
            "type": belief.subject.subject_type,
            "id": belief.subject.subject_id,
        },
        "axis": belief.axis.value,
        "inference_run_id": belief.inference_run_id,
        "posterior": dict(belief.posterior),
        "entropy": belief.entropy,
        "decision": belief.decision.value,
        "chosen": belief.chosen,
        "contributing_evidence_ids": list(belief.contributing_evidence_ids),
        "supersedes_belief_id": belief.supersedes_belief_id,
    }


def axis_belief_bundle(
    set_id: str, beliefs: Sequence[AxisBelief]
) -> dict[str, Any]:
    """Return a deterministic JSON-safe bundle for transport/evaluation."""
    records = [_belief_record(b) for b in beliefs]
    records.sort(key=lambda r: (r["subject"]["id"], r["axis"], r["belief_id"]))
    return {"schema": _BUNDLE_SCHEMA, "set_id": set_id, "beliefs": records}


def beliefs_from_bundle(bundle: Mapping[str, Any]) -> list[AxisBelief]:
    """Strictly reconstruct axis beliefs from a JSON-decoded bundle."""
    if bundle.get("schema") != _BUNDLE_SCHEMA:
        raise ValueError(f"unsupported belief bundle schema: {bundle.get('schema')!r}")
    beliefs: list[AxisBelief] = []
    for row in bundle.get("beliefs", []):
        subject = row["subject"]
        beliefs.append(
            AxisBelief(
                belief_id=str(row["belief_id"]),
                subject=SubjectRef(str(subject["type"]), str(subject["id"])),
                axis=Axis(str(row["axis"])),
                inference_run_id=str(row["inference_run_id"]),
                posterior={
                    str(key): float(value)
                    for key, value in dict(row["posterior"]).items()
                },
                entropy=float(row["entropy"]),
                decision=Decision(str(row["decision"])),
                chosen=str(row["chosen"]) if row.get("chosen") is not None else None,
                contributing_evidence_ids=tuple(
                    str(value) for value in row["contributing_evidence_ids"]
                ),
                supersedes_belief_id=(
                    str(row["supersedes_belief_id"])
                    if row.get("supersedes_belief_id") is not None
                    else None
                ),
            )
        )
    return beliefs


def write_belief_bundle(
    path: str | Path, set_id: str, beliefs: Sequence[AxisBelief]
) -> None:
    Path(path).write_text(
        json.dumps(axis_belief_bundle(set_id, beliefs), indent=2, sort_keys=True)
        + "\n"
    )


def read_belief_bundle(path: str | Path) -> tuple[str, list[AxisBelief]]:
    bundle = json.loads(Path(path).read_text())
    return str(bundle["set_id"]), beliefs_from_bundle(bundle)


def _chosen_payload(belief: AxisBelief, expected_schema: str) -> dict[str, Any]:
    if belief.decision is not Decision.ACCEPTED or belief.chosen is None:
        raise ValueError("cannot decode an unresolved belief")
    try:
        payload = json.loads(belief.chosen)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{belief.axis.value} belief {belief.belief_id} has a non-JSON candidate"
        ) from exc
    if payload.get("schema") != expected_schema:
        raise ValueError(
            f"{belief.axis.value} belief {belief.belief_id} candidate schema "
            f"{payload.get('schema')!r} != {expected_schema!r}"
        )
    return payload


def _belief_index(
    set_id: str, beliefs: Sequence[AxisBelief]
) -> dict[tuple[str, Axis], AxisBelief]:
    out: dict[tuple[str, Axis], AxisBelief] = {}
    prefix = f"{set_id}::"
    for belief in beliefs:
        if belief.axis not in (Axis.PLACEMENT, Axis.STRUCTURE):
            continue
        if not belief.subject.subject_id.startswith(prefix):
            raise ValueError(
                f"belief subject {belief.subject.subject_id!r} is not in set {set_id}"
            )
        key = (belief.subject.subject_id, belief.axis)
        if key in out:
            raise ValueError(f"duplicate {belief.axis.value} belief for {key[0]}")
        out[key] = belief
    return out


def decode_timeline_document(
    timeline: Mapping[str, Any],
    beliefs: Sequence[AxisBelief],
) -> dict[str, Any]:
    """Return a JSON-safe timeline decoded only from calibrated axis beliefs."""
    doc = copy.deepcopy(dict(timeline))
    set_id = str(doc.get("set_id") or "")
    if not set_id:
        raise ValueError("timeline has no set_id")
    index = _belief_index(set_id, beliefs)
    source_bytes = json.dumps(timeline, sort_keys=True, separators=(",", ":")).encode()
    source_sha = hashlib.sha256(source_bytes).hexdigest()

    for span in doc.get("spans", []):
        subject_id = f"{set_id}::{span['slot_label']}"
        try:
            placement = index[(subject_id, Axis.PLACEMENT)]
            structure = index[(subject_id, Axis.STRUCTURE)]
        except KeyError as exc:
            raise ValueError(f"missing axis belief for {subject_id}") from exc

        old_start = span.get("set_start_s")
        old_end = span.get("set_end_s")
        source_structure = {
            "ref_start_s": span.get("ref_start_s"),
            "ref_end_s": span.get("ref_end_s"),
            "ref_segments": copy.deepcopy(span.get("ref_segments")),
        }
        duration = (
            float(old_end) - float(old_start)
            if old_start is not None and old_end is not None
            else None
        )

        if placement.decision is Decision.ACCEPTED:
            payload = _chosen_payload(placement, "placement-v1")
            set_start_s = float(payload["set_start_s"])
            span["set_start_s"] = set_start_s
            span["set_end_s"] = (
                set_start_s + duration if duration is not None else None
            )
        else:
            span["set_start_s"] = None
            span["set_end_s"] = None

        if structure.decision is Decision.ACCEPTED:
            payload = _chosen_payload(structure, "structure-v1")
            if payload.get("mix_time_frame") != "set_absolute_s":
                raise ValueError(
                    f"structure belief {structure.belief_id} has ambiguous mix frame"
                )
            segments = list(payload["segments"])
            span["ref_segments"] = segments
            span["ref_start_s"] = (
                float(segments[0]["ref_start_s"]) if segments else None
            )
            span["ref_end_s"] = (
                float(segments[-1]["ref_end_s"]) if segments else None
            )
        else:
            span["ref_segments"] = None
            span["ref_start_s"] = None
            span["ref_end_s"] = None

        span["axis_beliefs"] = {
            axis.value: {
                "belief_id": belief.belief_id,
                "inference_run_id": belief.inference_run_id,
                "decision": belief.decision.value,
                "chosen": belief.chosen,
                "posterior": dict(belief.posterior),
                "entropy": belief.entropy,
                "contributing_evidence_ids": list(
                    belief.contributing_evidence_ids
                ),
            }
            for axis, belief in (
                (Axis.PLACEMENT, placement),
                (Axis.STRUCTURE, structure),
            )
        }
        span["source_axis_values"] = {"structure": source_structure}

    doc["belief_decode"] = {
        "schema": _DECODE_SCHEMA,
        "source_timeline_sha256": source_sha,
        "axes": [Axis.PLACEMENT.value, Axis.STRUCTURE.value],
        "decoder_input": "calibrated_axis_posteriors",
    }
    return doc


def materialize_evaluation_view(decoded: Mapping[str, Any]) -> dict[str, Any]:
    """Legacy-scorer view: omit placement abstentions and report coverage.

    Structure abstentions remain as spans so identity and placement are still
    scored; their legacy trajectory fields are restored as an empty list only
    in this compatibility view and explicitly marked as abstained.
    """
    doc = copy.deepcopy(dict(decoded))
    original = list(doc.get("spans", []))
    kept: list[dict[str, Any]] = []
    structure_abstained = 0
    for span in original:
        if span.get("set_start_s") is None or span.get("set_end_s") is None:
            continue
        if span.get("ref_segments") is None:
            structure_abstained += 1
            legacy = span["source_axis_values"]["structure"]
            span["ref_segments"] = legacy["ref_segments"] or []
            span["ref_start_s"] = legacy["ref_start_s"]
            span["ref_end_s"] = legacy["ref_end_s"]
            span["structure_abstained"] = True
        kept.append(span)
    doc["spans"] = kept
    doc["belief_evaluation"] = {
        "source_spans": len(original),
        "placement_accepted": len(kept),
        "placement_abstained": len(original) - len(kept),
        "structure_abstained_among_placement_accepted": structure_abstained,
        "compatibility_note": (
            "legacy source structure fields are retained only for schema "
            "compatibility and excluded from structure scoring; canonical "
            "decoded timeline keeps None"
        ),
    }
    return doc


def prepare_belief_evaluation_timeline(
    source_timeline_path: str | Path,
    belief_bundle_path: str | Path,
    output_path: str | Path,
    *,
    expected_set_id: str,
) -> dict[str, int]:
    """Write the explicit legacy-scorer compatibility view.

    This function is called only by the scorer's opt-in
    ``--axis-belief-bundle`` seam. It returns coverage counts so every measured
    result carries the abstention denominator beside it.
    """
    timeline = json.loads(Path(source_timeline_path).read_text())
    bundle_set_id, beliefs = read_belief_bundle(belief_bundle_path)
    timeline_set_id = str(timeline.get("set_id") or "")
    if bundle_set_id != expected_set_id or timeline_set_id != expected_set_id:
        raise ValueError(
            "set mismatch: "
            f"expected={expected_set_id!r}, timeline={timeline_set_id!r}, "
            f"beliefs={bundle_set_id!r}"
        )
    decoded = decode_timeline_document(timeline, beliefs)
    view = materialize_evaluation_view(decoded)
    Path(output_path).write_text(
        json.dumps(view, indent=2, sort_keys=True) + "\n"
    )
    return {
        key: int(value)
        for key, value in view["belief_evaluation"].items()
        if isinstance(value, int)
    }


def decode_and_register_timeline(
    timeline_path: str | Path,
    beliefs: Sequence[AxisBelief],
    *,
    store: ArtifactStore,
    repo: ProvenanceRepository,
    code_commit: str = "unknown",
) -> DecodedTimeline:
    """Decode, content-address, and link the output to timeline + belief bundle."""
    timeline_path = Path(timeline_path)
    source_raw = timeline_path.read_bytes()
    timeline = json.loads(source_raw)
    set_id = str(timeline.get("set_id") or "")
    bundle = axis_belief_bundle(set_id, beliefs)
    bundle_raw = json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode()
    decoded = decode_timeline_document(timeline, beliefs)
    decoded_raw = (
        json.dumps(decoded, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )

    source_artifact = store.put_bytes(
        source_raw,
        kind=ArtifactKind.PREDICTED_TIMELINE,
        media_type="application/json",
        source_uri=str(timeline_path),
        source_system="alignment_prototype.infer",
    )
    belief_artifact = store.put_bytes(
        bundle_raw,
        kind=ArtifactKind.DIAGNOSTICS,
        media_type="application/json",
        source_system="core.provenance.AxisBelief",
    )
    decoded_artifact = store.put_bytes(
        decoded_raw,
        kind=ArtifactKind.PREDICTED_TIMELINE,
        media_type="application/json",
        source_system=_PROCESS,
    )

    run = repo.begin_run(
        process=_PROCESS,
        process_version=_VERSION,
        code_commit=code_commit,
        params_hash="0",
    )
    try:
        repo.add_input(run, source_artifact, role="source_timeline", ordinal=0)
        repo.add_input(run, belief_artifact, role="axis_beliefs", ordinal=1)
        repo.add_output(run, decoded_artifact, role="decoded_timeline")
        repo.derive(decoded_artifact, source_artifact, run, relation="decoded_from")
        repo.derive(decoded_artifact, belief_artifact, run, relation="decoded_from")
        repo.succeed(run)
    except Exception as exc:
        repo.fail(run, {"error": repr(exc)})
        raise

    spans = list(decoded.get("spans", []))
    return DecodedTimeline(
        document=decoded,
        artifact=decoded_artifact,
        run=run,
        n_spans=len(spans),
        n_placement_abstained=sum(
            span.get("set_start_s") is None for span in spans
        ),
        n_structure_abstained=sum(
            span.get("ref_segments") is None for span in spans
        ),
    )
