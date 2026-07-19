"""JSONL persistence and read-only baseline candidate extraction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from .schema import CandidateBankMetadata, CandidateEvidence, PlacementCandidate


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_baseline_bank(
    timeline_path: Path,
    output_path: Path,
    *,
    producer_sha: str,
) -> Path:
    """Materialize a provenance-stamped baseline bank from a timeline file."""
    timeline = json.loads(timeline_path.read_text())
    metadata = CandidateBankMetadata(
        producer_sha=producer_sha,
        source_timeline_sha256=sha256_file(timeline_path),
    )
    return write_candidate_bank(
        output_path,
        metadata,
        baseline_candidates(timeline),
    )


def baseline_candidates(timeline: dict) -> tuple[PlacementCandidate, ...]:
    """Materialize one immutable baseline candidate per input span."""
    set_id = str(timeline.get("set_id") or "")
    return tuple(
        PlacementCandidate(
            set_id=set_id,
            slot_label=str(span["slot_label"]),
            recording_id=str(span["recording_id"]),
            claimed_stem=str(span.get("claimed_stem") or "regular"),
            source="baseline",
            rank=0,
            set_start_s=float(span["set_start_s"]),
            ref_start_s=(
                float(span["ref_start_s"])
                if span.get("ref_start_s") is not None
                else None
            ),
            native_confidence=None,
            evidence=CandidateEvidence(
                baseline_delta_s=0.0,
                baseline_source=str(span.get("start_source") or "unknown"),
            ),
        )
        for span in timeline["spans"]
    )


def write_candidate_bank(
    path: Path,
    metadata: CandidateBankMetadata,
    candidates: Iterable[PlacementCandidate],
) -> Path:
    """Write a deterministic metadata-first JSONL candidate bank."""
    candidates = tuple(candidates)
    keys = [candidate.key for candidate in candidates]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate candidate key")
    rows = [
        {"record_type": "metadata", **metadata.to_dict()},
        *(
            {"record_type": "candidate", **candidate.to_dict()}
            for candidate in candidates
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )
    return path


def read_candidate_bank(
    path: Path,
) -> tuple[CandidateBankMetadata, tuple[PlacementCandidate, ...]]:
    """Read and validate a metadata-first JSONL candidate bank."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or rows[0].pop("record_type", None) != "metadata":
        raise ValueError("candidate bank must start with metadata")
    metadata = CandidateBankMetadata.from_dict(rows[0])
    candidates: list[PlacementCandidate] = []
    for row in rows[1:]:
        if row.pop("record_type", None) != "candidate":
            raise ValueError("unexpected candidate bank record type")
        candidates.append(PlacementCandidate.from_dict(row))
    keys = [candidate.key for candidate in candidates]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate candidate key")
    return metadata, tuple(candidates)
