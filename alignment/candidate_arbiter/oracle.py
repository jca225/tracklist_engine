"""Candidate-oracle headroom measurement using evaluation-only ground truth."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import yaml

from alignment.drivers.base import finalize, norm_slot
from alignment.path_decode import gt_placement_onset

from .io import read_candidate_bank
from .schema import PlacementCandidate


@dataclass(frozen=True)
class CandidateError:
    candidate: PlacementCandidate
    error_s: float


def _gt_by_recording(
    gt_path: Path,
    *,
    set_id: str,
) -> dict[str, tuple[dict, ...]]:
    document = yaml.safe_load(gt_path.read_text())
    id_map_path = gt_path.parent / "id_maps" / f"{set_id}.json"
    id_map: dict[str, str] = (
        json.loads(id_map_path.read_text()) if id_map_path.is_file() else {}
    )
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in document["tracks"]:
        if str(row.get("slot_label")) == "mix":
            continue
        recording_id = str(row.get("track_id") or "")
        recording_id = id_map.get(recording_id, recording_id)
        if recording_id:
            grouped[recording_id].append(row)
    return {key: tuple(rows) for key, rows in grouped.items()}


def label_candidate_errors(
    candidates: tuple[PlacementCandidate, ...],
    *,
    gt_path: Path,
) -> tuple[CandidateError, ...]:
    """Attach scorer-compatible placement error to candidates with matching GT."""
    if not candidates:
        return ()
    set_id = candidates[0].set_id
    if any(candidate.set_id != set_id for candidate in candidates):
        raise ValueError("candidate bank contains multiple set_ids")
    by_recording = _gt_by_recording(gt_path, set_id=set_id)
    out: list[CandidateError] = []
    for candidate in candidates:
        rows = by_recording.get(candidate.recording_id)
        if not rows:
            continue
        row = min(
            rows,
            key=lambda item: abs(
                float(item["set_start_s"]) - candidate.set_start_s
            ),
        )
        out.append(
            CandidateError(
                candidate=candidate,
                error_s=abs(gt_placement_onset(row) - candidate.set_start_s),
            )
        )
    return tuple(out)


def select_oracle_candidates(
    rows: tuple[CandidateError, ...],
) -> dict[tuple[str, str], PlacementCandidate]:
    """Select the lowest-error existing candidate for each predicted span."""
    grouped: dict[tuple[str, str], list[CandidateError]] = defaultdict(list)
    for row in rows:
        key = (
            norm_slot(row.candidate.slot_label),
            row.candidate.recording_id,
        )
        grouped[key].append(row)
    return {
        key: min(options, key=lambda option: option.error_s).candidate
        for key, options in grouped.items()
    }


def apply_oracle_selection(
    timeline: dict,
    selected: dict[tuple[str, str], PlacementCandidate],
) -> dict:
    """Apply placement-only oracle choices while preserving ref-content decode."""
    out = deepcopy(timeline)
    changed = 0
    for span in out["spans"]:
        key = (norm_slot(span["slot_label"]), str(span["recording_id"]))
        candidate = selected.get(key)
        if candidate is None:
            continue
        before = float(span["set_start_s"])
        after = candidate.set_start_s
        span["set_start_s"] = after
        span["set_end_s"] = float(span["set_end_s"]) + (after - before)
        span["oracle_candidate_source"] = candidate.source
        span["oracle_candidate_rank"] = candidate.rank
        changed += after != before
    out["driver"] = "candidate oracle (evaluation only)"
    out["oracle_changed_spans"] = changed
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    timeline = json.loads(args.timeline.read_text())
    _metadata, candidates = read_candidate_bank(args.candidates)
    if candidates and candidates[0].set_id != timeline.get("set_id"):
        raise ValueError("timeline and candidate bank set_id differ")
    errors = label_candidate_errors(candidates, gt_path=args.gt)
    selected = select_oracle_candidates(errors)
    payload = apply_oracle_selection(timeline, selected)
    finalize(payload, args.output)
    print(
        f"{timeline['set_id']}: selected {len(selected)} spans; "
        f"changed {payload['oracle_changed_spans']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
