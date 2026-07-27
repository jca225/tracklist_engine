"""Overlay only winning instrumental-FP placements onto a baseline timeline.

This is an integration-proof harness, not a production decoder. It isolates the
stem-gated FP change from unrelated agentic channels: every baseline span stays
unchanged unless GT routes it to the instrumental lane and the candidate
placement's winning source includes ``fp``.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from alignment.drivers.base import (
    finalize,
    gt_stem_by_slot,
    norm_slot,
)

_PLACEMENT_FIELDS = (
    "set_start_s",
    "set_end_s",
    "driver_mode",
    "agentic_quality",
    "start_source",
)


def overlay_instrumental_fp(
    baseline: dict,
    candidate: dict,
    *,
    gt_stems: dict[str, str],
) -> tuple[dict, list[str]]:
    """Return baseline plus candidate placements for instrumental FP winners."""
    if baseline.get("set_id") != candidate.get("set_id"):
        raise ValueError("baseline and candidate set_id differ")

    candidate_by_slot = {
        norm_slot(span["slot_label"]): span for span in candidate["spans"]
    }
    out = deepcopy(baseline)
    changed: list[str] = []

    for span in out["spans"]:
        slot = norm_slot(span["slot_label"])
        cand = candidate_by_slot.get(slot)
        if cand is None:
            raise ValueError(f"candidate missing baseline slot {slot}")
        if cand.get("recording_id") != span.get("recording_id"):
            raise ValueError(f"recording_id differs at slot {slot}")
        if gt_stems.get(slot, "regular") != "instrumental":
            continue
        winning_probes = str(cand.get("start_source") or "").split(":")[-1].split("+")
        if "fp" not in winning_probes:
            continue

        before = float(span["set_start_s"])
        after = float(cand["set_start_s"])
        for field in _PLACEMENT_FIELDS:
            if field in cand:
                span[field] = cand[field]
            else:
                span.pop(field, None)
        span["delta_base_set_start_s"] = before
        span["delta_instrumental_fp"] = True
        if after != before:
            changed.append(slot)

    out["driver"] = "agentic instrumental-fp delta"
    out["delta_changed_slots"] = changed
    return out, changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    payload, changed = overlay_instrumental_fp(
        baseline,
        candidate,
        gt_stems=gt_stem_by_slot(args.gt),
    )
    finalize(payload, args.output)
    print(f"{payload['set_id']}: overlaid {len(changed)} instrumental FP placements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
