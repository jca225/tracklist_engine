"""Materialize a shadow segment bank onto a baseline PredictedTimeline.

Shadow-only: writes a new timeline JSON. Does not touch the default driver or
canonical state. Spans without accepted segments are preserved byte-for-byte.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from workspaces.alignment_prototype.drivers.base import finalize

from .stem_overrides import norm_slot


def _bank_by_slot(bank: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in bank.get("spans") or []:
        out[norm_slot(row["slot_label"])] = row
    return out


def apply_segments_to_span(
    span: dict[str, Any], segments: list[dict[str, Any]]
) -> dict[str, Any]:
    """Return a new span dict with segment-bank placement applied."""
    updated = copy.deepcopy(span)
    mix_starts = [float(s["mix_start_s"]) for s in segments]
    mix_ends = [float(s["mix_end_s"]) for s in segments]
    updated["set_start_s"] = min(mix_starts)
    updated["set_end_s"] = max(mix_ends)
    # Anchor ref_start to the segment that begins at the new set_start.
    first = min(segments, key=lambda s: float(s["mix_start_s"]))
    updated["ref_start_s"] = float(first["ref_start_s"])
    updated["ref_end_s"] = float(max(s["ref_end_s"] for s in segments))
    updated["ref_segments"] = [
        {
            "mix_start_s": float(s["mix_start_s"]),
            "ref_start_s": float(s["ref_start_s"]),
            "ref_end_s": float(s["ref_end_s"]),
        }
        for s in sorted(segments, key=lambda s: float(s["mix_start_s"]))
    ]
    updated["start_source"] = "fp_segment_dp"
    return updated


def materialize_timeline(
    baseline: dict[str, Any],
    bank: dict[str, Any],
    output_path: Path,
) -> Path:
    """Write a validated shadow timeline merging decoded bank rows into baseline."""
    if str(baseline.get("set_id")) != str(bank.get("set_id")):
        raise ValueError("baseline and bank set_id must match")
    by_slot = _bank_by_slot(bank)
    spans_out: list[dict[str, Any]] = []
    n_applied = 0
    for span in baseline["spans"]:
        row = by_slot.get(norm_slot(span["slot_label"]))
        segments = list((row or {}).get("segments") or [])
        if row is not None and row.get("status") == "decoded" and segments:
            spans_out.append(apply_segments_to_span(span, segments))
            n_applied += 1
        else:
            spans_out.append(copy.deepcopy(span))
    payload = copy.deepcopy(baseline)
    payload["spans"] = spans_out
    payload["fp_segment_materialize"] = {
        "shadow_only": True,
        "source_bank_set_id": bank.get("set_id"),
        "observation": bank.get("observation"),
        "lane": bank.get("lane"),
        "spans_applied": n_applied,
    }
    return finalize(payload, output_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    baseline = json.loads(args.baseline.read_text())
    bank = json.loads(args.bank.read_text())
    path = materialize_timeline(baseline, bank, args.output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
