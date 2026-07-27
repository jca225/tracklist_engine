"""Materialize a shadow segment bank onto a baseline PredictedTimeline.

Shadow-only: writes a new timeline JSON. Does not touch the default driver or
canonical state. Spans without accepted segments are preserved byte-for-byte.

Optional ``gate_s`` reuses the same consistency idea as instrumental stem /
fp-placement: only apply segments whose mix starts lie within ``gate_s`` of the
baseline ``set_start_s``. Far collision runs keep the baseline. Default gate
``90.0`` matches ``--instr-stem-gate-s`` / fp-placement (not mined on BB11/BB12).

``ref_segments_only`` keeps baseline mix placement and only rewrites
``ref_start_s`` / ``ref_segments`` (avoids set_start regressions from false paths).
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from alignment.drivers.base import finalize

from .stem_overrides import norm_slot

# Same frozen consistency window as instr-stem / fp-placement gates.
DEFAULT_GATE_S = 90.0


def _bank_by_slot(bank: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in bank.get("spans") or []:
        out[norm_slot(row["slot_label"])] = row
    return out


def filter_segments_for_gate(
    segments: list[dict[str, Any]],
    *,
    baseline_set_start_s: float,
    gate_s: float,
) -> list[dict[str, Any]]:
    """Keep only segments whose mix_start is within ``gate_s`` of the prior."""
    if gate_s < 0.0:
        raise ValueError("gate_s must be non-negative")
    return [
        seg
        for seg in segments
        if abs(float(seg["mix_start_s"]) - baseline_set_start_s) <= gate_s
    ]


def apply_segments_to_span(
    span: dict[str, Any],
    segments: list[dict[str, Any]],
    *,
    ref_segments_only: bool = False,
) -> dict[str, Any]:
    """Return a new span dict with segment-bank placement applied.

    When ``ref_segments_only`` is true, keep baseline ``set_start_s`` /
    ``set_end_s`` and only rewrite ref mapping fields.
    """
    updated = copy.deepcopy(span)
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
    if ref_segments_only:
        updated["start_source"] = "fp_segment_dp_ref"
    else:
        mix_starts = [float(s["mix_start_s"]) for s in segments]
        mix_ends = [float(s["mix_end_s"]) for s in segments]
        updated["set_start_s"] = min(mix_starts)
        updated["set_end_s"] = max(mix_ends)
        updated["start_source"] = "fp_segment_dp"
    return updated


def materialize_timeline(
    baseline: dict[str, Any],
    bank: dict[str, Any],
    output_path: Path,
    *,
    gate_s: float | None = None,
    ref_segments_only: bool = False,
) -> Path:
    """Write a validated shadow timeline merging decoded bank rows into baseline.

    When ``gate_s`` is set, segments farther than ``gate_s`` from the baseline
    ``set_start_s`` are dropped; if none remain, the baseline span is kept.
    When ``ref_segments_only`` is set, baseline mix placement is preserved.
    """
    if str(baseline.get("set_id")) != str(bank.get("set_id")):
        raise ValueError("baseline and bank set_id must match")
    if gate_s is not None and gate_s < 0.0:
        raise ValueError("gate_s must be non-negative")
    by_slot = _bank_by_slot(bank)
    spans_out: list[dict[str, Any]] = []
    n_applied = 0
    n_gated_reject = 0
    for span in baseline["spans"]:
        row = by_slot.get(norm_slot(span["slot_label"]))
        segments = list((row or {}).get("segments") or [])
        if row is None or row.get("status") != "decoded" or not segments:
            spans_out.append(copy.deepcopy(span))
            continue
        if gate_s is not None:
            segments = filter_segments_for_gate(
                segments,
                baseline_set_start_s=float(span["set_start_s"]),
                gate_s=gate_s,
            )
            if not segments:
                spans_out.append(copy.deepcopy(span))
                n_gated_reject += 1
                continue
        spans_out.append(
            apply_segments_to_span(span, segments, ref_segments_only=ref_segments_only)
        )
        n_applied += 1
    payload = copy.deepcopy(baseline)
    payload["spans"] = spans_out
    payload["fp_segment_materialize"] = {
        "shadow_only": True,
        "source_bank_set_id": bank.get("set_id"),
        "observation": bank.get("observation"),
        "lane": bank.get("lane"),
        "spans_applied": n_applied,
        "spans_gated_reject": n_gated_reject,
        "gate_s": gate_s,
        "ref_segments_only": ref_segments_only,
    }
    return finalize(payload, output_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gate-s",
        type=float,
        default=None,
        help=f"baseline consistency gate in seconds (use {DEFAULT_GATE_S} to "
        "match instr-stem/fp-placement; omit for ungated)",
    )
    parser.add_argument(
        "--gated",
        action="store_true",
        help=f"shorthand for --gate-s {DEFAULT_GATE_S}",
    )
    parser.add_argument(
        "--ref-segments-only",
        action="store_true",
        help="keep baseline set_start/set_end; only rewrite ref mapping",
    )
    args = parser.parse_args(argv)
    gate_s = DEFAULT_GATE_S if args.gated else args.gate_s
    baseline = json.loads(args.baseline.read_text())
    bank = json.loads(args.bank.read_text())
    path = materialize_timeline(
        baseline,
        bank,
        args.output,
        gate_s=gate_s,
        ref_segments_only=args.ref_segments_only,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
