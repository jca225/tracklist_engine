"""Serialize the GT recordings that no predicted span matched — the residual
source's fuel. Pure over ``gt_rows`` / ``spans`` dicts (see score_timeline_vs_gt).
"""

from __future__ import annotations

import json
from pathlib import Path


def never_matched_recordings(gt_rows: list[dict], spans: list[dict]) -> list[dict]:
    """GT rows whose ``track_id`` matched no span's ``recording_id``.

    Covers every stem (not just acappella). Rows without a ``track_id``
    (mix-only / phantom hosts) are skipped — there is nothing to acquire.
    """
    matched = {str(s.get("recording_id")) for s in spans if s.get("recording_id")}
    out: list[dict] = []
    for r in gt_rows:
        tid = r.get("track_id")
        if not tid:
            continue
        if str(tid) in matched:
            continue
        out.append(
            {
                "recording_id": str(tid),
                "slot_label": str(r.get("slot_label") or ""),
                "claimed_stem": str(r.get("claimed_stem") or ""),
                "reason": "never matched by any predicted span",
            }
        )
    return out


def write_never_matched(
    set_id: str, gt_rows: list[dict], spans: list[dict], out_path: Path
) -> dict:
    """Write ``{set_id, never_matched_recordings}`` JSON; return the doc."""
    doc = {
        "set_id": set_id,
        "never_matched_recordings": never_matched_recordings(gt_rows, spans),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return doc
