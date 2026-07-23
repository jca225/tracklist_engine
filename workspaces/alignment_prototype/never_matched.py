"""Serialize the GT recordings that no predicted span matched — the residual
source's fuel. Pure over ``gt_rows`` / ``spans`` dicts (see score_timeline_vs_gt).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def _ss(row: dict) -> float:
    try:
        return float(row.get("set_start_s"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def assign_spans_to_forms(
    gt_rows: list[dict], spans: list[dict]
) -> tuple[dict[int, dict], list[dict]]:
    """Injective per-recording assignment of predicted spans to GT forms.

    Within each recording (``track_id``==``recording_id``), assign spans to forms
    greedily by ascending ``|Δset_start|`` with no reuse — each span gets at most
    one form and each form at most one span. Greedy is optimal for the
    handful-sized per-recording sets here.

    Returns ``(assigned, unmatched_forms)`` where ``assigned`` maps ``id(span)``
    to its GT form row (spans that won a form) and ``unmatched_forms`` lists the
    GT rows no span was assigned to — the honest recall loss. Trackless GT rows
    are skipped (nothing to bridge to a span).
    """
    by_tid: dict[str, list[dict]] = defaultdict(list)
    for row in gt_rows:
        if row.get("track_id"):
            by_tid[str(row["track_id"])].append(row)
    spans_by_rid: dict[str, list[dict]] = defaultdict(list)
    for span in spans:
        spans_by_rid[str(span.get("recording_id") or "")].append(span)

    assigned: dict[int, dict] = {}
    unmatched: list[dict] = []
    for rid, forms in by_tid.items():
        sp = spans_by_rid.get(rid, [])
        pairs = sorted(
            (
                (abs(_ss(s) - _ss(f)), si, fi)
                for si, s in enumerate(sp)
                for fi, f in enumerate(forms)
            ),
            key=lambda t: t[0],
        )
        used_s: set[int] = set()
        used_f: set[int] = set()
        for _dist, si, fi in pairs:
            if si in used_s or fi in used_f:
                continue
            assigned[id(sp[si])] = forms[fi]
            used_s.add(si)
            used_f.add(fi)
        unmatched.extend(f for fi, f in enumerate(forms) if fi not in used_f)
    return assigned, unmatched


def unmatched_gt_forms(gt_rows: list[dict], spans: list[dict]) -> list[dict]:
    """GT rows (forms) that no predicted span SELECTS under the scorer's
    nearest-``set_start`` assignment.

    ``never_matched_recordings`` is recording-level: it marks a whole recording
    matched if ANY span shares its ``track_id``. That hides the multi-form loss
    -- when a slot is played across a stem transition (instrumental->full;
    acappella over instrumental) GT carries several rows for one recording, but
    ``score_timeline_vs_gt`` scores each span against only its nearest row
    (``g = min(rows, key=|set_start diff|)``), so the other forms are silently
    unscored. This returns exactly those unselected forms -- the honest loss the
    per-span scorecard omits. Trackless rows (no bridge to a span) are skipped.
    """
    by_tid: dict[str, list[dict]] = defaultdict(list)
    for row in gt_rows:
        if row.get("track_id"):
            by_tid[str(row["track_id"])].append(row)

    selected: set[int] = set()
    for span in spans:
        rid = str(span.get("recording_id") or "")
        rows = by_tid.get(rid)
        if not rows:
            continue
        picked = min(rows, key=lambda r: abs(_ss(r) - _ss(span)))
        selected.add(id(picked))

    out: list[dict] = []
    for row in gt_rows:
        if not row.get("track_id") or id(row) in selected:
            continue
        out.append(
            {
                "recording_id": str(row["track_id"]),
                "slot_label": str(row.get("slot_label") or ""),
                "claimed_stem": str(row.get("claimed_stem") or ""),
                "set_start_s": _ss(row),
                "set_end_s": float(row.get("set_end_s") or _ss(row)),
                "reason": "form not selected by any predicted span",
            }
        )
    return out


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
