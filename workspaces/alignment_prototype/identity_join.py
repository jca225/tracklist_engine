"""Correct pred↔GT identity-join primitives + a guard against the slot-label trap.

Finding (2026-07-23, cost real debugging time): the GT fixture and
`set_track_slots` number slots on DIFFERENT schemes — timeline slot "2" and GT
slot "2" are *different songs*. Joining predictions to GT by `slot_label`
silently yields garbage (a "4% identity" artifact this session). The correct
per-span identity join is **TIME-overlap** (the canonical scorer,
`score_timeline_vs_gt` / `build_span_table`); the correct *set-level* signal is
the recording bag below. This module exposes the id canonicalization every join
needs (predictions carry canonical `recording_id`; GT carries fixture `track_id`
— they must be bridged through the set's id_map first), plus `slots_share_songs`
so a regression test can lock the finding in.
"""

from __future__ import annotations

import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_IDMAP = _REPO / "labeling" / "fixtures" / "id_maps"


def _idmap(set_id: str) -> dict:
    p = _IDMAP / f"{set_id}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def bridge(recording_id: str, set_id: str) -> str:
    """Canonicalize any scrape/spine/GT id to the recording namespace the
    predictions use, via the set's bridge id_map. ALWAYS bridge before comparing
    a GT `track_id` to a predicted `recording_id`."""
    return _idmap(set_id).get(str(recording_id), str(recording_id))


def gt_recordings(gt_doc: dict, set_id: str) -> set[str]:
    """Canonical recordings the GT asserts were played (content rows only)."""
    return {
        bridge(str(r["track_id"]), set_id)
        for r in gt_doc["tracks"]
        if r.get("track_id") and str(r.get("slot_label")) != "mix"
    }


def pred_recordings(timeline: dict, set_id: str) -> set[str]:
    """Canonical recordings the aligner emitted."""
    return {
        bridge(str(s["recording_id"]), set_id)
        for s in timeline["spans"]
        if s.get("recording_id")
    }


def slots_share_songs(timeline: dict, gt_doc: dict) -> float:
    """Fraction of shared normalized slot_labels whose predicted `name` and GT
    `track` name agree (token overlap ≥ 0.5). Near-1 would mean slot_labels ARE
    a valid cross-source key; on this data it is near-0 — the guard. Used by the
    regression test, not for scoring.
    """
    from workspaces.alignment_prototype.score_timeline_vs_gt import norm_slot

    def toks(s: str) -> set[str]:
        return {w for w in str(s).lower().replace("-", " ").split() if len(w) > 2}

    tl = {norm_slot(str(s["slot_label"])): s.get("name", "") for s in timeline["spans"]}
    gt: dict[str, str] = {}
    for r in gt_doc["tracks"]:
        sl = r.get("slot_label")
        if sl and str(sl) != "mix" and r.get("track_id"):
            gt.setdefault(norm_slot(str(sl)), r.get("track", ""))
    shared = set(tl) & set(gt)
    if not shared:
        return 0.0
    agree = 0
    for sl in shared:
        a, b = toks(tl[sl]), toks(gt[sl])
        if a and b and len(a & b) / min(len(a), len(b)) >= 0.5:
            agree += 1
    return agree / len(shared)
