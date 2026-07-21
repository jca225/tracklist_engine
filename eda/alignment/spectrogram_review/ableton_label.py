"""Resolve Ableton lane index + name from GT fixtures (canonical labeling)."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from workspaces.alignment_prototype.score_timeline_vs_gt import norm_slot

_REPO = Path(__file__).resolve().parents[3]

_GT_PATHS = {
    "1fsnxchk": _REPO / "labeling/fixtures/bb12_ground_truth.yaml",
    "2nvzlh2k": _REPO / "labeling/fixtures/bb11_ground_truth.yaml",
}


@lru_cache(maxsize=8)
def _gt_rows(set_id: str) -> list[dict[str, Any]]:
    path = _GT_PATHS.get(set_id)
    if path is None or not path.is_file():
        return []
    doc = yaml.safe_load(path.read_text())
    rows = []
    for r in doc.get("tracks") or []:
        sl = r.get("slot_label")
        if sl is None or str(sl) == "mix":
            continue
        rows.append(r)
    return rows


def parse_ableton_track_num(slot_label: str) -> int | None:
    """'053' / '42w3' / '6w2' → Ableton track index (lane number)."""
    m = re.match(r"^0*(\d+)(w\d+)?$", str(slot_label).strip())
    return int(m.group(1)) if m else None


# Below this gain the clip is effectively muted (fader ducked under another layer).
_LOUD_GAIN = 0.05


def _loud_intervals(r: dict[str, Any]) -> list[tuple[float, float]]:
    """Intervals of [set_start, set_end] where the clip is actually audible.

    Uses ``audible_start_s``/``audible_end_s`` when present (WS0 accounting) and
    carves out regions where the ``gain_curve`` drops below ``_LOUD_GAIN`` — a
    ducked instrumental under a vocal layer is NOT the audible truth there.
    """
    s0 = float(r["set_start_s"])
    s1 = float(r["set_end_s"])
    a0 = r.get("audible_start_s")
    a1 = r.get("audible_end_s")
    lo = float(a0) if isinstance(a0, (int, float)) else s0
    hi = float(a1) if isinstance(a1, (int, float)) else s1
    if hi <= lo:
        lo, hi = s0, s1
    # Start from the audible span, then subtract ducked sub-regions.
    intervals = [(max(s0, lo), min(s1, hi))]
    curve = r.get("gain_curve") or []
    if not curve:
        return intervals
    pts = sorted(
        [(float(t), float(g)) for t, g in curve if isinstance(t, (int, float))],
        key=lambda x: x[0],
    )
    # Build ducked (gain < _LOUD_GAIN) sub-intervals and subtract from `intervals`.
    ducked: list[tuple[float, float]] = []
    for i in range(len(pts) - 1):
        t0, g0 = pts[i]
        t1, _ = pts[i + 1]
        if t1 <= t0:
            continue
        if g0 < _LOUD_GAIN:
            ducked.append((t0, t1))
    if not ducked:
        return intervals
    out: list[tuple[float, float]] = []
    for lo_i, hi_i in intervals:
        cursor = lo_i
        for d0, d1 in ducked:
            if d1 <= cursor or d0 >= hi_i:
                continue
            if d0 > cursor:
                out.append((cursor, min(d0, hi_i)))
            cursor = max(cursor, d1)
            if cursor >= hi_i:
                break
        if cursor < hi_i:
            out.append((cursor, hi_i))
    return [(a, b) for a, b in out if b > a] or [(lo, hi)]


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    return max(0.0, hi - lo)


# "Same song" is decided CANONICALLY by the ``work_siblings`` map (recording_ids
# sharing a ``work_id`` in the canonical DB) — see ``resolve_ableton`` and
# ``scripts/build_work_map.py``. The name-token heuristic that used to live here
# was removed: it collides unrelated songs sharing a common word ("The
# Lawnmower" vs every "The …" song). A prediction whose recording_id is absent
# from the work map is a DATA GAP, surfaced — not guessed via string matching.


def _stem_agrees(pred_stem: str, row_stem: str) -> bool:
    """A pred ``acappella`` agrees with any vocal-bearing row (acappella/regular),
    not with ``instrumental``. ``instrumental`` pred agrees with
    regular/instrumental (instrumental present), not acappella. ``regular`` is
    song-level — agrees with any stem of the same song."""
    p = (pred_stem or "regular").lower()
    r = (row_stem or "regular").lower()
    if p == r:
        return True
    if p == "regular":
        return True  # song-level
    if p == "acappella":
        return r in ("acappella", "regular")  # vocals present
    if p == "instrumental":
        return r in ("instrumental", "regular")  # instrumental present
    return False


def resolve_ableton(
    set_id: str,
    *,
    recording_id: str,
    gt_set_start_s: float,
    fallback_slot: str,
    fallback_name: str,
    gt_stem: str = "regular",
    pred_set_start_s: float | None = None,
    pred_set_end_s: float | None = None,
    pred_stem: str | None = None,
) -> dict[str, Any]:
    """Pick the audible-truth GT row → Ableton lane #, clip name, mix timestamp.

    Does NOT filter by the prediction's ``recording_id`` — that would hide the
    actually-audible clip when the model guessed the wrong recording (e.g. an
    "acappella" pred over a section where a different remix's vocal layer is
    what's loud). Instead, among same-song GT rows overlapping the predicted
    span, prefer (1) stem agreement with the pred, (2) loud overlap with the
    pred window, (3) start-time proximity. ``recording_id`` is a final
    tiebreaker only.
    """
    rows = list(_gt_rows(set_id))
    if not rows:
        return _fallback(fallback_slot, fallback_name, gt_set_start_s)

    # Canonical same-song signal (recording_id -> work siblings) from the
    # canonical DB, built by scripts/build_work_map.py. When present this
    # REPLACES the _same_song name-token heuristic for pool construction, so the
    # website's truth display matches the scorer (no "The Lawnmower"-style
    # collisions). The heuristic remains as a fallback.
    import json as _json

    _wm = _REPO / "labeling" / "fixtures" / "id_maps" / f"{set_id}_work.json"
    work_siblings: dict[str, list[str]] | None = (
        _json.loads(_wm.read_text()) if _wm.is_file() else None
    )
    sibs = set(work_siblings.get(recording_id) or []) if work_siblings else set()
    sibs.add(recording_id)

    pred_lo = pred_set_start_s if pred_set_start_s is not None else gt_set_start_s
    pred_hi = (
        pred_set_end_s
        if pred_set_end_s is not None
        else (gt_set_start_s if pred_set_start_s is None else pred_lo + 1.0)
    )
    pred_stem_norm = (pred_stem or gt_stem or "regular").lower()
    want_name = (fallback_name or "").lower()

    pool = [r for r in rows if str(r.get("track_id") or "") in sibs]
    # Drop rows without usable bounds — they can't be scored for audible overlap.
    pool = [
        r
        for r in pool
        if isinstance(r.get("set_start_s"), (int, float))
        and isinstance(r.get("set_end_s"), (int, float))
    ]
    if not pool:
        return _fallback(fallback_slot, fallback_name, gt_set_start_s)

    BIG = 1e12

    def _score(
        r: dict[str, Any],
    ) -> tuple[float, float, float, float, float, float, float]:
        row_stem = (r.get("claimed_stem") or "regular").lower()
        stem_ok = 0.0 if _stem_agrees(pred_stem_norm, row_stem) else 1.0
        loud = _loud_intervals(r)
        ov = sum(_overlap((pred_lo, pred_hi), seg) for seg in loud)
        # Earliest time inside the pred window where this clip is loud. Clips
        # already loud at pred_lo get onset = pred_lo (best). No overlap → BIG.
        if ov > 0:
            onset = min(
                max(seg[0], pred_lo)
                for seg in loud
                if _overlap((pred_lo, pred_hi), seg) > 0
            )
            no_ov = 0.0
        else:
            onset = BIG
            no_ov = 1.0
        dt = abs(float(r["set_start_s"]) - pred_lo)
        name = str(r.get("track") or "").lower()
        name_pen = (
            0.0
            if (want_name and want_name[:24] in name)
            or (name and name[:24] in want_name)
            else 1.0
        )
        rid_pen = 0.0 if str(r.get("track_id") or "") == str(recording_id) else 1.0
        # 1. prefer stem agreement; 2. prefer any overlap; 3. prefer the SAME
        # recording (clean hit) — but only among rows that DO overlap, so a silent
        # same-recording row can't steal the win over an audible different-recording
        # row; 4. earliest loud onset at/after pred start; 5. more overlap; 6. closer
        # set_start; 7. name.
        return (stem_ok, no_ov, rid_pen, onset, -ov, dt, name_pen)

    g = min(pool, key=_score)
    slot = str(g.get("slot_label") or fallback_slot)
    name = str(g.get("track") or fallback_name)
    start = float(g["set_start_s"])
    end = float(g["set_end_s"])
    num = parse_ableton_track_num(slot)
    if num is None:
        num = parse_ableton_track_num(fallback_slot)
    layer = ""
    for candidate in (slot, fallback_slot):
        m = re.match(r"^0*\d+(w\d+)$", norm_slot(candidate))
        if m:
            layer = m.group(1)
            break
    return {
        "ableton_track_num": num,
        "ableton_slot": slot,
        "ableton_layer": layer,
        "ableton_name": name,
        "ableton_recording_id": str(g.get("track_id") or ""),
        "ableton_stem": (g.get("claimed_stem") or "regular").lower(),
        "mix_start_s": start,
        "mix_end_s": end,
    }


def _fallback(slot: str, name: str, start: float) -> dict[str, Any]:
    num = parse_ableton_track_num(slot)
    layer = ""
    m = re.match(r"^0*\d+(w\d+)$", norm_slot(slot))
    if m:
        layer = m.group(1)
    return {
        "ableton_track_num": num,
        "ableton_slot": slot,
        "ableton_layer": layer,
        "ableton_name": name,
        "ableton_recording_id": "",
        "ableton_stem": "regular",
        "mix_start_s": start,
        "mix_end_s": start,
    }


def format_ableton_line(info: dict[str, Any]) -> str:
    """e.g. 'Ableton Track 53 · Honest (Acappella) · mix 20:11.1'"""
    from eda.alignment.spectrogram_review.labels import fmt_clock

    num = info.get("ableton_track_num")
    track = f"Track {num}" if num is not None else f"slot {info.get('ableton_slot')}"
    layer = info.get("ableton_layer") or ""
    if layer:
        track = f"{track} ({layer})"
    name = info.get("ableton_name") or ""
    t0 = fmt_clock(float(info.get("mix_start_s") or 0.0))
    return f"Ableton {track} · {name} · mix {t0}"
