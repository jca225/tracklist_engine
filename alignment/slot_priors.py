"""Tracklist cue anchors for local span search (from pi set_track_slots)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from core.result import Err, Ok, Result

PI_HOST = "pi-storage"
PI_DB = "/mnt/storage/data/db/music_database.db"
_CACHE = Path(__file__).resolve().parent / ".cache" / "slot_cues"


def normalize_slot(label: str) -> str:
    """Align GT labels (14) with scrape labels (014)."""
    if "w" in label:
        base, _, suffix = label.partition("w")
        if base.isdigit():
            return f"{int(base)}w{suffix}"
        return label
    if label.isdigit():
        return str(int(label))
    return label


def _cache_path(set_id: str) -> Path:
    _CACHE.mkdir(parents=True, exist_ok=True)
    return _CACHE / f"{set_id}_cues.json"


def fetch_slot_cues(set_id: str, *, refresh: bool = False) -> Result[dict[str, float], str]:
    cache = _cache_path(set_id)
    if cache.is_file() and not refresh:
        return Ok(json.loads(cache.read_text()))

    q = (
        "SELECT slot_label, cue_seconds, cue_time_seconds "
        f"FROM set_track_slots WHERE set_id='{set_id}' ORDER BY row_index"
    )
    cmd = ["ssh", PI_HOST, f"sqlite3 -separator '|' {PI_DB} {q!r}"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        return Err((e.stderr or e.stdout or str(e)).strip())

    cues: dict[str, float] = {}
    for ln in r.stdout.splitlines():
        if not ln.strip():
            continue
        parts = ln.split("|")
        if len(parts) != 3:
            continue
        label = normalize_slot(parts[0])
        cue_s, cue_t = parts[1], parts[2]
        val = cue_s if cue_s else cue_t
        if val:
            cues[label] = float(val)
        else:
            cues.setdefault(label, 0.0)
    cache.write_text(json.dumps(cues, indent=2))
    return Ok(cues)


def median_start_by_label(targets) -> dict[str, float]:
    import numpy as np

    by: dict[str, list[float]] = {}
    for t in targets:
        by.setdefault(t.slot_label, []).append(t.set_start_s)
        base = t.slot_label.split("w", 1)[0]
        if base != t.slot_label:
            by.setdefault(base, []).append(t.set_start_s)
    return {k: float(np.median(v)) for k, v in by.items()}


def anchor_for_slot(slot_label: str, train_medians: dict[str, float]) -> float:
    """Mix-time prior from train GT: exact label, else interpolate by slot number."""
    norm = normalize_slot(slot_label)
    if norm in train_medians:
        return train_medians[norm]
    base = norm.split("w", 1)[0]
    if not base.isdigit():
        return 0.0

    by_num: dict[int, list[float]] = {}
    for key, val in train_medians.items():
        kbase = normalize_slot(key).split("w", 1)[0]
        if kbase.isdigit():
            by_num.setdefault(int(kbase), []).append(val)
    points = sorted((num, sum(vs) / len(vs)) for num, vs in by_num.items())
    if not points:
        return 0.0

    n = int(base)
    if n <= points[0][0]:
        return points[0][1]
    if n >= points[-1][0]:
        return points[-1][1]
    for (n0, t0), (n1, t1) in zip(points, points[1:]):
        if n0 <= n <= n1:
            frac = (n - n0) / max(n1 - n0, 1)
            return t0 + frac * (t1 - t0)
    return points[-1][1]


def slot_anchor(
    slot_label: str,
    *,
    train_medians: dict[str, float],
) -> float:
    return anchor_for_slot(slot_label, train_medians)
