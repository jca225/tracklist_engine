"""Timeline provenance stamp + freshness check.

The friction this kills: a predicted timeline is a function of {aligner code,
the claim spine `set_track_slots`, the GT it trained on, the bridge id_map, and
the run's flags}. Any of those can change under a saved timeline, silently making
it stale — the "is this July-6 file current?" fog that cost a whole debugging
session (state-of-record decision #18).

The fix: stamp each timeline with a hash of every input at write time; recompute
those hashes from the CURRENT state at read time and shout if any drifted. No new
tables, no object store — just a `provenance` block in the JSON the aligner
already writes, and one function the scorer calls.

Usage:
    payload["provenance"] = stamp(set_id, rows, args, extra={...})   # at write
    fresh, drift = check(timeline_dict, set_id)                       # at read
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
_IDMAP_DIR = _REPO / "labeling" / "fixtures" / "id_maps"


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _git_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(_REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return r.stdout.strip()
    except Exception:
        return "unknown"


def _file_hash(path: Path) -> str:
    return _sha(path.read_text()) if path.exists() else "absent"


def spine_hash(rows: tuple[dict, ...]) -> str:
    """Hash the identity-bearing spine columns (order-stable)."""
    payload = [
        (
            str(r.get("slot_label")),
            str(r.get("recording_id")),
            str(r.get("claimed_stem")),
        )
        for r in rows
    ]
    return _sha(json.dumps(sorted(payload)))


def _idmap_hash(set_id: str) -> str:
    return _file_hash(_IDMAP_DIR / f"{set_id}.json")


def stamp(
    set_id: str,
    rows: tuple[dict, ...],
    *,
    gt_paths: list[Path] | None = None,
    flags: dict[str, Any] | None = None,
    written_at: str | None = None,
) -> dict:
    """Build the provenance block. `written_at` is passed in (caller owns the
    clock) so this module has no hidden time dependency."""
    gt = {p.name: _file_hash(p) for p in (gt_paths or [])}
    return {
        "code_sha": _git_sha(),
        "spine_hash": spine_hash(rows),
        "idmap_hash": _idmap_hash(set_id),
        "gt_hashes": gt,
        "flags": flags or {},
        "written_at": written_at,
    }


def check(
    timeline: dict,
    set_id: str,
    rows: tuple[dict, ...] | None = None,
    gt_paths: list[Path] | None = None,
) -> tuple[bool, list[str]]:
    """Recompute hashes from CURRENT state; return (fresh, list-of-drifted).

    Components whose current value can't be recomputed (e.g. `rows` not supplied
    because pi is unreachable) are skipped, not falsely flagged. An unstamped
    timeline is reported as drift=['unstamped'] — absence of provenance is itself
    a staleness signal.
    """
    prov = timeline.get("provenance")
    if not prov:
        return False, ["unstamped"]

    drift: list[str] = []
    if prov.get("code_sha") not in (_git_sha(), "unknown"):
        drift.append(f"code_sha ({prov.get('code_sha')} != {_git_sha()})")
    if prov.get("idmap_hash") != _idmap_hash(set_id):
        drift.append("id_map")
    if rows is not None and prov.get("spine_hash") != spine_hash(rows):
        drift.append("set_track_slots (spine)")
    for p in gt_paths or []:
        stamped = (prov.get("gt_hashes") or {}).get(p.name)
        if stamped is not None and stamped != _file_hash(p):
            drift.append(f"GT {p.name}")
    return (not drift), drift
