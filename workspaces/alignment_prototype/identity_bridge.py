"""One set-scoped legacy track-id → canonical recording-id boundary.

GT fixtures preserve some historical scrape IDs while current manifests and
predictions use recording IDs.  Consumers must normalize once, before joining;
ad-hoc per-command bridges caused the scorer to work while trajectory training
silently skipped the same rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

_REPO = Path(__file__).resolve().parents[2]


def load_identity_map(set_id: str, *, repo: Path = _REPO) -> dict[str, str]:
    path = repo / "labeling" / "fixtures" / "id_maps" / f"{set_id}.json"
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text())
    return {str(old): str(new) for old, new in raw.items() if old and new}


def canonical_recording_id(
    track_id: object, identity_map: dict[str, str]
) -> str | None:
    if track_id is None or str(track_id) == "":
        return None
    tid = str(track_id)
    return identity_map.get(tid, tid)


def canonicalize_gt_rows(
    set_id: str,
    rows: Iterable[dict],
    *,
    repo: Path = _REPO,
) -> list[dict]:
    identity_map = load_identity_map(set_id, repo=repo)
    out: list[dict] = []
    for source in rows:
        row = dict(source)
        rid = canonical_recording_id(row.get("track_id"), identity_map)
        if rid is not None:
            row["track_id"] = rid
        out.append(row)
    return out

