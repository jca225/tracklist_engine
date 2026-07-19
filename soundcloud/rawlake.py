"""Append-only raw JSONL snapshots — the audit/reprocess layer under the lake."""

from __future__ import annotations

import json
from pathlib import Path


def write_snapshot(
    raw_root: Path, entity: str, entity_id: int, fetched_at: str, records: list[dict]
) -> str:
    safe = fetched_at.replace(":", "-")
    rel = f"{entity}/{entity_id}/{safe}.jsonl"
    path = raw_root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, default=str) + "\n")
    return rel
