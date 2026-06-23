#!/usr/bin/env python3
"""Backfill set_track_slots.layer_role (and constituents_json per section)."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core.slot_inventory import derive_layer_role

_W = re.compile(r"^(\d+)w\d+$")


def backfill(db_path: Path, *, dry_run: bool = False) -> int:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cols = {r[1] for r in conn.execute("PRAGMA table_info(set_track_slots)")}
    if "layer_role" not in cols:
        print(
            "layer_role column missing — run scripts/migrate_layer_role.sql first",
            file=sys.stderr,
        )
        return 1

    rows = conn.execute(
        "SELECT set_id, row_index, slot_label, is_concurrent, claimed_stem, "
        "COALESCE(recording_id, track_id) AS rid "
        "FROM set_track_slots ORDER BY set_id, row_index"
    ).fetchall()

    updates: list[tuple[str, str | None, str, int]] = []
    by_set_section: dict[tuple[str, str], list[str]] = {}

    for r in rows:
        label = r["slot_label"] or ""
        role = derive_layer_role(
            label,
            is_concurrent=bool(r["is_concurrent"]),
            claimed_stem=r["claimed_stem"],
        )
        primary = (
            label if not _W.match(label) else f"{int(_W.match(label).group(1)):03d}"
        )
        if _W.match(label):
            by_set_section.setdefault((r["set_id"], primary), []).append(r["rid"])
        updates.append((role, r["set_id"], label, r["row_index"]))

    constituents_updates: list[tuple[str | None, str, int]] = []
    for r in rows:
        label = r["slot_label"] or ""
        if not _W.match(label):
            primary = label
            cids = by_set_section.get((r["set_id"], primary), [])
            constituents_updates.append(
                (
                    json.dumps(cids) if cids else None,
                    r["set_id"],
                    r["row_index"],
                )
            )

    print(f"Would update layer_role on {len(updates)} rows")
    if dry_run:
        for role, sid, label, _ in updates[:10]:
            print(f"  {sid} {label} -> {role}")
        return 0

    for role, sid, label, ri in updates:
        conn.execute(
            "UPDATE set_track_slots SET layer_role = ? WHERE set_id = ? AND row_index = ?",
            (role, sid, ri),
        )
    if "constituents_json" in cols:
        for cjson, sid, ri in constituents_updates:
            conn.execute(
                "UPDATE set_track_slots SET constituents_json = ? "
                "WHERE set_id = ? AND row_index = ? AND slot_label NOT GLOB '*w*'",
                (cjson, sid, ri),
            )
    conn.commit()
    print(f"Updated {len(updates)} rows")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--db",
        type=Path,
        default=Path(
            os.environ.get("TRACKLIST_DB", "/mnt/storage/data/db/music_database.db")
        ),
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    return backfill(args.db, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
