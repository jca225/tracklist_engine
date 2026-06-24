#!/usr/bin/env python3
"""Promote labeling/identity_overrides YAML into set_track_slots + ledger."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
OVERRIDES = REPO / "labeling" / "identity_overrides"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set-id", required=True)
    ap.add_argument(
        "--db",
        type=Path,
        default=Path(
            os.environ.get("TRACKLIST_DB", "/mnt/storage/data/db/music_database.db")
        ),
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = OVERRIDES / f"{args.set_id}.yaml"
    if not path.is_file():
        print(f"no overrides: {path}", file=sys.stderr)
        return 1
    doc = yaml.safe_load(path.read_text()) or {}
    conn = sqlite3.connect(args.db)
    n = 0
    for o in doc.get("overrides", []):
        tid = o.get("track_id")
        label = o.get("slot_label") or o.get("label")
        if not tid or not label:
            continue
        if args.dry_run:
            print(f"  {label} -> recording_id={tid}")
        else:
            conn.execute(
                "UPDATE set_track_slots SET recording_id = ?, track_id = ? "
                "WHERE set_id = ? AND slot_label = ?",
                (tid, tid, args.set_id, label),
            )
        n += 1
    if not args.dry_run:
        conn.commit()
    print(f"{'Would update' if args.dry_run else 'Updated'} {n} slot(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
