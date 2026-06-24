#!/usr/bin/env python3
"""Summarize track_audio_correction ledger."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--db",
        type=Path,
        default=Path(
            os.environ.get("TRACKLIST_DB", "/mnt/storage/data/db/music_database.db")
        ),
    )
    ap.add_argument("--set-id", default=None, help="Filter by set_id in ledger")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        if args.set_id:
            rows = conn.execute(
                "SELECT axis, action, source FROM track_audio_correction WHERE set_id = ?",
                (args.set_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT axis, action, source FROM track_audio_correction"
            ).fetchall()
    except sqlite3.OperationalError as e:
        print(f"ledger unavailable: {e}", file=sys.stderr)
        return 1

    print(f"corrections: {len(rows)} rows")
    for label, ctr in (
        ("by axis", Counter(r[0] for r in rows)),
        ("by action", Counter(r[1] for r in rows)),
        ("by source", Counter(r[2] or "unknown" for r in rows)),
    ):
        print(f"  {label}:")
        for k, n in ctr.most_common():
            print(f"    {k}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
