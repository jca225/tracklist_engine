#!/usr/bin/env python3
"""Report ref_source distribution from set_ground_truth."""

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
    ap.add_argument("--set-id", default=None)
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    if args.set_id:
        rows = conn.execute(
            "SELECT ref_source FROM set_ground_truth WHERE set_id = ?",
            (args.set_id,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT ref_source FROM set_ground_truth").fetchall()

    ctr = Counter(r[0] or "reference" for r in rows)
    total = sum(ctr.values()) or 1
    print(f"ref_source report ({sum(ctr.values())} rows):")
    for src, n in ctr.most_common():
        print(f"  {src:20s} {n:5d}  ({100 * n / total:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
