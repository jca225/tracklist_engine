#!/usr/bin/env python3
"""Apply reviewed proposed_matches.csv to canonical pi-storage."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = REPO / "venvs" / "audio" / "bin" / "python"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv", type=Path, required=True, help="Reviewed proposed_matches.csv"
    )
    ap.add_argument("--accept", default="accept", help="decision column value to apply")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set-id", default=None)
    ap.add_argument("--reason", default="source:discord|identity:reviewed")
    args = ap.parse_args()

    applied = 0
    with args.csv.open() as f:
        for row in csv.DictReader(f):
            if row.get("decision", "").lower() != args.accept.lower():
                continue
            path = row.get("file") or row.get("path")
            recording_id = row.get("recording_id") or row.get("track_id")
            stem = (row.get("stem") or "acappella").lower()
            if not path or not recording_id:
                continue
            role = "acappella" if "acap" in stem or stem == "vocals" else "instrumental"
            cmd = [
                str(PY),
                str(REPO / "scripts" / "ingest_stem_url.py"),
                "--file",
                path,
                "--track-id",
                recording_id,
                "--role",
                role,
                "--reason",
                args.reason,
            ]
            if args.set_id:
                cmd.extend(["--set-id", args.set_id])
            if args.dry_run:
                cmd.append("--dry-run")
            print("+", " ".join(cmd))
            if not args.dry_run:
                rc = subprocess.call(cmd)
                if rc == 0:
                    applied += 1
            else:
                applied += 1

    print(f"{'Would apply' if args.dry_run else 'Applied'} {applied} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
