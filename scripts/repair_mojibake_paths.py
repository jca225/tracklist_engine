#!/usr/bin/env python3
"""Repair double-encoded (mojibake) `track_audio.path` rows (issue #74, step 2).

**Ordering is load-bearing — run this ONLY after the pi UTF-8 locale fix (step
1).** Under pi's default `iso8859-1` locale a repaired path (correct UTF-8, e.g.
containing `U+2013`) cannot be encoded to the filesystem, so pi-local access
would break. This tool therefore *refuses* `--apply` unless the running process
is under a UTF-8 filesystem encoding — the locale fix is a hard gate, not a
convention.

The disk is already correct (the file's real name is proper UTF-8); only the DB
string is mojibake. So this NEVER touches files — it rewrites the DB `path` to
the value that actually resolves, after verifying that value exists on disk.

This is an *encoding* repair, not an identity correction, so it is audit-logged
(stdout) rather than written to `track_audio_correction` (whose axes are
version/variant/stem/recording).

    # dry-run (safe anywhere): show what WOULD change
    python scripts/repair_mojibake_paths.py --db /mnt/storage/data/db/music_database.db
    # apply (pi, ONLY after the UTF-8 locale fix):
    LANG=C.UTF-8 python scripts/repair_mojibake_paths.py --db <db> --apply
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys

from core.mojibake import is_double_encoded_utf8, repair_double_encoded_utf8

CANONICAL_DB = "/mnt/storage/data/db/music_database.db"


def _fs_is_utf8() -> bool:
    return (os.environ.get("PYTHONUTF8") == "1") or (
        "utf-8" in sys.getfilesystemencoding().lower().replace("_", "-")
    )


def find_mojibake(conn: sqlite3.Connection) -> list[tuple[int, str, str, bool]]:
    """Return (track_audio_id, current_path, repaired_path, repaired_exists) for
    every mojibake row. `repaired_exists` is a disk stat of the repaired path
    (meaningful only under a UTF-8 locale)."""
    rows = conn.execute(
        "SELECT track_audio_id, path FROM track_audio "
        "WHERE path IS NOT NULL AND path != ''"
    ).fetchall()
    out: list[tuple[int, str, str, bool]] = []
    for taid, path in rows:
        if not is_double_encoded_utf8(path):
            continue
        repaired = repair_double_encoded_utf8(path)
        try:
            exists = os.path.exists(repaired)
        except (OSError, UnicodeError):
            exists = False
        out.append((int(taid), path, repaired, exists))
    return out


def run(db_path: str, *, apply: bool) -> int:
    if apply and not _fs_is_utf8():
        print(
            "REFUSING --apply: filesystem encoding is "
            f"{sys.getfilesystemencoding()!r}, not UTF-8. Run the pi locale fix "
            "first (issue #74 step 1), then re-run under LANG=C.UTF-8 / "
            "PYTHONUTF8=1. Repairing under a non-UTF-8 locale would store a path "
            "pi-local processes cannot open.",
            file=sys.stderr,
        )
        return 2

    mode = sqlite3.connect(f"file:{db_path}?mode={'rw' if apply else 'ro'}", uri=True)
    try:
        rows = find_mojibake(mode)
        print(f"mojibake rows: {len(rows)}  (db={db_path}, apply={apply})\n")
        applied = 0
        skipped = 0
        for taid, cur, repaired, exists in rows:
            tag = "OK" if exists else "MISSING-ON-DISK"
            print(f"  taid={taid} [{tag}]")
            print(f"    from: {cur!r}")
            print(f"    to  : {repaired!r}")
            if not apply:
                continue
            if not exists:
                # Never point the DB at a path that doesn't resolve — leave it
                # for manual inspection rather than silently breaking the row.
                print("    SKIP: repaired path not found on disk")
                skipped += 1
                continue
            mode.execute(
                "UPDATE track_audio SET path=? WHERE track_audio_id=?",
                (repaired, taid),
            )
            applied += 1
        if apply:
            mode.commit()
            print(f"\napplied={applied} skipped={skipped}")
        return 1 if (apply and skipped) else 0
    finally:
        mode.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=os.environ.get("TRACKLIST_DB", CANONICAL_DB))
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Write repairs (default: dry-run). Refused unless FS encoding is UTF-8.",
    )
    args = ap.parse_args(argv)
    return run(args.db, apply=args.apply)


if __name__ == "__main__":
    sys.exit(main())
