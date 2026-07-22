#!/usr/bin/env python3
"""Read-only scan for stem-candidate wrong-recording mis-attaches.

Seed of Crush Phase-4 part 3 (audit). Here it also backs the validation test:
parse the acquired song from each stem/add correction's reason and (given a DB)
compare to the target recording's title via labels_overlap. NO mutations.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.labels import labels_overlap  # noqa: E402

_FILE_RE = re.compile(r"file:\s*(?P<name>[^;]+)", re.IGNORECASE)
_SLOT_PREFIX = re.compile(r"^\d+(?:w\d+)?__")
_PARENS = re.compile(r"\((?:acapella|acappella|instrumental)[^)]*\)", re.IGNORECASE)


def parse_acquired_song(reason: str) -> str | None:
    """Pull the acquired song title out of a stem/add correction reason."""
    if not reason:
        return None
    m = _FILE_RE.search(reason)
    if not m:
        return None
    name = m.group("name").strip()
    name = _SLOT_PREFIX.sub("", name)
    name = re.sub(r"\.(wav|m4a|mp3|flac|opus)$", "", name, flags=re.IGNORECASE)
    name = _PARENS.sub("", name).strip(" -")
    return name or None


def scan_ledger_tsv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r.get("axis") != "stem" or r.get("action") != "add":
                continue
            song = parse_acquired_song(r.get("reason", ""))
            if song is None:
                continue
            rows.append(
                {
                    "track_id": r.get("track_id", ""),
                    "stem_value": r.get("stem_value", ""),
                    "acquired_song": song,
                    "reason": r.get("reason", ""),
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Scan the correction ledger for stem mis-attaches (read-only)."
    )
    ap.add_argument("--ledger-tsv", type=Path, required=True)
    a = ap.parse_args(argv)
    rows = scan_ledger_tsv(a.ledger_tsv)
    print(f"stem/add rows with a parseable acquired song: {len(rows)}")
    for r in rows[:50]:
        print(f"  {r['track_id']}  {r['stem_value']:12s}  {r['acquired_song']}")
    if len(rows) > 50:
        print(
            f"  ... and {len(rows) - 50} more (NOT truncated in analysis — display cap only)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
