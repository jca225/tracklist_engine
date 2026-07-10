#!/usr/bin/env python3
"""Corpus data-integrity checks - the DATA analogue of scripts/guardrails.py.

`guardrails.py` polices the *source tree* (stale names, dead flags). This
polices the *canonical corpus*: the identity invariants that docs assert
(docs/inventory_coherence_contract.md, docs/audio_identity_taxonomy) but that
no tool mechanically enforced until now. Each invariant is one SQL query that
returns the count (and a sample) of violating rows - the "singular-test"
pattern, kept dependency-free (sqlite3 stdlib only) so it runs on pi-storage.

Severities:
  ERROR - a structural violation that corrupts analysis/alignment (must be 0).
  WARN  - an acquisition/routing backlog, not corruption (informational).

Run against the canonical DB on pi::

    make check-corpus                 # ssh pi + canonical DB
    python scripts/corpus_integrity.py --db /path/to/music_database.db

Exit 0 if no ERROR-severity violations, 1 otherwise.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Invariant:
    key: str
    severity: str  # "ERROR" | "WARN"
    title: str
    # count_sql returns one row, one column: the number of violations.
    count_sql: str
    # sample_sql returns a few offending rows for the report (optional).
    sample_sql: str = ""


INVARIANTS: tuple[Invariant, ...] = (
    Invariant(
        "dup_reference",
        "ERROR",
        "recording with more than one is_reference=1",
        "SELECT COUNT(*) FROM (SELECT recording_id FROM track_audio "
        "WHERE is_reference=1 GROUP BY recording_id HAVING COUNT(*)>1)",
        "SELECT recording_id, COUNT(*) n FROM track_audio WHERE is_reference=1 "
        "GROUP BY recording_id HAVING n>1 LIMIT 8",
    ),
    Invariant(
        "reference_on_stem_when_regular_exists",
        "ERROR",
        "is_reference points at a stem though a regular row exists",
        "SELECT COUNT(*) FROM track_audio ta WHERE ta.is_reference=1 "
        "AND ta.stem!='regular' AND EXISTS(SELECT 1 FROM track_audio t2 "
        "WHERE t2.recording_id=ta.recording_id AND t2.stem='regular')",
        "SELECT recording_id, stem, track_audio_id FROM track_audio ta "
        "WHERE ta.is_reference=1 AND ta.stem!='regular' AND EXISTS("
        "SELECT 1 FROM track_audio t2 WHERE t2.recording_id=ta.recording_id "
        "AND t2.stem='regular') LIMIT 8",
    ),
    Invariant(
        "null_path",
        "ERROR",
        "track_audio row with NULL/empty path",
        "SELECT COUNT(*) FROM track_audio WHERE path IS NULL OR path=''",
        "SELECT track_audio_id, recording_id, stem FROM track_audio "
        "WHERE path IS NULL OR path='' LIMIT 8",
    ),
    Invariant(
        "no_recording_id",
        "ERROR",
        "track_audio row with no recording_id",
        "SELECT COUNT(*) FROM track_audio WHERE recording_id IS NULL",
        "SELECT track_audio_id, stem, substr(path,-40) FROM track_audio "
        "WHERE recording_id IS NULL LIMIT 8",
    ),
    Invariant(
        "regular_claim_served_by_stem",
        "ERROR",
        "recording a set plays as regular, but its reference is a stem "
        "(the DJ Kool delete-before-insert class)",
        "SELECT COUNT(*) FROM (SELECT DISTINCT ta.recording_id FROM track_audio ta "
        "JOIN set_track_slots s ON s.recording_id=ta.recording_id "
        "WHERE ta.is_reference=1 AND ta.stem!='regular' AND s.claimed_stem='regular')",
        "SELECT DISTINCT ta.recording_id, ta.stem, s.full_name FROM track_audio ta "
        "JOIN set_track_slots s ON s.recording_id=ta.recording_id "
        "WHERE ta.is_reference=1 AND ta.stem!='regular' AND s.claimed_stem='regular' "
        "LIMIT 8",
    ),
    Invariant(
        "same_platform_dup_regular",
        "WARN",
        "recording with two regular__regular rows on the SAME platform "
        "(true duplicate download)",
        "SELECT COUNT(*) FROM (SELECT recording_id, platform FROM track_audio "
        "WHERE stem='regular' AND variant='regular' "
        "GROUP BY recording_id, platform HAVING COUNT(*)>1)",
        "SELECT recording_id, platform, COUNT(*) n FROM track_audio "
        "WHERE stem='regular' AND variant='regular' "
        "GROUP BY recording_id, platform HAVING n>1 LIMIT 8",
    ),
    # ── WARN: acquisition / routing backlogs, not corruption ──────────────
    Invariant(
        "regular_claim_no_regular_audio",
        "WARN",
        "recording a set plays as regular has audio but NO regular row "
        "(only stem - acquire the full track)",
        "SELECT COUNT(*) FROM (SELECT DISTINCT s.recording_id FROM set_track_slots s "
        "WHERE s.claimed_stem='regular' "
        "AND EXISTS(SELECT 1 FROM track_audio ta WHERE ta.recording_id=s.recording_id) "
        "AND NOT EXISTS(SELECT 1 FROM track_audio ta WHERE ta.recording_id=s.recording_id "
        "AND ta.stem='regular'))",
        "SELECT DISTINCT s.recording_id, s.full_name FROM set_track_slots s "
        "WHERE s.claimed_stem='regular' "
        "AND EXISTS(SELECT 1 FROM track_audio ta WHERE ta.recording_id=s.recording_id) "
        "AND NOT EXISTS(SELECT 1 FROM track_audio ta WHERE ta.recording_id=s.recording_id "
        "AND ta.stem='regular') LIMIT 8",
    ),
    Invariant(
        "acappella_claim_unserved",
        "WARN",
        "claimed-acappella recording has audio but no acappella stem "
        "(route to Demucs vocals, or acquire)",
        "SELECT COUNT(*) FROM (SELECT DISTINCT s.recording_id FROM set_track_slots s "
        "WHERE s.claimed_stem='acappella' "
        "AND EXISTS(SELECT 1 FROM track_audio ta WHERE ta.recording_id=s.recording_id) "
        "AND NOT EXISTS(SELECT 1 FROM track_audio ta WHERE ta.recording_id=s.recording_id "
        "AND ta.stem='acappella'))",
    ),
    Invariant(
        "instrumental_claim_unserved",
        "WARN",
        "claimed-instrumental recording has audio but no instrumental stem",
        "SELECT COUNT(*) FROM (SELECT DISTINCT s.recording_id FROM set_track_slots s "
        "WHERE s.claimed_stem='instrumental' "
        "AND EXISTS(SELECT 1 FROM track_audio ta WHERE ta.recording_id=s.recording_id) "
        "AND NOT EXISTS(SELECT 1 FROM track_audio ta WHERE ta.recording_id=s.recording_id "
        "AND ta.stem='instrumental'))",
    ),
    Invariant(
        "extended_claim_unserved",
        "WARN",
        "set slot claims variant=extended but no extended audio for the recording "
        "(variant axis largely unpopulated)",
        "SELECT COUNT(*) FROM (SELECT DISTINCT s.recording_id FROM set_track_slots s "
        "WHERE s.claimed_variant='extended' "
        "AND EXISTS(SELECT 1 FROM track_audio ta WHERE ta.recording_id=s.recording_id) "
        "AND NOT EXISTS(SELECT 1 FROM track_audio ta WHERE ta.recording_id=s.recording_id "
        "AND ta.variant='extended'))",
    ),
    Invariant(
        "multi_platform_dup_regular",
        "WARN",
        "recording with regular rows across multiple platforms "
        "(benign; dedup opportunity)",
        "SELECT COUNT(*) FROM (SELECT recording_id FROM track_audio "
        "WHERE stem='regular' AND variant='regular' "
        "GROUP BY recording_id HAVING COUNT(DISTINCT platform)>1)",
    ),
)


def _scalar(conn: sqlite3.Connection, sql: str) -> int:
    return int(conn.execute(sql).fetchone()[0])


def run(db_path: str, *, show_samples: bool) -> int:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    errors = 0
    warns = 0
    print(f"corpus-integrity: {db_path}\n")
    print(f"{'sev':<6}{'count':>8}  invariant")
    print("-" * 72)
    failed_samples: list[tuple[Invariant, list]] = []
    for inv in INVARIANTS:
        n = _scalar(conn, inv.count_sql)
        if n:
            if inv.severity == "ERROR":
                errors += 1
            else:
                warns += 1
            if inv.sample_sql:
                failed_samples.append((inv, conn.execute(inv.sample_sql).fetchall()))
        flag = "" if n == 0 else ("  <== " + inv.severity)
        print(f"{inv.severity:<6}{n:>8}  {inv.title}{flag}")

    if show_samples and failed_samples:
        print("\nsamples:")
        for inv, rows in failed_samples:
            print(f"\n  [{inv.key}] {inv.title}")
            for r in rows:
                print(f"    {tuple(r)}")

    print(
        f"\n{'FAIL' if errors else 'OK'}: "
        f"{errors} ERROR-severity, {warns} WARN-severity violation classes"
    )
    conn.close()
    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--db",
        default=os.environ.get(
            "TRACKLIST_DB", "/mnt/storage/data/db/music_database.db"
        ),
        help="Path to the SQLite DB (default: canonical pi-storage path).",
    )
    ap.add_argument(
        "--no-samples",
        action="store_true",
        help="Suppress the per-violation sample rows.",
    )
    args = ap.parse_args(argv)
    return run(args.db, show_samples=not args.no_samples)


if __name__ == "__main__":
    sys.exit(main())
