#!/usr/bin/env python3
"""Apply reviewed proposed_matches.csv to canonical pi-storage.

Two selection modes:

  default        apply rows whose decision equals --accept (human-reviewed CSV)
  --auto         apply only double-confirmed rows: decision == 'auto_accept'
                 (new matcher CSVs), or decision == --accept AND audio_score
                 clears the per-stem floor (older verified CSVs, whose decision
                 column ignored audio). Metadata-accepted rows that are NOT
                 audio-confirmed are routed to --review-out — the human queue.

Re-runs are safe: rows pass --skip-if-ingested to ingest_stem_url (sha256
dedup against canonical track_audio) unless --force is given.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = REPO / "venvs" / "audio" / "bin" / "python"


def _role_for(stem: str) -> str:
    stem = stem.lower()
    return "acappella" if "acap" in stem or stem == "vocals" else "instrumental"


def audio_confirmed(
    row: dict[str, str], hubert_floor: float, chroma_floor: float
) -> bool:
    """Does the row's Stage-3 audio score clear the floor for its stem?"""
    raw = (row.get("audio_score") or "").strip()
    try:
        score = float(raw)
    except ValueError:
        return False
    floor = (
        chroma_floor
        if _role_for(row.get("stem", "")) == "instrumental"
        else hubert_floor
    )
    return score >= floor


def row_action(
    row: dict[str, str],
    *,
    accept: str,
    auto: bool,
    hubert_floor: float,
    chroma_floor: float,
) -> str:
    """'apply' | 'review' | 'skip' for one CSV row.

    In --auto mode, 'review' collects the whole human queue: rows the matcher
    itself banded as review, plus accept-band rows lacking audio confirmation
    (the legacy verified-CSV shape whose decision ignored audio).
    """
    decision = (row.get("decision") or "").lower()
    if auto:
        if decision == "auto_accept":
            return "apply"
        if decision == "review":
            return "review"
        if decision == accept.lower():
            return (
                "apply"
                if audio_confirmed(row, hubert_floor, chroma_floor)
                else "review"
            )
        return "skip"
    return "apply" if decision == accept.lower() else "skip"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv", type=Path, required=True, help="Reviewed proposed_matches.csv"
    )
    ap.add_argument("--accept", default="accept", help="decision column value to apply")
    ap.add_argument(
        "--auto",
        action="store_true",
        help="unreviewed CSV: apply only metadata-AND-audio double-confirms; "
        "metadata-only accepts go to --review-out instead",
    )
    ap.add_argument(
        "--review-out",
        type=Path,
        default=None,
        help="with --auto: write not-auto-applied accept rows here for human review",
    )
    ap.add_argument(
        "--hubert-floor",
        type=float,
        default=0.55,
        help="acappella audio-confirm floor (matches match_stem_library)",
    )
    ap.add_argument(
        "--chroma-floor",
        type=float,
        default=0.35,
        help="instrumental audio-confirm floor (matches match_stem_library)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="re-ingest even if identical content already landed (drops the "
        "--skip-if-ingested guard)",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set-id", default=None)
    ap.add_argument("--reason", default="source:discord|identity:reviewed")
    ap.add_argument(
        "--pi-files",
        action="store_true",
        help="row paths live on pi — pass them via ingest_stem_url --pi-file",
    )
    ap.add_argument(
        "--path-map",
        default=None,
        help="OLD=NEW prefix rewrite applied to each row path (e.g. a mirror root)",
    )
    args = ap.parse_args()

    applied = 0
    review_rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    with args.csv.open() as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            action = row_action(
                row,
                accept=args.accept,
                auto=args.auto,
                hubert_floor=args.hubert_floor,
                chroma_floor=args.chroma_floor,
            )
            if action == "review":
                review_rows.append(row)
                continue
            if action == "skip":
                continue
            path = row.get("path") or row.get("file")
            recording_id = (
                row.get("cand_recording_id")
                or row.get("recording_id")
                or row.get("track_id")
            )
            if path and args.path_map and "=" in args.path_map:
                old_pfx, new_pfx = args.path_map.split("=", 1)
                if path.startswith(old_pfx):
                    path = new_pfx + path[len(old_pfx) :]
            if not path or not recording_id:
                continue
            role = _role_for(row.get("stem") or "acappella")
            cmd = [
                str(PY),
                str(REPO / "scripts" / "ingest_stem_url.py"),
                "--pi-file" if args.pi_files else "--file",
                path,
                "--track-id",
                recording_id,
                "--role",
                role,
                "--reason",
                args.reason,
            ]
            if not args.force:
                cmd.append("--skip-if-ingested")
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

    if review_rows:
        if args.review_out:
            with args.review_out.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(review_rows)
            print(f"review queue: {len(review_rows)} rows -> {args.review_out}")
        else:
            print(
                f"review queue: {len(review_rows)} accept-but-unconfirmed rows "
                "NOT applied (pass --review-out to capture them)",
                file=sys.stderr,
            )

    print(f"{'Would apply' if args.dry_run else 'Applied'} {applied} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
