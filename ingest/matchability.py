"""Gate-1: is a replaced track_audio row *matchable* by the aligner yet?

A row is matchable once the identity channels exist for it: a landmark
fingerprint (``track_fingerprints`` by ``recording_id``+``stem``) and at least
one MERT measure (``track_mert_measures`` by ``track_audio_id``). These are
computed asynchronously by the analyze loop after a replace, so this is a
deferred check, not a synchronous gate.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from core.acquisition_case import open_worklist


def has_matchable_features(db_path: Path, track_audio_id: int) -> bool:
    conn = sqlite3.connect(db_path)
    try:
        ta = conn.execute(
            "SELECT recording_id, stem FROM track_audio WHERE track_audio_id = ?",
            (track_audio_id,),
        ).fetchone()
        if ta is None:
            return False
        recording_id, stem = ta
        fp = conn.execute(
            "SELECT 1 FROM track_fingerprints WHERE recording_id = ? AND stem = ?",
            (recording_id, stem),
        ).fetchone()
        if fp is None:
            return False
        mert = conn.execute(
            "SELECT 1 FROM track_mert_measures WHERE track_audio_id = ? LIMIT 1",
            (track_audio_id,),
        ).fetchone()
        return mert is not None
    finally:
        conn.close()


def verify_worklist(
    db_path: Path, root: str | Path = "data/acquisition_cases"
) -> list[tuple[str, bool]]:
    """For each OPEN case that already picked a winning asset, report whether that
    asset is matchable yet. Cases with no resolution asset are skipped.
    """
    out: list[tuple[str, bool]] = []
    for case in open_worklist(root=root):
        taid = case.resolution.track_audio_id if case.resolution else None
        if taid is None:
            continue
        out.append((case.case_id, has_matchable_features(db_path, int(taid))))
    return out


def _main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="report matchability of resolved cases")
    ap.add_argument(
        "--db", type=Path, default=Path("/mnt/storage/data/db/music_database.db")
    )
    ap.add_argument("--root", type=Path, default=Path("data/acquisition_cases"))
    args = ap.parse_args()
    for case_id, matchable in verify_worklist(args.db, root=args.root):
        print(f"{'MATCHABLE ' if matchable else 'PENDING   '} {case_id}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main())
