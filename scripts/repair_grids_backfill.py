"""Backfill grid repair + cue snapping over already-persisted analysis rows.

New analysis runs get repair/snap at derivation time (analysis/pipeline.py,
analysis/set_analysis.py, analysis/canonical_cues.py). This backfills what is
already in the DB:

  * track_analysis  — repair beat/downbeat grids, re-derive measure_times
                      (4/4: measures = downbeats), snap cue_points_json to
                      the repaired downbeat grid
  * set_analysis    — repair beat/downbeat/measure grids
  * canonical_track_cue_points — snap cues to the source audio's repaired
                      downbeat grid

DRY-RUN BY DEFAULT — prints per-row deltas and writes nothing. Pass --apply
to persist. Run against the local dev DB from repo root, or on pi-storage
against the canonical DB after `make deploy`:

    venvs/audio/bin/python scripts/repair_grids_backfill.py            # dry-run
    venvs/audio/bin/python scripts/repair_grids_backfill.py --apply
    venvs/audio/bin/python scripts/repair_grids_backfill.py --db /mnt/storage/data/db/music_database.db --apply
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.grid_repair import GRID_REPAIR_VERSION, repair_beat_grid, snap_to_grid

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data/db/music_database.db"


def _loads(blob: str | None) -> list[float]:
    return json.loads(blob) if blob else []


def _mark_versions(blob: str | None) -> str:
    versions = json.loads(blob) if blob else {}
    versions["grid_repair"] = GRID_REPAIR_VERSION
    return json.dumps(versions)


def backfill_track_analysis(conn: sqlite3.Connection, apply: bool) -> tuple[int, int]:
    touched = total = 0
    rows = conn.execute(
        """SELECT track_audio_id, beat_times_json, downbeat_times_json,
                  cue_points_json, analyzer_versions_json
           FROM track_analysis WHERE beat_times_json IS NOT NULL"""
    ).fetchall()
    for r in rows:
        total += 1
        versions = _loads_versions(r["analyzer_versions_json"])
        if versions.get("grid_repair") == GRID_REPAIR_VERSION:
            continue  # already repaired at analysis time
        beats = repair_beat_grid(_loads(r["beat_times_json"]))
        downs = repair_beat_grid(_loads(r["downbeat_times_json"]))
        cues_raw = _loads(r["cue_points_json"])
        cues = snap_to_grid(cues_raw, downs.times) if downs.times else tuple(cues_raw)
        cues_moved = sum(1 for a, b in zip(cues_raw, cues) if a != b)
        if not (beats.changed or downs.changed or cues_moved):
            continue
        touched += 1
        print(
            f"  track_audio_id={r['track_audio_id']}: beats -{beats.n_removed}/"
            f"+{beats.n_inserted}  downbeats -{downs.n_removed}/+{downs.n_inserted}"
            f"  cues moved {cues_moved}/{len(cues_raw)}"
        )
        if apply:
            conn.execute(
                """UPDATE track_analysis SET
                     beat_times_json = ?, downbeat_times_json = ?,
                     measure_times_json = ?, cue_points_json = ?,
                     analyzer_versions_json = ?
                   WHERE track_audio_id = ?""",
                (
                    json.dumps(list(beats.times)),
                    json.dumps(list(downs.times)),
                    json.dumps(list(downs.times)),  # 4/4: measures = downbeats
                    json.dumps(list(cues)),
                    _mark_versions(r["analyzer_versions_json"]),
                    r["track_audio_id"],
                ),
            )
    return touched, total


def _loads_versions(blob: str | None) -> dict:
    try:
        v = json.loads(blob) if blob else {}
        return v if isinstance(v, dict) else {}
    except json.JSONDecodeError:
        return {}


def backfill_set_analysis(conn: sqlite3.Connection, apply: bool) -> tuple[int, int]:
    touched = total = 0
    try:
        rows = conn.execute(
            """SELECT set_audio_id, beat_times_json, downbeat_times_json,
                      analyzer_versions_json
               FROM set_analysis WHERE beat_times_json IS NOT NULL"""
        ).fetchall()
    except sqlite3.OperationalError:
        return 0, 0  # table absent on this DB generation
    for r in rows:
        total += 1
        versions = _loads_versions(r["analyzer_versions_json"])
        if versions.get("grid_repair") == GRID_REPAIR_VERSION:
            continue
        beats = repair_beat_grid(_loads(r["beat_times_json"]))
        downs = repair_beat_grid(_loads(r["downbeat_times_json"]))
        if not (beats.changed or downs.changed):
            continue
        touched += 1
        print(
            f"  set_audio_id={r['set_audio_id']}: beats -{beats.n_removed}/"
            f"+{beats.n_inserted}  downbeats -{downs.n_removed}/+{downs.n_inserted}"
        )
        if apply:
            conn.execute(
                """UPDATE set_analysis SET
                     beat_times_json = ?, downbeat_times_json = ?,
                     measure_times_json = ?, analyzer_versions_json = ?
                   WHERE set_audio_id = ?""",
                (
                    json.dumps(list(beats.times)),
                    json.dumps(list(downs.times)),
                    json.dumps(list(downs.times)),
                    _mark_versions(r["analyzer_versions_json"]),
                    r["set_audio_id"],
                ),
            )
    return touched, total


def backfill_canonical_cues(conn: sqlite3.Connection, apply: bool) -> tuple[int, int]:
    touched = total = 0
    rows = conn.execute(
        """SELECT cp.track_id, cp.cue_points_json, cp.source_track_audio_id,
                  an.downbeat_times_json
           FROM canonical_track_cue_points cp
           LEFT JOIN track_analysis an
             ON an.track_audio_id = cp.source_track_audio_id"""
    ).fetchall()
    for r in rows:
        total += 1
        if not r["downbeat_times_json"]:
            continue  # no grid for the source audio — leave raw
        grid = repair_beat_grid(_loads(r["downbeat_times_json"])).times
        cues_raw = _loads(r["cue_points_json"])
        cues = snap_to_grid(cues_raw, grid)
        moved = sum(1 for a, b in zip(cues_raw, cues) if a != b)
        if not moved:
            continue
        touched += 1
        print(f"  track_id={r['track_id']}: cues moved {moved}/{len(cues_raw)}")
        if apply:
            conn.execute(
                "UPDATE canonical_track_cue_points SET cue_points_json=? WHERE track_id=?",
                (json.dumps(list(cues)), r["track_id"]),
            )
    return touched, total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write changes (default: dry-run report only)",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] {args.db}")

    print("\ntrack_analysis:")
    t1 = backfill_track_analysis(conn, args.apply)
    print("\nset_analysis:")
    t2 = backfill_set_analysis(conn, args.apply)
    print("\ncanonical_track_cue_points:")
    t3 = backfill_canonical_cues(conn, args.apply)

    if args.apply:
        conn.commit()
    conn.close()
    print(
        f"\n{mode}: track_analysis {t1[0]}/{t1[1]}  set_analysis {t2[0]}/{t2[1]}  "
        f"canonical_cues {t3[0]}/{t3[1]} rows "
        + ("written" if args.apply else "would change")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
