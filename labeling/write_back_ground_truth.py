"""Write manual ground-truth YAML back to pi-storage canonical DB.

Loads a ``*_ground_truth.yaml`` (see labeling/ground_truth/schema.py) and
upserts rows into ``set_ground_truth``. This is Phase 5 of the identity plan —
the seam between Ableton labeling and downstream alignment training.

Usage (on pi-storage or against a local DB copy):

    venvs/audio/bin/python -m labeling.write_back_ground_truth \\
        --db /mnt/storage/data/db/music_database.db \\
        --yaml path/to/bb12_ground_truth.yaml

Dry-run (print rows, no write):

    venvs/audio/bin/python -m labeling.write_back_ground_truth --yaml ... --dry-run
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from labeling.ground_truth.schema import load as load_gt
from core.result import Err, Ok


def write_back(db_path: Path, yaml_path: Path, *, dry_run: bool = False) -> int:
    match load_gt(yaml_path):
        case Err(e):
            print(f"YAML error: {e.detail}", file=sys.stderr)
            return 1
        case Ok(gt):
            pass
    rows: list[tuple] = []
    for t in gt.tracks:
        stem = t.claimed_stem
        seg_json = json.dumps([
            {"ref_start_s": s.ref_start_s, "ref_end_s": s.ref_end_s, "mix_start_s": s.mix_start_s}
            for s in t.ref_segments
        ]) if t.ref_segments else None
        ml_json = json.dumps(t.media_links.as_dict()) if t.media_links.any() else None
        rows.append((
            gt.set_id,
            t.label,
            t.track_id,
            stem,
            t.set_start_s,
            t.set_end_s,
            t.ref_start_s,
            t.ref_end_s,
            int(t.is_loop),
            seg_json,
            ml_json,
            gt.source,
        ))
    if dry_run:
        print(f"would upsert {len(rows)} rows into set_ground_truth for set_id={gt.set_id}")
        for r in rows[:5]:
            print(f"  {r[1]} recording={r[2]} stem={r[3]} [{r[4]:.1f}-{r[5]:.1f}s]")
        if len(rows) > 5:
            print(f"  ... +{len(rows) - 5} more")
        return 0

    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO set_ground_truth (
                set_id, label, recording_id, claimed_stem,
                set_start_s, set_end_s, ref_start_s, ref_end_s,
                is_loop, ref_segments_json, media_links_json, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    print(f"wrote {len(rows)} ground-truth rows for set_id={gt.set_id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--yaml", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if not args.yaml.is_file():
        print(f"not found: {args.yaml}", file=sys.stderr)
        return 2
    return write_back(args.db, args.yaml, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
