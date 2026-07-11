"""Pull the low-rank-study inputs from the canonical DB on pi-storage.

Read-only over ssh. Writes TSVs into eda/alignment/low_rank/data/ (gitignored):

- sets.tsv      one row per dj_set (title, date, views, ...)
- slots.tsv     one row per set_track_slot (set_id, recording_id, claimed_stem)
- dj_map.tsv    set_id -> dj slug, derived locally from data/djs/*.json job files
- coverage.txt  audio-feature coverage numbers (Representation B gate)

Usage: venvs/audio/bin/python eda/alignment/low_rank/pull_data.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = HERE / "data"
DB = "/mnt/storage/data/db/music_database.db"

SETS_SQL = (
    "SELECT set_id,"
    " replace(replace(COALESCE(title,''),char(9),' '),char(10),' '),"
    " COALESCE(date_played,''), COALESCE(views,''), COALESCE(likes,''),"
    " COALESCE(total_tracks,''), COALESCE(ided_tracks,''),"
    " replace(replace(COALESCE(play_time,''),char(9),' '),char(10),' '),"
    " replace(replace(COALESCE(styles,''),char(9),' '),char(10),' ')"
    " FROM dj_sets;"
)

SLOTS_SQL = (
    "SELECT set_id, recording_id, claimed_stem FROM set_track_slots"
    " WHERE recording_id IS NOT NULL;"
)

COVERAGE_SQL = (
    "SELECT (SELECT COUNT(*) FROM set_track_slots),"
    " (SELECT COUNT(*) FROM set_track_slots s WHERE EXISTS ("
    "   SELECT 1 FROM track_audio ta"
    "   JOIN track_audio_features f ON f.track_audio_id = ta.track_audio_id"
    "   WHERE ta.recording_id = s.recording_id AND f.bpm IS NOT NULL)),"
    " (SELECT COUNT(DISTINCT recording_id) FROM set_track_slots),"
    " (SELECT COUNT(*) FROM track_audio_features WHERE bpm IS NOT NULL);"
)


def ssh_query(sql: str, out_path: Path, header: str) -> None:
    """Run a read-only sqlite query on pi-storage, gzip over the wire."""
    cmd = [
        "ssh",
        "pi-storage",
        f'sqlite3 -readonly -tabs {DB} "{sql}" | gzip -c',
    ]
    with out_path.open("wb") as fh:
        fh.write((header + "\n").encode())
        gz = subprocess.run(cmd, check=True, capture_output=True).stdout
        gunzip = subprocess.run(
            ["gunzip", "-c"], input=gz, check=True, capture_output=True
        ).stdout
        fh.write(gunzip)
    n = sum(1 for _ in out_path.open()) - 1
    print(f"  {out_path.name}: {n} rows")


def build_dj_map(djs_dir: Path) -> dict[str, str]:
    """set_id -> dj slug from the tracked per-DJ scrape job files.

    bb*.json are Two Friends Big Bootie scrapes -> 'twofriends'.
    *_acquire.json are acquisition side-jobs, skipped.
    Sets claimed by >1 distinct DJ get '__multi__' (excluded from per-DJ
    analyses downstream but kept in the global matrix).
    """
    claims: dict[str, set[str]] = {}
    for f in sorted(djs_dir.glob("*.json")):
        slug = f.stem
        if slug.endswith("_acquire"):
            continue
        if slug.startswith("bb") and slug[2].isdigit():
            slug = "twofriends"
        try:
            rows = json.loads(f.read_text())
        except json.JSONDecodeError:
            print(f"  WARN: unparseable {f.name}, skipped", file=sys.stderr)
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            tid = row.get("tracklist_id") if isinstance(row, dict) else None
            if tid:
                claims.setdefault(tid, set()).add(slug)
    return {
        tid: (next(iter(djs)) if len(djs) == 1 else "__multi__")
        for tid, djs in claims.items()
    }


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / ".gitignore").write_text("*\n")

    print("pulling dj_sets ...")
    ssh_query(
        SETS_SQL,
        DATA / "sets.tsv",
        "set_id\ttitle\tdate_played\tviews\tlikes\ttotal_tracks"
        "\tided_tracks\tplay_time\tstyles",
    )

    print("pulling set_track_slots ...")
    ssh_query(SLOTS_SQL, DATA / "slots.tsv", "set_id\trecording_id\tclaimed_stem")

    print("pulling feature coverage ...")
    ssh_query(
        COVERAGE_SQL,
        DATA / "coverage.txt",
        "n_slots\tn_slots_with_features\tn_recordings\tn_feature_rows",
    )

    print("building dj map from data/djs/*.json ...")
    dj_map = build_dj_map(REPO / "data" / "djs")
    with (DATA / "dj_map.tsv").open("w") as fh:
        fh.write("set_id\tdj\n")
        for tid, dj in sorted(dj_map.items()):
            fh.write(f"{tid}\t{dj}\n")
    multi = sum(1 for v in dj_map.values() if v == "__multi__")
    print(f"  dj_map.tsv: {len(dj_map)} sets, {multi} multi-DJ")


if __name__ == "__main__":
    main()
