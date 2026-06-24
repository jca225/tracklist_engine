"""Cache scraped tracklist cue times for info-dynamics runs.

Prefers ``set_track_slots.cue_seconds`` when materialized with enough distinct
values; falls back to parsing ``dj_set_rows`` HTML (``cue: 'NNN'``) when slots
are degenerate (many sets have cue_seconds=0 for every row until tokenizer fix).

    venvs/audio/bin/python scripts/cache_tracklist_boundaries.py --set-ids 2nvzlh2k,w1mgcjt
    venvs/audio/bin/python scripts/cache_tracklist_boundaries.py --bb9-25
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data/analysis"
CANONICAL_DB = "/mnt/storage/data/db/music_database.db"
PI_HOST = "pi-storage"

BB9_25 = (
    "1n81jy3k", "w1mgcjt", "2nvzlh2k", "1fsnxchk", "qj4v0wt", "1yl70ql1",
    "237tdqmk", "261s43wt", "zwf3n2t", "9l2wdv1", "z0mhsf1", "x5yyn4k",
    "21khc009", "2svckg31", "2vpur281", "1mpqt5wk", "2cxndfmk",
)

_FETCH_PY = r"""
import sqlite3, re, json, sys
set_id = sys.argv[1]
con = sqlite3.connect('/mnt/storage/data/db/music_database.db')
slot = [r[0] for r in con.execute(
    'SELECT cue_seconds FROM set_track_slots WHERE set_id=? '
    'AND cue_seconds IS NOT NULL ORDER BY row_index', (set_id,))]
slot_u = sorted(set(slot))
use_slots = len(slot_u) >= 5 and not (len(slot_u) == 1 and slot_u[0] == 0)
if use_slots:
    cues = slot_u
else:
    rows = con.execute(
        "SELECT raw_html FROM dj_set_rows WHERE set_id=? AND classes LIKE '%tlpTog%' "
        "ORDER BY row_index", (set_id,)).fetchall()
    cues = sorted({int(m.group(1)) for (h,) in rows
                   if (m:=re.search(r"cue:\s*'(\d+)'", h or ''))})
print(json.dumps(cues))
"""


def fetch_cues(set_id: str) -> list[int]:
    r = subprocess.run(
        ["ssh", PI_HOST, "cd ~/tracklist_engine && venvs/audio/bin/python -", set_id],
        input=_FETCH_PY,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(r.stdout.strip())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-ids", default=None, help="comma-separated set_ids")
    p.add_argument("--bb9-25", action="store_true")
    args = p.parse_args(argv)

    if args.bb9_25:
        ids = list(BB9_25)
    elif args.set_ids:
        ids = [s.strip() for s in args.set_ids.split(",") if s.strip()]
    else:
        p.error("pass --set-ids or --bb9-25")

    OUT.mkdir(parents=True, exist_ok=True)
    for sid in ids:
        cues = fetch_cues(sid)
        path = OUT / f"{sid}_tracklist_boundaries.json"
        path.write_text(json.dumps(cues))
        if cues:
            print(f"{sid}: {len(cues)} cues, {cues[0]}..{cues[-1]}s -> {path.relative_to(REPO)}")
        else:
            print(f"{sid}: EMPTY (no cue_seconds)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
