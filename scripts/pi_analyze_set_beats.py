#!/usr/bin/env python3
"""Run beat_this on a DJ-set mix (CPU-safe for hour-long mixes).

Writes ``set_analysis`` beat/downbeat/measure JSON only — no Demucs stems.
Use ``render_set_stems.py`` on Mac for set-side vocals/instrumental separately.

Usage (on pi-storage):
  venvs/audio/bin/python scripts/pi_analyze_set_beats.py --set-audio-id 5
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from analysis.adapters import beat_this_adapter
from core.result import Err

CANONICAL_DB = "/mnt/storage/data/db/music_database.db"
PI_HOST = "pi-storage"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pi_analyze_set_beats")


def _sql_lit(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _push_set_analysis(
    db_path: str,
    set_audio_id: int,
    beats_json: str,
    downbeats_json: str,
    measures_json: str,
    versions_json: str,
) -> None:
    sid = set_audio_id
    sql = "\n".join([
        ".bail on",
        "BEGIN;",
        f"DELETE FROM set_analysis WHERE set_audio_id={sid};",
        "INSERT INTO set_analysis (set_audio_id, beat_times_json, "
        "downbeat_times_json, measure_times_json, analyzer_versions_json) "
        f"VALUES ({sid}, {_sql_lit(beats_json)}, {_sql_lit(downbeats_json)}, "
        f"{_sql_lit(measures_json)}, {_sql_lit(versions_json)});",
        "COMMIT;",
    ])
    if db_path.startswith("ssh:"):
        host = db_path.split(":", 1)[1]
        subprocess.run(
            ["ssh", host, f"sqlite3 {CANONICAL_DB}"],
            input=sql,
            text=True,
            check=True,
        )
    else:
        subprocess.run(["sqlite3", db_path], input=sql, text=True, check=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-audio-id", type=int, required=True)
    p.add_argument("--mix-path", type=Path, default=None,
                   help="Local mix file (skip DB path lookup)")
    p.add_argument("--db", default=CANONICAL_DB,
                   help="SQLite path, or ssh:pi-storage to push remotely")
    args = p.parse_args()

    if args.mix_path is not None:
        mix = args.mix_path
        set_id = f"id={args.set_audio_id}"
    else:
        db_path = CANONICAL_DB if args.db.startswith("ssh:") else args.db
        row = subprocess.run(
            [
                "sqlite3", "-separator", "|", db_path,
                f"SELECT set_id, path FROM set_audio WHERE set_audio_id={args.set_audio_id}",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if not row:
            log.error("no set_audio row for id=%d", args.set_audio_id)
            return 1
        set_id, mix_path = row.split("|", 1)
        mix = Path(mix_path)
    log.info("[%d] %s beat grid on %s", args.set_audio_id, set_id, mix)

    loaded = beat_this_adapter.load(device="cpu")
    if isinstance(loaded, Err):
        log.error("beat_this load: %s", loaded.error)
        return 1
    handle = loaded.value

    predicted = beat_this_adapter.predict(handle, mix)
    if isinstance(predicted, Err):
        log.error("beat_this predict: %s — %s", predicted.error.kind, predicted.error.detail)
        return 1
    beat_times, downbeat_times = predicted.value
    bpm = beat_this_adapter.estimate_bpm(beat_times)
    measures = beat_this_adapter.measure_times(downbeat_times)

    beats_json = json.dumps(list(beat_times))
    downbeats_json = json.dumps(list(downbeat_times))
    measures_json = json.dumps(list(measures))
    versions_json = json.dumps({"beat_this": handle.version})

    sid = args.set_audio_id
    _push_set_analysis(
        args.db,
        sid,
        beats_json,
        downbeats_json,
        measures_json,
        versions_json,
    )
    log.info(
        "done: bpm=%.1f beats=%d measures=%d",
        bpm or 0,
        len(beat_times),
        len(measures),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
