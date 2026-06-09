"""MERT-only backfill: re-embed tracks that have beat grids but stale/missing 330M.

Unlike `mac_analyze_loop` / `vast_loop`, this does **not** run Demucs, beat_this,
cue-detr, or Essentia. It rsyncs the original mix, loads existing
`measure_times_json` from `track_analysis`, runs `embed_track_per_measure` with
the current adapter (`m-a-p/MERT-v1-330M`, all layers), and ships only
`track_mert_measures` (+ patches `analyzer_versions_json.mert` on canonical).

Queue predicate (default `--all`):
  - `track_analysis` row exists (measure grid required)
  - no `track_mert_measures` row with `dim = 1024` (330M per-layer dim)
  - `track_audio.path` is set

Optional `--set-ids` scopes to tracks linked from those DJ sets (same pattern as
the pilot loops, but not the default).

Run on Mac:
    tmux new -d -s mert_backfill \\
        'caffeinate -i venvs/audio/bin/python scripts/mert_backfill_loop.py \\
            --device mps 2>&1 | tee logs/mert_backfill.log'

On Vast, set REPO=/workspace/tracklist_engine and `--device cuda`.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

os.environ["TRACKLIST_DISABLE_FK"] = "1"

from analysis import persistence
from analysis.adapters import audio_io, mert_adapter
from analysis.adapters.mert_adapter import MERT_MODEL
PI_HOST = "pi-storage"
CANONICAL_DB = "/mnt/storage/data/db/music_database.db"
MERT_TARGET_DIM = 1024

SCRATCH_DIR = REPO / "_mac_scratch"
LOCAL_AUDIO = SCRATCH_DIR / "audio"
SCRATCH_DB = SCRATCH_DIR / "scratch.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("mert_backfill")


def ssh_pi(sql: str) -> str:
    full = f'sqlite3 -separator "|" {CANONICAL_DB} "{sql}"'
    r = subprocess.run(
        ["ssh", PI_HOST, full],
        capture_output=True,
        text=True,
        check=True,
    )
    return r.stdout.strip()


def _set_filter_clause(set_ids: tuple[str, ...] | None) -> str:
    if not set_ids:
        return ""
    csv = ",".join(f"'{s}'" for s in set_ids)
    return (
        "AND ta.track_id IN ("
        f"SELECT DISTINCT track_id FROM set_track_slots "
        f"WHERE set_id IN ({csv})) "
    )


def next_task(
    skip_tids: frozenset[int],
    set_ids: tuple[str, ...] | None,
) -> tuple[int, str] | None:
    skip_clause = ""
    if skip_tids:
        skip_csv = ",".join(str(t) for t in skip_tids)
        skip_clause = f"AND ta.track_audio_id NOT IN ({skip_csv}) "
    sql = (
        "SELECT ta.track_audio_id, ta.path FROM track_audio ta "
        "JOIN track_analysis tan ON tan.track_audio_id = ta.track_audio_id "
        "LEFT JOIN ("
        "  SELECT DISTINCT track_audio_id FROM track_mert_measures "
        f"  WHERE dim = {MERT_TARGET_DIM}"
        ") ok ON ok.track_audio_id = ta.track_audio_id "
        "WHERE ok.track_audio_id IS NULL "
        "AND ta.path IS NOT NULL AND length(ta.path) > 0 "
        f"{_set_filter_clause(set_ids)}"
        f"{skip_clause}"
        "ORDER BY ta.track_audio_id LIMIT 1"
    )
    out = ssh_pi(sql)
    if not out:
        return None
    parts = out.split("|", 1)
    return int(parts[0]), parts[1]


def fetch_measure_times(track_audio_id: int) -> tuple[float, ...]:
    sql = (
        "SELECT measure_times_json FROM track_analysis "
        f"WHERE track_audio_id = {track_audio_id}"
    )
    out = ssh_pi(sql)
    times = json.loads(out)
    return tuple(float(t) for t in times)


def rsync_in(remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["rsync", "-q", f"{PI_HOST}:{remote}", str(local)])


def init_scratch_db() -> None:
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    if SCRATCH_DB.exists():
        return
    schema = subprocess.check_output(
        ["ssh", PI_HOST, f"sqlite3 {CANONICAL_DB} '.schema'"],
        text=True,
    )
    cleaned: list[str] = []
    skip = False
    for line in schema.splitlines():
        if line.startswith("CREATE TABLE sqlite_sequence"):
            skip = True
            continue
        if skip:
            if line.rstrip().endswith(";"):
                skip = False
            continue
        cleaned.append(line)
    import sqlite3

    conn = sqlite3.connect(SCRATCH_DB)
    conn.executescript("\n".join(cleaned))
    conn.close()
    log.info("scratch DB initialized at %s", SCRATCH_DB)


def push_mert_rows(track_audio_id: int, mert_version: str) -> None:
    dump_script = (
        f".mode insert track_mert_measures\n"
        f"SELECT * FROM track_mert_measures WHERE track_audio_id={track_audio_id};"
    )
    dumped = subprocess.check_output(
        ["sqlite3", str(SCRATCH_DB)],
        input=dump_script,
        text=True,
    )
    versions_raw = ssh_pi(
        "SELECT analyzer_versions_json FROM track_analysis "
        f"WHERE track_audio_id = {track_audio_id}"
    )
    versions = json.loads(versions_raw) if versions_raw else {}
    versions["mert"] = mert_version
    versions_lit = json.dumps(versions).replace("'", "''")

    sql_lines = [
        ".bail on",
        "BEGIN;",
        f"DELETE FROM track_mert_measures WHERE track_audio_id={track_audio_id};",
        dumped.strip(),
        (
            f"UPDATE track_analysis SET analyzer_versions_json='{versions_lit}', "
            f"analyzed_at=CURRENT_TIMESTAMP WHERE track_audio_id={track_audio_id};"
        ),
        "COMMIT;",
    ]
    subprocess.run(
        ["ssh", PI_HOST, f"sqlite3 {CANONICAL_DB}"],
        input="\n".join(sql_lines),
        text=True,
        check=True,
    )
    import sqlite3 as _sqlite3

    conn = _sqlite3.connect(SCRATCH_DB)
    conn.execute(
        "DELETE FROM track_mert_measures WHERE track_audio_id = ?",
        (track_audio_id,),
    )
    conn.commit()
    conn.close()


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="MERT-only 330M backfill over pi-storage.")
    p.add_argument(
        "--set-ids",
        default=None,
        help="Comma-separated set_id list (pilot scope). Default: corpus-wide.",
    )
    p.add_argument(
        "--device",
        default="mps",
        help="Torch device for MERT (mps, cuda, cpu).",
    )
    p.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="Stop after N successful tracks.",
    )
    args = p.parse_args()
    set_ids = (
        tuple(s.strip() for s in args.set_ids.split(",") if s.strip())
        if args.set_ids
        else None
    )

    log.info(
        "starting MERT backfill (model=%s, dim=%d, device=%s, set_ids=%s)",
        MERT_MODEL,
        MERT_TARGET_DIM,
        args.device,
        set_ids or "ALL",
    )
    init_scratch_db()

    log.info("loading MERT…")
    t0 = time.time()
    mh = mert_adapter.load(device=args.device)
    if not mh.is_ok():
        log.error("MERT load failed: %s", mh.error.detail)
        return 1
    h = mh.value
    log.info("MERT ready in %.1fs (%s)", time.time() - t0, h.version)

    LOCAL_AUDIO.mkdir(parents=True, exist_ok=True)
    n_done = 0
    n_failed = 0
    failed_tids: set[int] = set()

    while True:
        nxt = next_task(frozenset(failed_tids), set_ids)
        if nxt is None:
            log.info("queue drained — embedded %d, failed %d", n_done, n_failed)
            return 0
        tid, remote_path = nxt
        local_audio = LOCAL_AUDIO / f"{tid}.m4a"

        try:
            log.info("[%d] pulling %s", tid, remote_path)
            rsync_in(remote_path, local_audio)

            measure_times = fetch_measure_times(tid)
            log.info("[%d] %d measure boundaries", tid, len(measure_times))

            wf_r = audio_io.load_mono(local_audio, target_sr=mert_adapter.MERT_SR)
            if not wf_r.is_ok():
                log.warning("[%d] load_mono failed: %s", tid, wf_r.error.detail)
                n_failed += 1
                failed_tids.add(tid)
                continue
            wf = wf_r.value

            t1 = time.time()
            emb_r = mert_adapter.embed_track_per_measure(
                h, wf.samples, tid, measure_times,
            )
            if not emb_r.is_ok():
                log.warning(
                    "[%d] embed failed: %s — %s",
                    tid, emb_r.error.kind, emb_r.error.detail,
                )
                n_failed += 1
                failed_tids.add(tid)
                continue
            measures = emb_r.value
            log.info(
                "[%d] embedded %d measures in %.1fs (dim=%d)",
                tid, len(measures), time.time() - t1, measures[0].dim,
            )

            p = persistence.persist_mert_measures(
                SCRATCH_DB, tid, measures, h.version,
            )
            if not p.is_ok():
                log.warning("[%d] scratch persist failed: %s", tid, p.error.detail)
                n_failed += 1
                failed_tids.add(tid)
                continue

            push_mert_rows(tid, h.version)
            n_done += 1
            log.info("[%d] pushed to canonical (n_done=%d)", tid, n_done)
            if args.max_tracks is not None and n_done >= args.max_tracks:
                log.info("hit --max-tracks=%d, exiting", args.max_tracks)
                return 0
        except subprocess.CalledProcessError as e:
            log.error("[%d] subprocess failed: %s", tid, e)
            n_failed += 1
            failed_tids.add(tid)
        finally:
            if local_audio.exists():
                local_audio.unlink()


if __name__ == "__main__":
    sys.exit(main())
