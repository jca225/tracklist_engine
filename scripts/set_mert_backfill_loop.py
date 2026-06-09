"""Set-side MERT-only backfill (P4 / 6b): embed DJ mixes on the beat grid.

Requires ``set_analysis.measure_times_json`` (run ``mac_analyze_sets.py`` first)
and ``scripts/migrate_set_mert_measures.sql`` on pi-storage.

Unlike ``mert_backfill_loop.py`` (reference tracks), this rsyncs ``set_audio``,
embeds per measure boundary, and writes ``set_mert_measures``.

Run on Mac after BB12 mix analysis:
    tmux new -d -s set_mert \\
        'caffeinate -i venvs/audio/bin/python scripts/set_mert_backfill_loop.py \\
            --set-ids 1fsnxchk --device mps 2>&1 | tee logs/set_mert_backfill.log'
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
LOCAL_SETS = SCRATCH_DIR / "sets"
SCRATCH_DB = SCRATCH_DIR / "scratch.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("set_mert_backfill")


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
    return f"AND sa.set_id IN ({csv}) "


def next_task(
    skip_ids: frozenset[int],
    set_ids: tuple[str, ...] | None,
) -> tuple[int, str, str] | None:
    skip_clause = ""
    if skip_ids:
        skip_csv = ",".join(str(i) for i in skip_ids)
        skip_clause = f"AND sa.set_audio_id NOT IN ({skip_csv}) "
    sql = (
        "SELECT sa.set_audio_id, sa.set_id, sa.path FROM set_audio sa "
        "JOIN set_analysis san ON san.set_audio_id = sa.set_audio_id "
        "LEFT JOIN ("
        "  SELECT DISTINCT set_audio_id FROM set_mert_measures "
        f"  WHERE dim = {MERT_TARGET_DIM}"
        ") ok ON ok.set_audio_id = sa.set_audio_id "
        "WHERE ok.set_audio_id IS NULL "
        "AND sa.path IS NOT NULL AND length(sa.path) > 0 "
        f"{_set_filter_clause(set_ids)}"
        f"{skip_clause}"
        "ORDER BY sa.set_audio_id LIMIT 1"
    )
    out = ssh_pi(sql)
    if not out:
        return None
    parts = out.split("|", 2)
    return int(parts[0]), parts[1], parts[2]


def fetch_measure_times(set_audio_id: int) -> tuple[float, ...]:
    sql = (
        "SELECT measure_times_json FROM set_analysis "
        f"WHERE set_audio_id = {set_audio_id}"
    )
    out = ssh_pi(sql)
    times = json.loads(out)
    return tuple(float(t) for t in times)


def rsync_in(remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["rsync", "-q", f"{PI_HOST}:{remote}", str(local)])


def init_scratch_db() -> None:
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    import sqlite3

    if not SCRATCH_DB.exists():
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
        conn = sqlite3.connect(SCRATCH_DB)
        conn.executescript("\n".join(cleaned))
        conn.close()
        log.info("scratch DB initialized at %s", SCRATCH_DB)
        return

    conn = sqlite3.connect(SCRATCH_DB)
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='set_mert_measures'"
    ).fetchone()
    if row is None:
        migrate = (REPO / "scripts/migrate_set_mert_measures.sql").read_text()
        conn.executescript(migrate)
        conn.commit()
        log.info("applied set_mert_measures migration to stale scratch DB")
    conn.close()


def push_set_mert_rows(set_audio_id: int, mert_version: str) -> None:
    dump_script = (
        f".mode insert set_mert_measures\n"
        f"SELECT * FROM set_mert_measures WHERE set_audio_id={set_audio_id};"
    )
    dumped = subprocess.check_output(
        ["sqlite3", str(SCRATCH_DB)],
        input=dump_script,
        text=True,
    )
    sql_lines = [
        ".bail on",
        "BEGIN;",
        f"DELETE FROM set_mert_measures WHERE set_audio_id={set_audio_id};",
        dumped.strip(),
        "COMMIT;",
    ]
    subprocess.run(
        ["ssh", PI_HOST, f"sqlite3 {CANONICAL_DB}"],
        input="\n".join(sql_lines),
        text=True,
        check=True,
    )
    versions_raw = ssh_pi(
        "SELECT analyzer_versions_json FROM set_analysis "
        f"WHERE set_audio_id = {set_audio_id}"
    )
    versions = json.loads(versions_raw) if versions_raw else {}
    versions["mert"] = mert_version
    versions_lit = json.dumps(versions).replace("'", "''")
    subprocess.run(
        ["ssh", PI_HOST, f"sqlite3 {CANONICAL_DB}"],
        input=(
            f"UPDATE set_analysis SET analyzer_versions_json='{versions_lit}', "
            f"analyzed_at=CURRENT_TIMESTAMP WHERE set_audio_id={set_audio_id};"
        ),
        text=True,
        check=True,
    )
    import sqlite3 as _sqlite3

    conn = _sqlite3.connect(SCRATCH_DB)
    conn.execute(
        "DELETE FROM set_mert_measures WHERE set_audio_id = ?",
        (set_audio_id,),
    )
    conn.commit()
    conn.close()


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Set-side MERT 330M backfill (P4 / 6b).")
    p.add_argument(
        "--set-ids",
        default=None,
        help="Comma-separated set_id list (e.g. 1fsnxchk). Default: all pending mixes.",
    )
    p.add_argument("--device", default="mps")
    p.add_argument("--max-sets", type=int, default=None)
    args = p.parse_args()
    set_ids = (
        tuple(s.strip() for s in args.set_ids.split(",") if s.strip())
        if args.set_ids
        else None
    )

    log.info(
        "starting set MERT backfill (model=%s, dim=%d, device=%s, set_ids=%s)",
        MERT_MODEL, MERT_TARGET_DIM, args.device, set_ids or "ALL",
    )
    init_scratch_db()

    mh = mert_adapter.load(device=args.device)
    if not mh.is_ok():
        log.error("MERT load failed: %s", mh.error.detail)
        return 1
    h = mh.value
    log.info("MERT ready (%s)", h.version)

    LOCAL_SETS.mkdir(parents=True, exist_ok=True)
    n_done = n_failed = 0
    failed: set[int] = set()

    while True:
        nxt = next_task(frozenset(failed), set_ids)
        if nxt is None:
            log.info("queue drained — embedded %d sets, failed %d", n_done, n_failed)
            return 0
        sid, set_id, remote_path = nxt
        local_audio = LOCAL_SETS / f"{sid}.m4a"
        try:
            log.info("[%d] %s pulling %s", sid, set_id, remote_path)
            rsync_in(remote_path, local_audio)
            measure_times = fetch_measure_times(sid)
            log.info("[%d] %d measure boundaries", sid, len(measure_times))

            wf_r = audio_io.load_mono(local_audio, target_sr=mert_adapter.MERT_SR)
            if not wf_r.is_ok():
                log.warning("[%d] load_mono failed: %s", sid, wf_r.error.detail)
                n_failed += 1
                failed.add(sid)
                continue
            wf = wf_r.value

            t0 = time.time()
            emb_r = mert_adapter.embed_track_per_measure(
                h, wf.samples, sid, measure_times,
            )
            if not emb_r.is_ok():
                log.warning("[%d] embed failed: %s", sid, emb_r.error.detail)
                n_failed += 1
                failed.add(sid)
                continue
            measures = emb_r.value
            log.info(
                "[%d] embedded %d measures in %.1fs",
                sid, len(measures), time.time() - t0,
            )

            p = persistence.persist_set_mert_measures(
                SCRATCH_DB, sid, measures, h.version,
            )
            if not p.is_ok():
                log.warning("[%d] scratch persist failed: %s", sid, p.error.detail)
                n_failed += 1
                failed.add(sid)
                continue

            push_set_mert_rows(sid, h.version)
            n_done += 1
            log.info("[%d] pushed to canonical (n_done=%d)", sid, n_done)
            if args.max_sets is not None and n_done >= args.max_sets:
                return 0
        except subprocess.CalledProcessError as e:
            log.error("[%d] subprocess failed: %s", sid, e)
            n_failed += 1
            failed.add(sid)
        finally:
            if local_audio.exists():
                local_audio.unlink()


if __name__ == "__main__":
    sys.exit(main())
