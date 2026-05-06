"""Vast-side BB10-15 analysis loop. Runs in a tmux session on Vast.

No FUSE needed — uses SSH+rsync over Tailscale-userspace SOCKS to talk
to pi-storage. For each unanalyzed BB10-15 track:

    1. SSH pi-storage → SELECT next (track_audio_id, path) where no analysis
    2. rsync that audio file to /workspace/audio/<tid>.m4a
    3. Run analyze_track on local file
    4. rsync stems → pi-storage:/mnt/storage/stems/<tid>/
    5. Pipe DB rows (track_stems / track_analysis / track_audio_features /
       track_mert_measures) into canonical pi-storage DB via SSH-piped SQL
    6. Delete local audio + stems, loop

Idempotent — uses ON CONFLICT clauses on canonical writes (DELETE+INSERT
per track) so re-runs don't double-insert.

Run on Vast in tmux:
    tmux new -d -s analyze \\
        '/venv/main/bin/python /workspace/tracklist_engine/scripts/vast_loop.py 2>&1 | tee /workspace/vast_loop.log'

Log progress at /workspace/vast_loop.log. tmux session 'analyze' lets
the loop survive Mac SSH disconnects.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import replace as dc_replace
from pathlib import Path

REPO = Path("/workspace/tracklist_engine")
sys.path.insert(0, str(REPO))

from audio_pipeline.adapters import db as db_adapter
from audio_pipeline.analysis.pipeline import load_analyzers, analyze_track
from audio_pipeline.models import AudioAsset

PI_HOST = "pi-storage"             # ~/.ssh/config alias on Vast (Tailscale SOCKS5 proxy)
PI_USER = "johncabrahams"
CANONICAL_DB = "/mnt/storage/data/db/music_database.db"
PI_STEMS_ROOT = "/mnt/storage/stems"

LOCAL_AUDIO = Path("/workspace/audio")
LOCAL_STEMS = Path("/workspace/stems")
SCRATCH_DB = Path("/workspace/scratch.db")

BB_SETS = ("w1mgcjt", "2nvzlh2k", "1fsnxchk", "qj4v0wt", "1yl70ql1", "237tdqmk")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vast_loop")


def ssh_pi(sql: str) -> str:
    """Run a sqlite3 query on pi-storage via SSH; return stdout text."""
    full = f'sqlite3 -separator "|" {CANONICAL_DB} "{sql}"'
    r = subprocess.run(["ssh", PI_HOST, full],
                       capture_output=True, text=True, check=True)
    return r.stdout.strip()


def next_task() -> tuple[int, str] | None:
    bb_csv = ",".join(f"'{s}'" for s in BB_SETS)
    sql = (
        "SELECT ta.track_audio_id, ta.path FROM track_audio ta "
        "LEFT JOIN track_analysis tan ON tan.track_audio_id=ta.track_audio_id "
        f"WHERE tan.track_audio_id IS NULL AND ta.track_id IN "
        f"(SELECT DISTINCT track_id FROM dj_set_track_media_links WHERE set_id IN ({bb_csv})) "
        "ORDER BY ta.track_audio_id LIMIT 1"
    )
    out = ssh_pi(sql)
    if not out:
        return None
    parts = out.split("|", 1)
    return int(parts[0]), parts[1]


def fetch_asset(track_audio_id: int, local_audio_path: Path) -> AudioAsset:
    """Load the track_audio row from canonical pi-storage DB and return
    an AudioAsset with the path overridden to the locally-rsync'd file."""
    sql = (
        "SELECT track_audio_id, track_id, platform, "
        "COALESCE(source_url,''), COALESCE(player_id,''), path, "
        "COALESCE(sha256,''), COALESCE(duration_s,0.0), "
        "COALESCE(sample_rate,0), COALESCE(codec,''), "
        "COALESCE(bitrate_kbps,0) FROM track_audio "
        f"WHERE track_audio_id={track_audio_id}"
    )
    out = ssh_pi(sql)
    parts = out.split("|")
    return AudioAsset(
        track_audio_id=int(parts[0]),
        track_id=parts[1],
        platform=parts[2],
        source_url=parts[3],
        player_id=parts[4],
        path=str(local_audio_path),
        sha256=parts[6] or None,
        duration_s=float(parts[7]) if parts[7] not in ("", "0.0") else None,
        sample_rate=int(parts[8]) if parts[8] not in ("", "0") else None,
        codec=parts[9] or None,
        bitrate_kbps=int(parts[10]) if parts[10] not in ("", "0") else None,
    )


def rsync_in(remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["rsync", "-q", f"{PI_HOST}:{remote}", str(local)])


def rsync_stems_out(local_dir: Path, track_audio_id: int) -> None:
    """Push <local>/<tid>/ to pi-storage:/mnt/storage/stems/<tid>/."""
    src = f"{local_dir}/"
    dst = f"{PI_HOST}:{PI_STEMS_ROOT}/{track_audio_id}/"
    subprocess.check_call(["ssh", PI_HOST, f"mkdir -p {PI_STEMS_ROOT}/{track_audio_id}"])
    subprocess.check_call(["rsync", "-aq", src, dst])


def init_scratch_db() -> None:
    """Make scratch.db's schema match canonical (one-time per Vast lifetime).
    Used as a write target for persist_analysis; we ship the new rows to
    canonical after each track and clear scratch."""
    if SCRATCH_DB.exists():
        return
    schema = subprocess.check_output(
        ["ssh", PI_HOST, f"sqlite3 {CANONICAL_DB} '.schema'"],
        text=True,
    )
    # SQLite reserves sqlite_sequence for AUTOINCREMENT bookkeeping; can't
    # CREATE TABLE it manually. Strip its statement out of the dump.
    cleaned = []
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


def push_track_rows(track_audio_id: int) -> None:
    """For each analysis table populated by persist_analysis, dump scratch
    rows for this track_audio_id and apply to canonical via SSH-piped SQL.
    Wraps in a transaction with DELETEs first so re-runs are idempotent."""
    tables = (
        "track_stems",
        "track_analysis",
        "track_audio_features",
        "track_mert_measures",
    )
    # Generate INSERT statements via .mode insert per table
    sql_lines = ["BEGIN;"]
    for t in tables:
        sql_lines.append(f"DELETE FROM {t} WHERE track_audio_id={track_audio_id};")

    # Build the INSERT dump
    dump_script = "\n".join(
        f".mode insert {t}\nSELECT * FROM {t} WHERE track_audio_id={track_audio_id};"
        for t in tables
    )
    dumped = subprocess.check_output(
        ["sqlite3", str(SCRATCH_DB)], input=dump_script, text=True,
    )
    sql_lines.append(dumped)
    sql_lines.append("COMMIT;")
    full_sql = "\n".join(sql_lines)

    subprocess.run(
        ["ssh", PI_HOST, f"sqlite3 {CANONICAL_DB}"],
        input=full_sql, text=True, check=True,
    )

    # Clear scratch rows for this track so scratch doesn't grow unbounded
    import sqlite3
    conn = sqlite3.connect(SCRATCH_DB)
    for t in tables:
        conn.execute(f"DELETE FROM {t} WHERE track_audio_id={track_audio_id}")
    conn.commit()
    conn.close()


def main() -> int:
    log.info("starting BB10-15 analyze loop on Vast")
    init_scratch_db()

    log.info("loading analyzers (cuda)…")
    t0 = time.time()
    ar = load_analyzers(device="cuda")
    if not ar.is_ok():
        log.error("load_analyzers failed: %s", ar.error)
        return 1
    a = ar.value
    log.info("analyzers ready in %.1fs (with_essentia=%s)", time.time() - t0, a.with_essentia)

    LOCAL_AUDIO.mkdir(parents=True, exist_ok=True)
    LOCAL_STEMS.mkdir(parents=True, exist_ok=True)

    n_done = 0
    n_failed = 0
    while True:
        nxt = next_task()
        if nxt is None:
            log.info("queue drained — analyzed %d, failed %d", n_done, n_failed)
            return 0
        tid, remote_path = nxt
        local_audio = LOCAL_AUDIO / f"{tid}.m4a"

        try:
            log.info("[%d] pulling %s", tid, remote_path)
            rsync_in(remote_path, local_audio)

            asset = fetch_asset(tid, local_audio)
            log.info("[%d] analyzing %s (%s)", tid, asset.track_id, asset.platform)
            t1 = time.time()
            r = analyze_track(a, asset, stems_dir=LOCAL_STEMS)
            if not r.is_ok():
                log.warning("[%d] analyze_track failed: %s — %s", tid, r.error.kind, r.error.detail)
                n_failed += 1
                continue
            log.info("[%d] analyzed in %.1fs", tid, time.time() - t1)

            p = db_adapter.persist_analysis(SCRATCH_DB, r.value)
            if not p.is_ok():
                log.warning("[%d] persist failed: %s", tid, p.error.detail)
                n_failed += 1
                continue

            stem_local = LOCAL_STEMS / str(tid)
            if stem_local.exists():
                log.info("[%d] pushing stems", tid)
                rsync_stems_out(stem_local, tid)

            log.info("[%d] pushing DB rows to canonical", tid)
            push_track_rows(tid)

            n_done += 1
            log.info("[%d] DONE  (total: %d ok, %d failed)", tid, n_done, n_failed)
        except subprocess.CalledProcessError as e:
            log.error("[%d] subprocess failed: %s", tid, e)
            n_failed += 1
        finally:
            if local_audio.exists():
                local_audio.unlink()
            stem_local = LOCAL_STEMS / str(tid)
            if stem_local.exists():
                shutil.rmtree(stem_local, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
