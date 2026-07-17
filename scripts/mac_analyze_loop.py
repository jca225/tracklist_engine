"""Mac-side BB10-15 analysis loop. Sibling of scripts/vast_loop.py.

Runs the full analysis pipeline (Demucs / beat_this / cue-detr / MERT /
Essentia) on Apple Silicon GPU (MPS backend) instead of CUDA. Pulls
audio from pi-storage via SSH+rsync over Tailscale (no SOCKS5 proxy
needed since Mac has direct Tailscale, unlike Vast's userspace setup).

Performance baseline: Vast 4090 was ~85 s/track post-FLAC+pipeline.
Apple Silicon GPU is ~3× slower for the heavy CUDA-bound stages, so
expect ~200–250 s/track. For the ~585 BB tracks remaining, that's
~36 hours of dedicated Mac time — runs over a long weekend.

Run via:
    tmux new -d -s analyze_mac \\
        'caffeinate -i venvs/audio/bin/python scripts/mac_analyze_loop.py \\
            2>&1 | tee logs/mac_analyze.log'

`caffeinate -i` keeps the system awake while the loop runs (lid-closed
sleep would kill the process). Energy Saver "Prevent computer sleep
when display is off" achieves the same.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import replace as dc_replace
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Scratch DB has the canonical schema; FK enforcement would block every
# persist_analysis since the scratch DB has no track_audio rows. Disable
# FK on scratch — canonical re-enforces when we ship rows over.
os.environ["TRACKLIST_DISABLE_FK"] = "1"

from core import db as db_adapter
from analysis.pipeline import load_analyzers, analyze_track
from analysis import persistence
from core.models import AudioAsset
from scripts.loop_hardening import (
    RSYNC_IO_TIMEOUT,
    RSYNC_TIMEOUT,
    SSH_OPTS,
    SSH_PUSH_TIMEOUT,
    SSH_QUERY_TIMEOUT,
    Counts,
    exit_status,
    register_transient,
    sha256_file,
    transient_backoff_seconds,
)

# Pi-storage configuration (same Tailscale alias as vast_loop, just no
# SOCKS proxy since Mac runs Tailscale natively).
PI_HOST = "pi-storage"
PI_USER = "johncabrahams"
CANONICAL_DB = "/mnt/storage/data/db/music_database.db"
PI_STEMS_ROOT = "/mnt/storage/stems"

# Local Mac scratch — under the repo so it's easy to inspect / nuke.
# Gitignored via .gitignore _mac_scratch entry (see repo root).
SCRATCH_DIR = REPO / "_mac_scratch"
LOCAL_AUDIO = SCRATCH_DIR / "audio"
LOCAL_STEMS = SCRATCH_DIR / "stems"
SCRATCH_DB = SCRATCH_DIR / "scratch.db"

BB_SETS = (
    "w1mgcjt",
    "2nvzlh2k",
    "1fsnxchk",
    "qj4v0wt",
    "1yl70ql1",
    "237tdqmk",
)

# MPS = Apple Silicon GPU. analyze_track uses this for Demucs / MERT /
# cue-detr. beat_this is CPU-light. Essentia runs in its own venv subprocess.
DEVICE = "mps"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mac_analyze")


def ssh_pi(sql: str) -> str:
    full = f'sqlite3 -separator "|" {CANONICAL_DB} "{sql}"'
    r = subprocess.run(
        ["ssh", *SSH_OPTS, PI_HOST, full],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="surrogateescape",
        check=True,
        timeout=SSH_QUERY_TIMEOUT,
    )
    return r.stdout.strip()


def _set_filter_clause(set_ids: tuple[str, ...] | None) -> str:
    ids = set_ids if set_ids else BB_SETS
    csv = ",".join(f"'{s}'" for s in ids)
    return (
        "AND ta.track_id IN ("
        f"SELECT DISTINCT track_id FROM set_track_slots WHERE set_id IN ({csv})) "
    )


def next_task(
    skip_tids: frozenset[int] = frozenset(),
    *,
    set_ids: tuple[str, ...] | None = None,
    only_reference: bool = False,
) -> tuple[int, str] | None:
    """Next track with no track_analysis row yet. `skip_tids` excludes
    in-session failures so a corrupt file doesn't spin forever."""
    skip_clause = ""
    if skip_tids:
        skip_csv = ",".join(str(t) for t in skip_tids)
        skip_clause = f"AND ta.track_audio_id NOT IN ({skip_csv}) "
    ref_clause = "AND ta.is_reference = 1 " if only_reference else ""
    sql = (
        "SELECT ta.track_audio_id, ta.path FROM track_audio ta "
        "LEFT JOIN track_analysis tan ON tan.track_audio_id=ta.track_audio_id "
        "WHERE tan.track_audio_id IS NULL "
        f"{ref_clause}"
        f"{_set_filter_clause(set_ids)}"
        f"{skip_clause}"
        "ORDER BY ta.track_audio_id LIMIT 1"
    )
    out = ssh_pi(sql)
    if not out:
        return None
    parts = out.split("|", 1)
    return int(parts[0]), parts[1]


def fetch_asset(track_audio_id: int, local_audio_path: Path) -> AudioAsset:
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
        source_url=parts[3] or "",
        player_id=parts[4] or "",
        path=str(local_audio_path),  # override with locally rsync'd path
        sha256=parts[6] or None,
        duration_s=float(parts[7]) if parts[7] else None,
        sample_rate=int(parts[8]) if parts[8] else None,
        codec=parts[9] or None,
        bitrate_kbps=int(parts[10]) if parts[10] else None,
    )


def rsync_in(remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    # shlex.quote: the remote side of an rsync path goes through the remote
    # shell — unquoted spaces/parens in filenames abort with exit 2
    # (hit 2026-07-10 on "… (Official Karaoke Instrumental) ｜ SongJam.wav").
    # --timeout aborts on an I/O stall (broken pipe / dead connection); the
    # subprocess timeout is a hard wall-clock backstop (bug B1).
    subprocess.check_call(
        [
            "rsync",
            "-q",
            "--timeout",
            str(RSYNC_IO_TIMEOUT),
            f"{PI_HOST}:{shlex.quote(remote)}",
            str(local),
        ],
        timeout=RSYNC_TIMEOUT,
    )


def rsync_stems_out(local_dir: Path, track_audio_id: int) -> None:
    """Push <local_dir>/ to pi-storage:/mnt/storage/stems/<tid>/."""
    src = f"{local_dir}/"
    dst = f"{PI_HOST}:{PI_STEMS_ROOT}/{track_audio_id}/"
    subprocess.check_call(
        ["ssh", *SSH_OPTS, PI_HOST, f"mkdir -p {PI_STEMS_ROOT}/{track_audio_id}"],
        timeout=SSH_QUERY_TIMEOUT,
    )
    subprocess.check_call(
        ["rsync", "-aq", "--timeout", str(RSYNC_IO_TIMEOUT), src, dst],
        timeout=RSYNC_TIMEOUT,
    )


def init_scratch_db() -> None:
    """Mirror canonical schema into a Mac-local sqlite scratch DB."""
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    if SCRATCH_DB.exists():
        return
    schema = subprocess.check_output(
        ["ssh", *SSH_OPTS, PI_HOST, f"sqlite3 {CANONICAL_DB} '.schema'"],
        text=True,
        encoding="utf-8",
        errors="surrogateescape",
        timeout=SSH_QUERY_TIMEOUT,
    )
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
    # Rewrite local stems path → canonical pi-storage path before shipping.
    # (Same fix as vast_loop's commit 97a89cb — persist_analysis records
    # whatever the demucs adapter wrote to, which is local.)
    import sqlite3 as _sqlite3

    _conn = _sqlite3.connect(SCRATCH_DB)
    _conn.execute(
        "UPDATE track_stems SET path = REPLACE(path, ?, ?) WHERE track_audio_id = ?",
        (f"{LOCAL_STEMS}/", f"{PI_STEMS_ROOT}/", track_audio_id),
    )
    _conn.commit()
    _conn.close()

    # track_stems has an AUTOINCREMENT PK (track_stem_id) that collides
    # across DBs — scratch's local counter assigns the same value as some
    # unrelated row already on canonical. Build the INSERTs explicitly with
    # a column list that omits track_stem_id so canonical's autoincrement
    # assigns a fresh one. The other three tables key off track_audio_id
    # (globally unique) and don't have this problem, so .mode insert is OK.
    tables_keyed = (
        "track_analysis",
        "track_audio_features",
        "track_mert_measures",
    )
    _conn = _sqlite3.connect(SCRATCH_DB)
    _conn.row_factory = _sqlite3.Row
    stems_rows = _conn.execute(
        "SELECT track_audio_id, stem_name, path, codec, created_at "
        "FROM track_stems WHERE track_audio_id = ?",
        (track_audio_id,),
    ).fetchall()
    _conn.close()

    def _sql_lit(v: object) -> str:
        if v is None:
            return "NULL"
        if isinstance(v, (int, float)):
            return str(v)
        return "'" + str(v).replace("'", "''") + "'"

    stems_inserts = "\n".join(
        "INSERT INTO track_stems (track_audio_id, stem_name, path, codec, created_at) "
        f"VALUES ({', '.join(_sql_lit(r[c]) for c in ('track_audio_id', 'stem_name', 'path', 'codec', 'created_at'))});"
        for r in stems_rows
    )

    # .bail on so the CLI aborts the transaction on the first error instead
    # of pushing through and leaving partial state (which is what bit us
    # before — track_stems INSERT failed but the rest of the rows still
    # landed because sqlite3 kept executing).
    # busy_timeout + BEGIN IMMEDIATE (mirrors vast_loop.push_track_rows): this
    # loop shares the canonical DB with pi services and other pushers. Without a
    # lock wait, a concurrent writer makes sqlite3 die with "database is locked
    # (5)" immediately, and the caller quarantines the track after ~250 s of
    # wasted MPS analysis. IMMEDIATE takes the write lock up front so the retry
    # is on acquisition, not mid-transaction.
    sql_lines = [".bail on", "PRAGMA busy_timeout=120000;", "BEGIN IMMEDIATE;"]
    sql_lines.append(f"DELETE FROM track_stems WHERE track_audio_id={track_audio_id};")
    for t in tables_keyed:
        sql_lines.append(f"DELETE FROM {t} WHERE track_audio_id={track_audio_id};")
    sql_lines.append(stems_inserts)
    dump_script = "\n".join(
        f".mode insert {t}\nSELECT * FROM {t} WHERE track_audio_id={track_audio_id};"
        for t in tables_keyed
    )
    dumped = subprocess.check_output(
        ["sqlite3", str(SCRATCH_DB)],
        input=dump_script,
        text=True,
        encoding="utf-8",
        errors="surrogateescape",
        timeout=SSH_QUERY_TIMEOUT,
    )
    sql_lines.append(dumped)
    sql_lines.append("COMMIT;")
    full_sql = "\n".join(sql_lines)
    subprocess.run(
        ["ssh", *SSH_OPTS, PI_HOST, f"sqlite3 {CANONICAL_DB}"],
        input=full_sql,
        text=True,
        encoding="utf-8",
        errors="surrogateescape",
        check=True,
        timeout=SSH_PUSH_TIMEOUT,
    )
    tables = ("track_stems",) + tables_keyed
    # Clear scratch rows for this track
    conn = _sqlite3.connect(SCRATCH_DB)
    for t in tables:
        conn.execute(f"DELETE FROM {t} WHERE track_audio_id={track_audio_id}")
    conn.commit()
    conn.close()


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="Stop after N successful tracks (smoke testing).",
    )
    p.add_argument(
        "--separator",
        choices=["demucs", "uvr", "roformer"],
        default="demucs",
        help="Stem-separation backend (default: demucs).",
    )
    p.add_argument(
        "--set-ids",
        default=None,
        help="Comma-separated set_id scope (default: all BB10–15).",
    )
    p.add_argument(
        "--only-reference",
        action="store_true",
        help="Only analyze is_reference=1 rows (post re-source canonical refs).",
    )
    args = p.parse_args()
    set_ids = (
        tuple(s.strip() for s in args.set_ids.split(",") if s.strip())
        if args.set_ids
        else None
    )

    log.info(
        "starting analyze loop on Mac (device=%s, separator=%s, max_tracks=%s, "
        "set_ids=%s, only_reference=%s)",
        DEVICE,
        args.separator,
        args.max_tracks,
        set_ids or BB_SETS,
        args.only_reference,
    )
    init_scratch_db()

    log.info("loading analyzers (%s, %s)…", DEVICE, args.separator)
    t0 = time.time()
    ar = load_analyzers(device=DEVICE, separator=args.separator)
    if not ar.is_ok():
        log.error("load_analyzers failed: %s", ar.error)
        return 1
    a = ar.value
    log.info(
        "analyzers ready in %.1fs (with_essentia=%s)", time.time() - t0, a.with_essentia
    )

    LOCAL_AUDIO.mkdir(parents=True, exist_ok=True)
    LOCAL_STEMS.mkdir(parents=True, exist_ok=True)

    # Wipe orphans from any prior interrupted run (same fix as vast_loop's
    # 670c66b commit — kill -9 / ctrl-C / lid-sleep doesn't trigger the
    # cleanup `finally`, so files pile up and double the next bg rsync).
    for orphan in LOCAL_STEMS.iterdir():
        if orphan.is_dir():
            shutil.rmtree(orphan, ignore_errors=True)
    for orphan in LOCAL_AUDIO.iterdir():
        if orphan.is_file():
            orphan.unlink(missing_ok=True)
    log.info("cleared orphan local stems/audio from prior run")

    # Honest tally (C3): counts.done bumped by the bg tail only after the push
    # confirms; n_seen (dispatched-or-finally-failed) drives --max-tracks;
    # attempts backs the transient retry-with-backoff (C2).
    counts = Counts()
    n_seen = 0
    attempts: dict[int, int] = {}
    failed_tids: set[int] = set()
    bg: threading.Thread | None = None

    def _persist_in_bg(tid: int, stem_local: Path) -> None:
        try:
            if stem_local.exists():
                log.info("[%d] (bg) pushing stems", tid)
                rsync_stems_out(stem_local, tid)
            log.info("[%d] (bg) pushing DB rows to canonical", tid)
            push_track_rows(tid)
            # Confirmed pushed — only NOW is the track truly done (C3).
            counts.done_inc()
            log.info("[%d] (bg) DONE", tid)
        except subprocess.CalledProcessError as e:
            counts.persist_failed_inc()
            log.error("[%d] (bg) push failed: %s", tid, e)
        finally:
            if stem_local.exists():
                shutil.rmtree(stem_local, ignore_errors=True)

    while True:
        if bg is not None:
            bg.join()
            bg = None

        nxt = next_task(
            frozenset(failed_tids),
            set_ids=set_ids,
            only_reference=args.only_reference,
        )
        if nxt is None:
            done, failed, pf = counts.snapshot()
            code, level, msg = exit_status(done, failed, pf)
            log.log(level, "%s", msg)
            return code
        tid, remote_path = nxt
        local_audio = LOCAL_AUDIO / f"{tid}.m4a"
        handed_off = False

        try:
            log.info("[%d] pulling %s", tid, remote_path)
            rsync_in(remote_path, local_audio)

            asset = fetch_asset(tid, local_audio)
            # B2: verify the pulled file against canonical sha256 before analyzing
            # (rsync can hand back a truncated file). Mismatch = transient pull
            # failure → retry with backoff, not a permanent quarantine. (local_audio
            # is deleted in the finally either way.)
            if asset.sha256:
                local_sha = sha256_file(local_audio)
                if local_sha != asset.sha256:
                    detail = f"sha256 mismatch (got {local_sha[:12]}…, want {asset.sha256[:12]}…)"
                    if register_transient(
                        tid, detail, attempts, failed_tids, counts, log
                    ):
                        n_seen += 1
                    else:
                        time.sleep(transient_backoff_seconds(attempts[tid]))
                    continue
            log.info("[%d] analyzing %s (%s)", tid, asset.track_id, asset.platform)
            t1 = time.time()
            r = analyze_track(a, asset, stems_dir=LOCAL_STEMS)
            if not r.is_ok():
                # Decode/analyze failure is a poison pill — quarantine on sight (C2).
                log.warning(
                    "[%d] analyze_track failed: %s — %s",
                    tid,
                    r.error.kind,
                    r.error.detail,
                )
                counts.failed_inc()
                failed_tids.add(tid)
                n_seen += 1
                continue
            log.info("[%d] analyzed in %.1fs", tid, time.time() - t1)

            p = persistence.persist_analysis(SCRATCH_DB, r.value)
            if not p.is_ok():
                log.warning("[%d] persist failed: %s", tid, p.error.detail)
                counts.failed_inc()
                failed_tids.add(tid)
                n_seen += 1
                continue

            stem_local = LOCAL_STEMS / str(tid)
            bg = threading.Thread(
                target=_persist_in_bg,
                args=(tid, stem_local),
                daemon=False,
            )
            bg.start()
            handed_off = True
            # Dispatched (drives --max-tracks); honest done-count is bumped by the
            # bg tail once the push confirms (C3).
            n_seen += 1
            done, failed, pf = counts.snapshot()
            log.info(
                "[%d] handed off (seen=%d done=%d failed=%d persist_failed=%d)",
                tid,
                n_seen,
                done,
                failed,
                pf,
            )
            if args.max_tracks is not None and n_seen >= args.max_tracks:
                if bg is not None:
                    bg.join()
                done, failed, pf = counts.snapshot()
                code, level, msg = exit_status(done, failed, pf)
                log.log(level, "hit --max-tracks=%d — %s", args.max_tracks, msg)
                return code
        except subprocess.CalledProcessError as e:
            # Pull/ssh failures reaching here are transient (timeout, broken
            # pipe) — retry with backoff before quarantining (C2).
            if register_transient(tid, str(e), attempts, failed_tids, counts, log):
                n_seen += 1
            else:
                time.sleep(transient_backoff_seconds(attempts[tid]))
        finally:
            if local_audio.exists():
                local_audio.unlink()
            if not handed_off:
                stem_local = LOCAL_STEMS / str(tid)
                if stem_local.exists():
                    shutil.rmtree(stem_local, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
