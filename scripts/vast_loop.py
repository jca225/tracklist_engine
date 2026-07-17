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
import threading
import time
from dataclasses import replace as dc_replace
from pathlib import Path

REPO = Path("/workspace/tracklist_engine")
sys.path.insert(0, str(REPO))

# Scratch DB has the canonical schema (FKs and all) but no track_audio
# rows; FK enforcement would block every persist_analysis. Disable it on
# scratch — canonical re-enforces when we ship rows over.
os.environ["TRACKLIST_DISABLE_FK"] = "1"

from core import db as db_adapter
from analysis.pipeline import (
    analyze_track,
    compute_essentia,
    load_analyzers,
    merge_essentia,
)
from analysis import persistence
from core.models import AudioAsset
from scripts.loop_prefetch import PrefetchFailure, PrefetchItem, PrefetchSlot
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

PI_HOST = "pi-storage"  # ~/.ssh/config alias on Vast (Tailscale SOCKS5 proxy)
PI_USER = "johncabrahams"
CANONICAL_DB = "/mnt/storage/data/db/music_database.db"
PI_STEMS_ROOT = "/mnt/storage/stems"

LOCAL_AUDIO = Path("/workspace/audio")
LOCAL_STEMS = Path("/workspace/stems")
SCRATCH_DB = Path("/workspace/scratch.db")

BB_SETS = ("w1mgcjt", "2nvzlh2k", "1fsnxchk", "qj4v0wt", "1yl70ql1", "237tdqmk")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vast_loop")

# Subprocess/SSH hardening constants (A2 encoding + B1 timeouts/keepalive) and
# the failure-accounting helpers (C1/C2/C3) live in scripts/loop_hardening.py,
# shared with mac_analyze_loop and set_mert_backfill_loop.


def ssh_pi(sql: str) -> str:
    """Run a sqlite3 query on pi-storage via SSH; return stdout text."""
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


def next_task(
    skip_tids: frozenset[int] = frozenset(),
    sets: tuple[str, ...] = BB_SETS,
    shard: tuple[int, int] | None = None,
) -> tuple[int, str] | None:
    """Returns (track_audio_id, audio_path) for the next BB10-15 track that
    has no track_analysis row yet. `skip_tids` is the in-session "tried and
    failed" set — without this, a track whose audio fails to decode (or
    whose analyze_track raises mid-pipeline) re-appears on every poll
    forever, since persist_analysis is skipped on failure and the
    `track_analysis IS NULL` filter still matches. Caller maintains the
    set for the lifetime of the loop; restart re-attempts (good — could be
    a transient rsync truncation).
    """
    bb_csv = ",".join(f"'{s}'" for s in sets)
    # shard=(index, count): disjoint track partition so N boxes can work one
    # set concurrently without racing on the same next-track pick
    shard_clause = ""
    if shard is not None:
        idx, count = shard
        shard_clause = f"AND ta.track_audio_id % {count} = {idx} "
    skip_clause = ""
    if skip_tids:
        skip_csv = ",".join(str(t) for t in skip_tids)
        skip_clause = f"AND ta.track_audio_id NOT IN ({skip_csv}) "
    # Membership via set_track_slots (the canonical per-set spine), NOT
    # dj_set_track_media_links — gap rows (sided rows with no scraped media
    # links, e.g. 14 of BB11's re-sourced tail) have zero link rows and
    # would silently fall out of the loop.
    # Two queue conditions: never-analyzed rows, PLUS reference rows whose
    # separation is missing (analyzed pre-stems era — BB11's 11 stem-less
    # refs, 2026-07-09; analyze_track re-runs are idempotent DELETE+INSERT).
    sql = (
        "SELECT ta.track_audio_id, ta.path FROM track_audio ta "
        "LEFT JOIN track_analysis tan ON tan.track_audio_id=ta.track_audio_id "
        "WHERE (tan.track_audio_id IS NULL "
        "       OR (ta.is_reference=1 AND NOT EXISTS "
        "           (SELECT 1 FROM track_stems ts "
        "            WHERE ts.track_audio_id=ta.track_audio_id "
        "            AND ts.stem_name='vocals'))) "
        f"AND ta.track_id IN "
        f"(SELECT DISTINCT track_id FROM set_track_slots WHERE set_id IN ({bb_csv}) "
        "AND track_id IS NOT NULL) "
        f"{skip_clause}"
        f"{shard_clause}"
        # stem-missing REFERENCE rows first: they gate downstream consumers
        # (aligner re-runs); never-analyzed aux rows follow (62 of them sat
        # ahead of BB11's 6 refs by id order, 2026-07-09)
        "ORDER BY (CASE WHEN ta.is_reference=1 AND NOT EXISTS "
        "(SELECT 1 FROM track_stems ts2 WHERE ts2.track_audio_id=ta.track_audio_id "
        "AND ts2.stem_name='vocals') THEN 0 ELSE 1 END), "
        "ta.track_audio_id LIMIT 1"
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
    # -s / --protect-args: send the remote path verbatim instead of relying on
    # the remote login shell to strip shell-quotes. shlex.quote worked over
    # direct ssh/sshfs but FAILS over the Tailscale-userspace SOCKS ProxyCommand
    # (the single quotes survive → path resolves relative to $HOME, "No such
    # file"), which silently stalled the whole loop on space/paren filenames
    # (manual-ingest acappella/instrumental) on 2026-07-16. -s handles spaces
    # and parens on the remote side without any shell interpretation.
    # --timeout aborts if no I/O flows for RSYNC_IO_TIMEOUT s (the broken-pipe
    # hang); the subprocess timeout is a hard wall-clock backstop.
    subprocess.check_call(
        [
            "rsync",
            "-qs",
            "--timeout",
            str(RSYNC_IO_TIMEOUT),
            f"{PI_HOST}:{remote}",
            str(local),
        ],
        timeout=RSYNC_TIMEOUT,
    )


def rsync_stems_out(local_dir: Path, track_audio_id: int) -> None:
    """Push <local>/<tid>/ to pi-storage:/mnt/storage/stems/<tid>/."""
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
    """Make scratch.db's schema match canonical (one-time per Vast lifetime).
    Used as a write target for persist_analysis; we ship the new rows to
    canonical after each track and clear scratch."""
    if SCRATCH_DB.exists():
        return
    schema = subprocess.check_output(
        ["ssh", *SSH_OPTS, PI_HOST, f"sqlite3 {CANONICAL_DB} '.schema'"],
        text=True,
        encoding="utf-8",
        errors="surrogateescape",
        timeout=SSH_QUERY_TIMEOUT,
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
    # persist_analysis records track_stems.path as the local Vast scratch
    # path (/workspace/stems/<tid>/...). The actual file ends up at
    # pi-storage:/mnt/storage/stems/<tid>/... after rsync_stems_out, so the
    # row we ship to canonical needs the canonical path. Rewrite in scratch
    # before the dump so the .mode insert output already has it baked in.
    import sqlite3 as _sqlite3

    _conn = _sqlite3.connect(SCRATCH_DB)
    _conn.execute(
        "UPDATE track_stems SET path = REPLACE(path, ?, ?) WHERE track_audio_id = ?",
        (f"{LOCAL_STEMS}/", f"{PI_STEMS_ROOT}/", track_audio_id),
    )
    _conn.commit()
    _conn.close()

    tables = (
        "track_stems",
        "track_analysis",
        "track_audio_features",
        "track_mert_measures",
    )
    # Generate INSERT statements via .mode insert per table.
    # busy_timeout + BEGIN IMMEDIATE: concurrent pushers (sharded boxes) and
    # pi services share this DB — without a timeout a second writer dies with
    # "database is locked (5)" mid-transaction (hit 2026-07-09, 2-box fleet).
    # .bail on: abort the whole script on the first error (bug B5). Without it,
    # a mid-script failure (e.g. an INSERT hitting a lock after the DELETEs) still
    # falls through to COMMIT, committing the DELETEs with no re-INSERT — silent
    # data loss for this track's prior rows. .bail on makes sqlite3 exit nonzero
    # before COMMIT, so the transaction rolls back and check=True surfaces it.
    sql_lines = [".bail on", "PRAGMA busy_timeout=120000;", "BEGIN IMMEDIATE;"]
    for t in tables:
        sql_lines.append(f"DELETE FROM {t} WHERE track_audio_id={track_audio_id};")

    # Build the INSERT dump. Surrogate AUTOINCREMENT PKs (track_stem_id, ...)
    # must ship as NULL — scratch ids collide with canonical's (the Mac/pi
    # autoincrement class, a9199d1; hit again 2026-07-09 on track_stems).
    # track_audio_id-keyed tables keep their key.
    _conn = _sqlite3.connect(SCRATCH_DB)
    selects: dict[str, str] = {}
    for t in tables:
        info = list(_conn.execute(f"PRAGMA table_info({t})"))
        pk_cols = [r[1] for r in info if r[5]]
        drop_pk = len(pk_cols) == 1 and pk_cols[0] != "track_audio_id"
        selects[t] = ", ".join(
            "NULL" if (drop_pk and r[1] == pk_cols[0]) else r[1] for r in info
        )
    _conn.close()
    dump_script = "\n".join(
        f".mode insert {t}\nSELECT {selects[t]} FROM {t} "
        f"WHERE track_audio_id={track_audio_id};"
        for t in tables
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

    # Clear scratch rows for this track so scratch doesn't grow unbounded
    import sqlite3

    conn = sqlite3.connect(SCRATCH_DB)
    for t in tables:
        conn.execute(f"DELETE FROM {t} WHERE track_audio_id={track_audio_id}")
    conn.commit()
    conn.close()


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--separator",
        choices=["demucs", "uvr", "roformer"],
        default="demucs",
        help="Stem-separation backend (default: demucs).",
    )
    p.add_argument(
        "--set-ids",
        default=None,
        help="Comma-separated set_ids to scope the loop to "
        "(default: the 6 BB10-15 sets).",
    )
    p.add_argument(
        "--shard",
        default=None,
        help="i/N (e.g. 0/2): work only track_audio_id %% N == i, "
        "so N boxes can split one set without racing.",
    )
    p.add_argument(
        "--no-prefetch",
        action="store_true",
        help="disable the WS1 input-prefetch thread (serial legacy path; "
        "used as the A/B throughput baseline)",
    )
    p.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="stop after this many tracks (analyzed+failed); A/B run sizing",
    )
    p.add_argument(
        "--roformer-batch-size",
        type=int,
        default=4,
        help="chunks per RoFormer GPU forward pass (roformer only). 4 = validated "
        "corpus default (1.65x vs 1, plateaus there; output batch-invariant). "
        "1 = legacy A/B baseline.",
    )
    p.add_argument(
        "--defer-essentia",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run Essentia (CPU, ~15-18s) in the background tail so it overlaps "
        "the next track's GPU analyze (WS1.5). Default on = validated GO "
        "(A10G A/B 2026-07-15: -11%% critical path, no blocking/contention). "
        "--no-defer-essentia = legacy inline (A/B baseline).",
    )
    args = p.parse_args()
    shard: tuple[int, int] | None = None
    if args.shard:
        idx_s, count_s = args.shard.split("/", 1)
        shard = (int(idx_s), int(count_s))
        assert 0 <= shard[0] < shard[1], "--shard must be i/N with 0 <= i < N"
    active_sets: tuple[str, ...] = (
        tuple(s.strip() for s in args.set_ids.split(",") if s.strip())
        if args.set_ids
        else BB_SETS
    )

    log.info(
        "starting analyze loop on Vast (separator=%s, sets=%s)",
        args.separator,
        ",".join(active_sets),
    )
    init_scratch_db()

    log.info("loading analyzers (cuda, %s)…", args.separator)
    t0 = time.time()
    ar = load_analyzers(
        device="cuda",
        separator=args.separator,
        roformer_batch_size=args.roformer_batch_size,
    )
    if not ar.is_ok():
        log.error("load_analyzers failed: %s", ar.error)
        return 1
    a = ar.value
    log.info(
        "analyzers ready in %.1fs (with_essentia=%s)", time.time() - t0, a.with_essentia
    )

    LOCAL_AUDIO.mkdir(parents=True, exist_ok=True)
    LOCAL_STEMS.mkdir(parents=True, exist_ok=True)

    # Wipe orphan stems left behind by a previous run that was killed mid-
    # rsync. The cleanup `finally` block doesn't fire on SIGKILL/tmux-kill,
    # so files can pile up in /workspace/stems/<tid>/ and the next bg rsync
    # ends up shipping them again — slowing the first iteration after every
    # restart by ~2x. Also wipe LOCAL_AUDIO for the same reason.
    for orphan in LOCAL_STEMS.iterdir():
        if orphan.is_dir():
            shutil.rmtree(orphan, ignore_errors=True)
    for orphan in LOCAL_AUDIO.iterdir():
        if orphan.is_file():
            orphan.unlink(missing_ok=True)
    log.info("cleared orphan local stems/audio from prior run")

    # Honest tally (C3): `counts.done` is bumped by the bg tail only after the
    # canonical push confirms, so the headline can't over-report. `n_seen`
    # (dispatched-or-finally-failed) drives the --max-tracks gate. `attempts`
    # backs the transient-retry-with-backoff (C2).
    counts = Counts()
    n_seen = 0
    attempts: dict[int, int] = {}
    # In-session "tried and failed" set. Without this, any track whose audio
    # fails to decode (or any analyze_track failure mode that doesn't write a
    # track_analysis row) re-appears on every next_task() poll → infinite
    # spin on the same track. Hit this on track 149 (corrupt m4a, torchcodec
    # decode error) — generated 496 failures in 18 min before fix landed.
    failed_tids: set[int] = set()

    # --- streaming_mir WS1: two single-slot overlaps around the GPU stage ---
    # input:  PrefetchSlot pulls track N+1's audio while N analyzes.
    # output: _persist_in_bg pushes N's stems/rows while N+1 analyzes. (The
    #         pre-WS1 code started this thread at iteration end and joined it
    #         at the next iteration's TOP — milliseconds later — so the
    #         intended rsync-hiding never happened; the join now sits just
    #         before the next hand-off.)
    # Correctness: next_task picks from canonical, where an in-progress
    # track's rows haven't landed, so every prefetch pick skips
    # failed_tids ∪ {analyzing tid} ∪ {persisting tid}. A track whose PUSH
    # fails is re-picked once it leaves the in-flight set — same eventual
    # retry as before, one iteration later.
    bg: threading.Thread | None = None
    persist_tid: int | None = None

    def _finish_in_bg(
        tid: int,
        asset: AudioAsset,
        result: object,
        stem_local: Path,
        local_audio: Path,
        defer_essentia: bool,
    ) -> None:
        """Background tail: (WS1.5) Essentia on the CPU + merge, then persist
        to scratch, rsync stems, push rows — all overlapping the NEXT track's
        GPU analyze. Owns cleanup of both stem_local and local_audio.

        When defer_essentia is set, Essentia was skipped in analyze_track and
        runs here (needs local_audio, hence the deferred audio delete). When
        not set, Essentia already ran inline and `result` is complete.
        """
        try:
            essentia_s = 0.0
            if defer_essentia:
                t_e = time.monotonic()
                feat = compute_essentia(a, asset)
                result = merge_essentia(result, feat)
                essentia_s = time.monotonic() - t_e

            p_r = persistence.persist_analysis(SCRATCH_DB, result)
            if not p_r.is_ok():
                # canonical never gets track_analysis → tid re-picked later.
                # Count as a persist failure (C3) so the headline doesn't claim
                # this track as done.
                counts.persist_failed_inc()
                log.warning("[%d] (bg) persist failed: %s", tid, p_r.error.detail)
                return

            t_r = time.monotonic()
            if stem_local.exists():
                log.info("[%d] (bg) pushing stems", tid)
                rsync_stems_out(stem_local, tid)
            rsync_out_s = time.monotonic() - t_r
            log.info("[%d] (bg) pushing DB rows to canonical", tid)
            t_p = time.monotonic()
            push_track_rows(tid)
            log.info(
                "TIMING-BG tid=%d essentia_s=%.1f rsync_out_s=%.1f push_s=%.1f",
                tid,
                essentia_s,
                rsync_out_s,
                time.monotonic() - t_p,
            )
            # Confirmed persisted + pushed — only NOW is the track truly done (C3).
            counts.done_inc()
            log.info("[%d] (bg) DONE", tid)
        except subprocess.CalledProcessError as e:
            # canonical never received track_analysis (or partial), so
            # next_task re-picks this tid once it leaves the in-flight set.
            counts.persist_failed_inc()
            log.error("[%d] (bg) tail failed: %s", tid, e)
        finally:
            if stem_local.exists():
                shutil.rmtree(stem_local, ignore_errors=True)
            if local_audio.exists():
                local_audio.unlink(missing_ok=True)

    def _pick(skip: frozenset[int]) -> tuple[int, str] | None:
        return next_task(skip, sets=active_sets, shard=shard)

    def _pull(tid: int, remote_path: str) -> Path:
        local = LOCAL_AUDIO / f"{tid}.m4a"
        rsync_in(remote_path, local)
        return local

    prefetch = (
        None
        if args.no_prefetch
        else PrefetchSlot(pick=_pick, pull=_pull, hydrate=fetch_asset)
    )

    def _sync_item(
        skip: frozenset[int],
    ) -> PrefetchItem | PrefetchFailure | None:
        """Serial pick+pull+hydrate for --no-prefetch, the first iteration,
        and post-failure refills. Same shape as PrefetchSlot.take()."""
        picked = _pick(skip)
        if picked is None:
            return None
        tid, remote_path = picked
        try:
            t0 = time.monotonic()
            local = _pull(tid, remote_path)
            pull_s = time.monotonic() - t0
            asset = fetch_asset(tid, local)
        except Exception as exc:
            return PrefetchFailure(tid=tid, detail=str(exc))
        return PrefetchItem(tid=tid, local_audio=local, asset=asset, pull_s=pull_s)

    while True:
        if args.max_tracks is not None and n_seen >= args.max_tracks:
            if bg is not None:
                bg.join()
            done, failed, pf = counts.snapshot()
            code, level, msg = exit_status(done, failed, pf)
            log.log(level, "--max-tracks %d reached — %s", args.max_tracks, msg)
            return code

        # ---- obtain this iteration's item ----
        inflight = frozenset(t for t in (persist_tid,) if t is not None)
        if prefetch is None:
            # Legacy serial path (A/B baseline): join the persist tail FIRST,
            # exactly like the pre-WS1 loop, then pick+pull inline.
            if bg is not None:
                bg.join()
                bg = None
                persist_tid = None
            item = _sync_item(frozenset(failed_tids))
        elif prefetch.pending:
            item = prefetch.take()
        else:
            # First iteration / refill after a failure path.
            item = _sync_item(frozenset(failed_tids) | inflight)

        if item is None:
            # Drained *for the current skip set*. A persisting track whose
            # push failed still needs a re-pick: join the tail, then re-check
            # once with only failed_tids excluded.
            if bg is not None:
                bg.join()
                bg = None
                persist_tid = None
            item = _sync_item(frozenset(failed_tids))
            if item is None:
                done, failed, pf = counts.snapshot()
                code, level, msg = exit_status(done, failed, pf)
                log.log(level, "%s", msg)
                return code

        if isinstance(item, PrefetchFailure):
            # Input pull failures are transient (rsync/ssh timeout, broken pipe
            # over the SOCKS proxy) — retry with backoff before quarantining (C2),
            # rather than permanently skipping a track a re-pull would recover.
            if register_transient(
                item.tid, item.detail, attempts, failed_tids, counts, log
            ):
                n_seen += 1
            else:
                time.sleep(transient_backoff_seconds(attempts[item.tid]))
            continue

        tid = item.tid
        local_audio = item.local_audio
        handed_off = False

        # B2: verify the pulled file against canonical sha256 before analyzing.
        # rsync can hand back a truncated file (partial transfer / resumed run)
        # that decodes to garbage; treat a mismatch as a transient pull failure
        # (re-pull, not a permanent quarantine).
        if item.asset.sha256:
            try:
                local_sha = sha256_file(local_audio)
            except OSError as exc:
                local_sha = f"<unreadable: {exc}>"
            if local_sha != item.asset.sha256:
                if local_audio.exists():
                    local_audio.unlink(missing_ok=True)
                detail = f"sha256 mismatch (got {local_sha[:12]}…, want {item.asset.sha256[:12]}…)"
                if register_transient(tid, detail, attempts, failed_tids, counts, log):
                    n_seen += 1
                else:
                    time.sleep(transient_backoff_seconds(attempts[tid]))
                continue

        # Kick off the NEXT prefetch now — it downloads behind this track's
        # GPU work. Skip set: failures + this tid + the persisting tid.
        if prefetch is not None:
            prefetch.start(
                frozenset(failed_tids)
                | frozenset({tid})
                | frozenset(t for t in (persist_tid,) if t is not None)
            )

        try:
            log.info(
                "[%d] analyzing %s (%s)", tid, item.asset.track_id, item.asset.platform
            )
            t1 = time.monotonic()
            r = analyze_track(
                a,
                item.asset,
                stems_dir=LOCAL_STEMS,
                defer_essentia=args.defer_essentia,
            )
            analyze_s = time.monotonic() - t1
            if not r.is_ok():
                # A decode/analyze failure is a poison pill (deterministic) —
                # quarantine on first sight, no retry (C2).
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
            log.info(
                "TIMING tid=%d prefetched=%d deferess=%d pull_s=%.1f analyze_s=%.1f",
                tid,
                int(prefetch is not None),
                int(args.defer_essentia),
                item.pull_s,
                analyze_s,
            )

            # Hand off the tail (essentia + persist + rsync + push). Single
            # slot: join the PREVIOUS tail here (not at loop top) so it
            # overlapped this track's analyze. persist moved INTO the bg thread
            # so deferred Essentia (CPU) overlaps the next track's GPU work
            # (WS1.5). The bg thread now owns local_audio (Essentia needs it).
            if bg is not None:
                bg.join()
            stem_local = LOCAL_STEMS / str(tid)
            bg = threading.Thread(
                target=_finish_in_bg,
                args=(
                    tid,
                    item.asset,
                    r.value,
                    stem_local,
                    local_audio,
                    args.defer_essentia,
                ),
                daemon=False,
            )
            persist_tid = tid
            bg.start()
            handed_off = True
            # Dispatched (drives --max-tracks); the honest done-count is bumped
            # by the bg tail only once the push confirms (C3).
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
        except subprocess.CalledProcessError as e:
            log.error("[%d] subprocess failed: %s", tid, e)
            counts.failed_inc()
            failed_tids.add(tid)
            n_seen += 1
        finally:
            # On hand-off the bg thread owns BOTH local_audio (deferred
            # Essentia still needs it) and stem_local. Only clean up here when
            # we did NOT hand off (failed mid-track before the tail started).
            if not handed_off:
                if local_audio.exists():
                    local_audio.unlink()
                stem_local = LOCAL_STEMS / str(tid)
                if stem_local.exists():
                    shutil.rmtree(stem_local, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
