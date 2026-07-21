#!/usr/bin/env python3
"""Backfill ``track_fingerprints`` with stretch-tolerant landmark hashes.

Stores kind=landmark JSON blobs (not tempo-rigid chromaprint) for reference
``track_audio`` rows. Also writes a local cache under
``workspaces/alignment_prototype/.cache/fp_index/`` for Mac alignment tools.

Run on Mac against pi-storage (rsync audio, write canonical DB via SSH):

    venvs/audio/bin/python scripts/backfill_track_fingerprints.py --dry-run
    venvs/audio/bin/python scripts/backfill_track_fingerprints.py --limit 50
    venvs/audio/bin/python scripts/backfill_track_fingerprints.py

Local dev copy:

    venvs/audio/bin/python scripts/backfill_track_fingerprints.py \\
        --db data/db/music_database.db --local-audio-root /path/to/objects
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core.db import connect
from core.result import Err, Ok
from workspaces.alignment_prototype.fp_index import (
    DEFAULT_CACHE_DIR,
    FpKey,
    compute_from_file,
    save_cached,
    upsert_db,
)

PI_HOST = "pi-storage"
CANONICAL_DB = "/mnt/storage/data/db/music_database.db"
SCRATCH = REPO / "_mac_scratch" / "fp_backfill"


@dataclass(frozen=True)
class RefRow:
    recording_id: str
    stem: str
    path: str


def ssh_sql(sql: str) -> str:
    cmd = f'sqlite3 -separator "|" {CANONICAL_DB} "{sql}"'
    r = subprocess.run(
        ["ssh", PI_HOST, cmd], capture_output=True, text=True, check=True
    )
    return r.stdout.strip()


def _rows_sql(
    *,
    only_missing: bool,
    all_rows: bool,
    set_ids: tuple[str, ...] | None,
) -> str:
    missing = (
        """
      AND NOT EXISTS (
        SELECT 1 FROM track_fingerprints tf
        WHERE tf.recording_id = ta.recording_id AND tf.stem = ta.stem
      )
    """
        if only_missing
        else ""
    )
    # Default selection is is_reference=1 rows. That flag is sparse in the
    # canonical DB (438/19.6k rows, 2026-07-15) so --all-rows widens to the
    # best row per (recording_id, stem): same ORDER BY as
    # ingest.identity_gate.lookup_reference_row, but additionally partitioned
    # per (recording_id, stem) — that per-stem partition is this script's own
    # addition, not part of identity_gate's pick rule.
    if all_rows:
        selection = """
      ta.recording_id IS NOT NULL AND ta.recording_id != ''
      AND ta.track_audio_id = (
        SELECT ta2.track_audio_id FROM track_audio ta2
        WHERE ta2.recording_id = ta.recording_id AND ta2.stem = ta.stem
        ORDER BY ta2.is_reference DESC, ta2.downloaded_at DESC LIMIT 1
      )"""
    else:
        selection = "ta.is_reference = 1"
    scope = ""
    if set_ids:
        # This string is inlined into a sqlite3 CLI command over SSH (ssh_sql),
        # so bound params aren't available on that path — escape single quotes
        # (SQL-literal doubling) so a quote in a set id can't break/inject.
        csv = ",".join("'" + s.replace("'", "''") + "'" for s in set_ids)
        # slots key recordings on recording_id; legacy rows key on track_id —
        # match either (same join as the BB gap census).
        scope = f"""
      AND EXISTS (
        SELECT 1 FROM set_track_slots s WHERE s.set_id IN ({csv})
        AND (s.recording_id = ta.recording_id OR s.recording_id = ta.track_id)
      )"""
    return f"""
    SELECT ta.recording_id, ta.stem, ta.path
    FROM track_audio ta
    WHERE {selection}
      AND ta.path IS NOT NULL AND ta.path != ''
      {scope}
      {missing}
    ORDER BY ta.recording_id, ta.stem
    """


def fetch_pi_rows(
    *,
    only_missing: bool,
    all_rows: bool = False,
    set_ids: tuple[str, ...] | None = None,
) -> tuple[RefRow, ...]:
    sql = _rows_sql(only_missing=only_missing, all_rows=all_rows, set_ids=set_ids)
    rows: list[RefRow] = []
    for line in ssh_sql(sql).splitlines():
        if not line.strip():
            continue
        rid, stem, path = line.split("|", 2)
        rows.append(RefRow(rid, stem, path))
    return tuple(rows)


def fetch_local_rows(
    db_path: Path,
    *,
    only_missing: bool,
    all_rows: bool = False,
    set_ids: tuple[str, ...] | None = None,
) -> tuple[RefRow, ...]:
    sql = _rows_sql(only_missing=only_missing, all_rows=all_rows, set_ids=set_ids)
    with connect(db_path) as conn:
        cur = conn.execute(sql)
        return tuple(
            RefRow(str(r["recording_id"]), str(r["stem"]), str(r["path"]))
            for r in cur.fetchall()
        )


def resolve_audio(
    row: RefRow, *, local_root: Path | None, scratch: Path
) -> Path | None:
    if local_root is not None:
        p = local_root / row.path.lstrip("/")
        if p.is_file():
            return p
        alt = local_root / Path(row.path).name
        return alt if alt.is_file() else None
    remote = (
        row.path if row.path.startswith("/") else f"/mnt/storage/{row.path.lstrip('/')}"
    )
    dest = scratch / row.recording_id / Path(remote).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        return dest
    # Remote paths with spaces/parens (manual-ingest filenames) go through
    # the remote shell — quote them. (--protect-args is unavailable in
    # macOS's bundled openrsync, so shlex quoting is the portable fix.)
    # Bytes mode + tolerant decode: rsync error messages can echo the
    # filename in non-UTF-8 bytes, and text=True would raise
    # UnicodeDecodeError and kill the whole run instead of skipping one row.
    import shlex

    r = subprocess.run(
        ["rsync", "-az", f"{PI_HOST}:{shlex.quote(remote)}", str(dest)],
        capture_output=True,
    )
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", errors="replace").strip()[:120]
        print(f"  rsync failed {row.recording_id}: {err}", file=sys.stderr)
        return None
    return dest


def _upsert_with_retry(fp, key: FpKey, db_path: Path, *, attempts: int = 10) -> None:
    """upsert_db with lock-retry: the canonical DB has concurrent writers
    (vast-loop pushes hold the write lock for seconds while inserting MERT
    blobs) and core.db.connect uses sqlite's default 5s timeout — a plain
    upsert dies with 'database is locked' instead of waiting its turn."""
    import sqlite3 as _sq
    import time as _t

    for i in range(attempts):
        try:
            upsert_db(fp, key, db_path)
            return
        except _sq.OperationalError as e:
            if "locked" not in str(e) or i == attempts - 1:
                raise
            _t.sleep(3.0)


def push_row_to_pi(key: FpKey, blob: bytes, duration_s: float) -> None:
    import base64
    import tempfile

    b64 = base64.b64encode(blob).decode("ascii")
    # busy_timeout=120s: the canonical DB has concurrent writers (the sharded
    # GPU boxes hold the write lock for seconds pushing MERT blobs). Without
    # it this remote insert dies with "database is locked" and the fp never
    # reaches canonical (only the local cache). Matches vast_loop's pusher.
    py = f"""
import base64, sqlite3
conn = sqlite3.connect({CANONICAL_DB!r}, timeout=120)
conn.execute('PRAGMA busy_timeout=120000')
conn.execute('PRAGMA foreign_keys=ON')
blob = base64.b64decode({b64!r})
conn.execute(
    '''INSERT INTO track_fingerprints (recording_id, stem, fingerprint, duration_s)
       VALUES (?, ?, ?, ?)
       ON CONFLICT(recording_id, stem) DO UPDATE SET
         fingerprint=excluded.fingerprint,
         duration_s=excluded.duration_s,
         created_at=CURRENT_TIMESTAMP''',
    ({key.recording_id!r}, {key.stem!r}, blob, {duration_s}),
)
conn.commit()
"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(py)
        tmp = fh.name
    remote = f"/tmp/backfill_fp_{key.recording_id[:12]}.py"
    subprocess.run(["scp", tmp, f"{PI_HOST}:{remote}"], check=True)
    subprocess.run(
        ["ssh", PI_HOST, f"~/tracklist_engine/venvs/audio/bin/python {remote}"],
        check=True,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--db", type=Path, default=None, help="Local DB instead of pi-storage"
    )
    p.add_argument("--local-audio-root", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument(
        "--recompute", action="store_true", help="Replace existing fingerprints"
    )
    p.add_argument(
        "--no-push-pi", action="store_true", help="Cache only (with --db local)"
    )
    p.add_argument(
        "--all-rows",
        action="store_true",
        help="best row per (recording_id, stem) instead of is_reference=1 only "
        "(the flag is sparse: 438/19.6k rows as of 2026-07-15)",
    )
    p.add_argument(
        "--set-ids",
        default=None,
        help="comma-separated set_id scope (slots joined on recording_id/track_id)",
    )
    args = p.parse_args(argv)

    set_ids = (
        tuple(s.strip() for s in args.set_ids.split(",") if s.strip())
        if args.set_ids
        else None
    )
    if args.db is not None:
        rows = fetch_local_rows(
            args.db,
            only_missing=not args.recompute,
            all_rows=args.all_rows,
            set_ids=set_ids,
        )
    else:
        rows = fetch_pi_rows(
            only_missing=not args.recompute,
            all_rows=args.all_rows,
            set_ids=set_ids,
        )

    if args.limit is not None:
        rows = rows[: args.limit]

    print(f"candidates={len(rows)} cache={args.cache_dir}")
    if args.dry_run:
        for row in rows[:10]:
            print(f"  {row.recording_id}/{row.stem}  {row.path}")
        if len(rows) > 10:
            print(f"  ... +{len(rows) - 10} more")
        return 0

    ok = skip = push_fail = 0
    for row in rows:
        key = FpKey(row.recording_id, row.stem)
        audio = resolve_audio(row, local_root=args.local_audio_root, scratch=SCRATCH)
        if audio is None:
            skip += 1
            continue
        # compute_from_file returns Err for soundfile failures, but the
        # audioread fallback can raise (e.g. audioread MacError -50 on a
        # truncated m4a) straight through the match — a single bad file
        # would otherwise kill the whole run. Catch broadly and skip.
        try:
            computed = compute_from_file(audio)
        except Exception as e:  # noqa: BLE001 — one bad file must not abort the run
            print(
                f"  skip {key.recording_id}/{key.stem}: decode raised {e}",
                file=sys.stderr,
            )
            skip += 1
            continue
        match computed:
            case Err(msg):
                print(f"  skip {key.recording_id}/{key.stem}: {msg}", file=sys.stderr)
                skip += 1
                continue
            case Ok(fp):
                save_cached(fp, key, args.cache_dir)
                if args.db is not None:
                    _upsert_with_retry(fp, key, args.db)
                elif not args.no_push_pi:
                    # Best-effort: a transient pi SSH hiccup must not abort the
                    # whole run — the local cache is already written above, and
                    # only_missing re-queues failed pi pushes on the next run.
                    try:
                        push_row_to_pi(key, fp.to_blob(), fp.duration_s)
                    except (subprocess.CalledProcessError, OSError) as e:
                        push_fail += 1
                        print(
                            f"  pi-push failed {key.recording_id}/{key.stem}: {e} "
                            f"(cached locally; retry later)",
                            file=sys.stderr,
                        )
                ok += 1
                if ok % 25 == 0:
                    print(f"  … {ok} indexed ({push_fail} pi-push fails)")

    print(f"done ok={ok} skip={skip} pi_push_fail={push_fail}")
    return 0 if ok or not rows else 1


if __name__ == "__main__":
    sys.exit(main())
