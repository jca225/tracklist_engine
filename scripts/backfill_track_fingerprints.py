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


def fetch_pi_rows(*, only_missing: bool) -> tuple[RefRow, ...]:
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
    sql = f"""
    SELECT ta.recording_id, ta.stem, ta.path
    FROM track_audio ta
    WHERE ta.is_reference = 1
      AND ta.path IS NOT NULL AND ta.path != ''
      {missing}
    ORDER BY ta.recording_id, ta.stem
    """
    rows: list[RefRow] = []
    for line in ssh_sql(sql).splitlines():
        if not line.strip():
            continue
        rid, stem, path = line.split("|", 2)
        rows.append(RefRow(rid, stem, path))
    return tuple(rows)


def fetch_local_rows(db_path: Path, *, only_missing: bool) -> tuple[RefRow, ...]:
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
    sql = f"""
    SELECT ta.recording_id, ta.stem, ta.path
    FROM track_audio ta
    WHERE ta.is_reference = 1
      AND ta.path IS NOT NULL AND ta.path != ''
      {missing}
    ORDER BY ta.recording_id, ta.stem
    """
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
    r = subprocess.run(
        ["rsync", "-az", f"{PI_HOST}:{remote}", str(dest)],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(
            f"  rsync failed {row.recording_id}: {r.stderr.strip()[:120]}",
            file=sys.stderr,
        )
        return None
    return dest


def push_row_to_pi(key: FpKey, blob: bytes, duration_s: float) -> None:
    import base64
    import tempfile

    b64 = base64.b64encode(blob).decode("ascii")
    py = f"""
import base64, sqlite3
conn = sqlite3.connect({CANONICAL_DB!r})
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
    args = p.parse_args(argv)

    if args.db is not None:
        rows = fetch_local_rows(args.db, only_missing=not args.recompute)
    else:
        rows = fetch_pi_rows(only_missing=not args.recompute)

    if args.limit is not None:
        rows = rows[: args.limit]

    print(f"candidates={len(rows)} cache={args.cache_dir}")
    if args.dry_run:
        for row in rows[:10]:
            print(f"  {row.recording_id}/{row.stem}  {row.path}")
        if len(rows) > 10:
            print(f"  ... +{len(rows) - 10} more")
        return 0

    ok = skip = 0
    for row in rows:
        key = FpKey(row.recording_id, row.stem)
        audio = resolve_audio(row, local_root=args.local_audio_root, scratch=SCRATCH)
        if audio is None:
            skip += 1
            continue
        match compute_from_file(audio):
            case Err(msg):
                print(f"  skip {key.recording_id}/{key.stem}: {msg}", file=sys.stderr)
                skip += 1
                continue
            case Ok(fp):
                save_cached(fp, key, args.cache_dir)
                if args.db is not None:
                    upsert_db(fp, key, args.db)
                elif not args.no_push_pi:
                    push_row_to_pi(key, fp.to_blob(), fp.duration_s)
                ok += 1
                if ok % 25 == 0:
                    print(f"  … {ok} indexed")

    print(f"done ok={ok} skip={skip}")
    return 0 if ok or not rows else 1


if __name__ == "__main__":
    sys.exit(main())
