#!/usr/bin/env python3
"""Re-source specific track_ids via YT Music when pi redownload lost rows.

Uses Mac Safari cookies + replace_track_audio on pi (same path as
mac_redownload_bb_remix.py, but accepts an explicit track_id list).

Usage:
  venvs/audio/bin/python scripts/retry_redownload_track_ids.py --from-failures-tsv
  venvs/audio/bin/python scripts/retry_redownload_track_ids.py --track-ids 12m8zb3x 8744jmp
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core.result import Err
from ingest.adapters import ytmusic_adapter
from ingest.search_query import to_search_query

PI_HOST = "pi-storage"
PI_DB = "/mnt/storage/data/db/music_database.db"
PI_REPO = "~/tracklist_engine"
PI_PY = "venvs/audio/bin/python"
FAILURES_TSV = REPO / "logs/redownload_failures_bb10_15.tsv"

NODE = shutil.which("node") or "/opt/homebrew/bin/node"
YTDLP = REPO / "venvs/audio/bin/yt-dlp"
YTDLP_BASE = [
    str(YTDLP),
    "--js-runtimes", f"node:{NODE}",
    "--remote-components", "ejs:github",
    "--cookies-from-browser", "safari",
    "-f", "ba[ext=m4a]/bestaudio[ext=m4a]/bestaudio/best",
]

_log = logging.getLogger("retry_redownload")


@dataclass(frozen=True)
class Row:
    track_id: str
    track_audio_id: int | None
    query: str


def _ssh_sql(sql: str) -> str:
    cmd = f'sqlite3 -separator "|" {PI_DB} "{sql}"'
    r = subprocess.run(["ssh", PI_HOST, cmd], capture_output=True, text=True, check=True)
    return r.stdout.strip()


def load_rows(track_ids: tuple[str, ...]) -> list[Row]:
    csv = ",".join(f"'{t}'" for t in track_ids)
    sql = f"""
SELECT tm.track_id,
       (SELECT ta.track_audio_id FROM track_audio ta
        WHERE ta.track_id = tm.track_id AND ta.is_reference = 1 LIMIT 1),
       tm.full_name, tm.artists_json, tm.title, tm.version
FROM track_metadata tm
WHERE tm.track_id IN ({csv})
ORDER BY tm.track_id
"""
    out = _ssh_sql(sql)
    rows: list[Row] = []
    for line in out.splitlines():
        tid, taid_s, full_name, artists_json, title, version = line.split("|", 5)
        try:
            artists = json.loads(artists_json) if artists_json else []
        except json.JSONDecodeError:
            artists = []
        artists_csv = ", ".join(a for a in artists if a)
        q = to_search_query(full_name or None, artists_csv, title or None)
        rows.append(Row(
            track_id=tid,
            track_audio_id=int(taid_s) if taid_s else None,
            query=q or tid,
        ))
    return rows


def _pick_video(row: Row) -> str | None:
    sr = ytmusic_adapter.search(row.query, limit=8)
    if isinstance(sr, Err):
        _log.warning("%s search failed: %s", row.track_id, sr.error.detail)
        return None
    for hit in sr.value:
        dur = hit.duration_s or 0
        if dur > 1200:
            continue
        _log.info("%s pick %s (%ss): %s", row.track_id, hit.video_id, dur, hit.title)
        return hit.video_id
    return None


def _download(video_id: str, local: Path) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [*YTDLP_BASE, "-o", str(local), url]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        _log.error("yt-dlp failed: %s", (r.stderr or r.stdout)[-400:])
        return False
    return local.is_file() and local.stat().st_size > 100_000


def _replace_on_pi(row: Row, local: Path, video_id: str) -> bool:
    remote_tmp = f"/tmp/retry_redownload_{row.track_id}.m4a"
    subprocess.check_call(["scp", str(local), f"{PI_HOST}:{remote_tmp}"])
    parts = [
        "scripts/replace_track_audio.py",
        "--db", PI_DB,
        "--audio-root", "/mnt/storage",
        "--track-id", row.track_id,
        "--file", remote_tmp,
        f"--player-id={video_id}",
        "--reason", "retry_redownload_track_ids",
    ]
    if row.track_audio_id is not None:
        parts.extend(["--track-audio-id", str(row.track_audio_id)])
    remote_cmd = f"cd {PI_REPO} && {PI_PY} " + " ".join(shlex.quote(p) for p in parts)
    ssh = subprocess.run(["ssh", PI_HOST, remote_cmd], capture_output=True, text=True)
    subprocess.run(["ssh", PI_HOST, f"rm -f {remote_tmp}"], check=False)
    if ssh.returncode != 0:
        _log.error("%s replace failed: %s", row.track_id, ssh.stderr[-600:])
        return False
    return True


def _track_ids_from_tsv(path: Path) -> tuple[str, ...]:
    ids: set[str] = set()
    with path.open() as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row.get("track_id"):
                ids.add(row["track_id"])
    return tuple(sorted(ids))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--track-ids", nargs="+", default=None)
    p.add_argument("--from-failures-tsv", action="store_true")
    p.add_argument("--work-dir", type=Path, default=Path("/tmp/retry_redownload"))
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.from_failures_tsv:
        track_ids = _track_ids_from_tsv(FAILURES_TSV)
    elif args.track_ids:
        track_ids = tuple(args.track_ids)
    else:
        p.error("pass --track-ids or --from-failures-tsv")

    rows = load_rows(track_ids)
    _log.info("retrying %d track_ids", len(rows))
    ok = fail = 0
    t0 = time.monotonic()
    for i, row in enumerate(rows, 1):
        _log.info("[%d/%d] %s q=%r", i, len(rows), row.track_id, row.query)
        vid = _pick_video(row)
        if not vid:
            fail += 1
            continue
        local = args.work_dir / f"{row.track_id}.m4a"
        if not _download(vid, local):
            fail += 1
            continue
        if _replace_on_pi(row, local, vid):
            ok += 1
        else:
            fail += 1
        local.unlink(missing_ok=True)

    _log.info("done in %.0fs: ok=%d fail=%d", time.monotonic() - t0, ok, fail)
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
