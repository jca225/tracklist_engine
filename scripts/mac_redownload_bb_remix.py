#!/usr/bin/env python3
"""Mac-side BB10–15 remix re-source when pi-storage yt-dlp hits bot detection.

Loads remix/rework rows from the canonical pi DB, searches YT Music with
variant-aware ``full_name`` queries, downloads on the Mac (Safari cookies +
EJS), and replaces each row on pi via ``replace_track_audio.py``.

Usage:
  venvs/audio/bin/python scripts/mac_redownload_bb_remix.py --dry-run --max-tracks 5
  venvs/audio/bin/python scripts/mac_redownload_bb_remix.py
  venvs/audio/bin/python scripts/mac_redownload_bb_remix.py --max-tracks 50
"""
from __future__ import annotations

import argparse
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
BB_SETS = (
    "w1mgcjt", "2nvzlh2k", "1fsnxchk", "qj4v0wt", "1yl70ql1", "237tdqmk",
)
BB_CSV = ",".join(f"'{s}'" for s in BB_SETS)

NODE = shutil.which("node") or "/opt/homebrew/bin/node"
YTDLP = REPO / "venvs/audio/bin/yt-dlp"
YTDLP_BASE = [
    str(YTDLP),
    "--js-runtimes", f"node:{NODE}",
    "--remote-components", "ejs:github",
    "--cookies-from-browser", "safari",
    "-f", "ba[ext=m4a]/bestaudio[ext=m4a]/bestaudio/best",
]

_log = logging.getLogger("mac_redownload_bb_remix")


@dataclass(frozen=True)
class Candidate:
    track_audio_id: int
    track_id: str
    full_name: str | None
    artists_csv: str
    title: str
    version: str | None
    set_id: str

    @property
    def query(self) -> str:
        return to_search_query(self.full_name, self.artists_csv, self.title)


def _ssh_sql(sql: str) -> str:
    cmd = f'sqlite3 -separator "|" {PI_DB} "{sql}"'
    r = subprocess.run(
        ["ssh", PI_HOST, cmd],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return r.stdout.strip()


def load_candidates() -> tuple[Candidate, ...]:
    sql = f"""
SELECT ta.track_audio_id, ta.track_id, tm.full_name, tm.artists_json, tm.title, tm.version,
  (
    SELECT m.set_id FROM dj_set_track_media_links m
    WHERE m.track_id = ta.track_id AND m.set_id IN ({BB_CSV})
    ORDER BY m.set_id LIMIT 1
  ) AS set_id
FROM track_audio ta
JOIN track_metadata tm ON tm.track_id = ta.track_id
WHERE ta.platform IN ('youtube', 'youtube_music')
  AND tm.version IN ('remix', 'rework', 'altversion', 'edit', 'bootleg')
  AND EXISTS (
    SELECT 1 FROM dj_set_track_media_links m
    WHERE m.track_id = ta.track_id AND m.set_id IN ({BB_CSV})
  )
ORDER BY ta.track_audio_id
"""
    out: list[Candidate] = []
    for line in _ssh_sql(sql).splitlines():
        parts = line.split("|")
        if len(parts) < 7:
            continue
        taid_s, tid, full_name, artists_json, title, version, set_id = parts[:7]
        try:
            artists = json.loads(artists_json) if artists_json else []
        except json.JSONDecodeError:
            artists = []
        artists_csv = ", ".join(a for a in artists if a)
        out.append(Candidate(
            track_audio_id=int(taid_s),
            track_id=tid,
            full_name=full_name or None,
            artists_csv=artists_csv,
            title=title or "",
            version=version or None,
            set_id=set_id or "",
        ))
    return tuple(out)


def _pick_video(c: Candidate) -> str | None:
    sr = ytmusic_adapter.search(c.query, limit=8)
    if isinstance(sr, Err):
        _log.warning("%s search failed: %s", c.track_id, sr.error.detail)
        return None
    for hit in sr.value:
        dur = hit.duration_s or 0
        if dur > 1200:
            continue
        _log.info("%s pick %s (%ss): %s", c.track_id, hit.video_id, dur, hit.title)
        return hit.video_id
    return None


def _download(vid: str, out: Path) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file() and out.stat().st_size > 100_000:
        return True
    cmd = [*YTDLP_BASE, "-o", str(out), f"https://www.youtube.com/watch?v={vid}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        _log.warning("download failed: %s", (r.stderr or r.stdout)[-400:])
        return False
    return out.is_file() and out.stat().st_size > 100_000


def _replace_on_pi(c: Candidate, local: Path, video_id: str, *, dry_run: bool) -> bool:
    remote_tmp = f"/tmp/mac_redownload_{c.track_id}.m4a"
    reason = (
        f"mac_rescue:bb10_15_remix|version:{c.version or 'unknown'}|"
        f"set:{c.set_id}|query:{c.query[:80]}"
    )
    parts = [
        "scripts/replace_track_audio.py",
        "--db", PI_DB,
        "--audio-root", "/mnt/storage",
        "--track-audio-id", str(c.track_audio_id),
        "--track-id", c.track_id,
        "--file", remote_tmp,
        f"--player-id={video_id}",
        "--reason", reason,
    ]
    if c.set_id:
        parts.extend(["--set-id", c.set_id])
    remote_cmd = f"cd {PI_REPO} && {PI_PY} " + " ".join(shlex.quote(p) for p in parts)

    if dry_run:
        _log.info("DRY scp %s -> %s:%s", local, PI_HOST, remote_tmp)
        _log.info("DRY %s", remote_cmd)
        return True

    scp = subprocess.run(
        ["scp", "-q", str(local), f"{PI_HOST}:{remote_tmp}"],
        capture_output=True, text=True,
    )
    if scp.returncode != 0:
        _log.error("scp failed: %s", scp.stderr)
        return False

    ssh = subprocess.run(["ssh", PI_HOST, remote_cmd], capture_output=True, text=True)
    if ssh.stdout.strip():
        _log.info(ssh.stdout.strip())
    if ssh.returncode != 0:
        _log.error("replace failed: %s", ssh.stderr[-600:])
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-tracks", type=int, default=None)
    p.add_argument("--work-dir", type=Path, default=Path("/tmp/mac_redownload_bb"))
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    candidates = load_candidates()
    if args.max_tracks is not None:
        candidates = candidates[: args.max_tracks]
    _log.info("loaded %d remix/rework candidates from pi", len(candidates))
    if not candidates:
        return 0

    ok = fail = 0
    t0 = time.monotonic()
    for i, c in enumerate(candidates, 1):
        _log.info("[%d/%d] %s taid=%d v=%s q=%r", i, len(candidates),
                  c.track_id, c.track_audio_id, c.version, c.query)
        vid = _pick_video(c)
        if not vid:
            fail += 1
            continue
        local = args.work_dir / f"{c.track_id}.m4a"
        if not args.dry_run and not _download(vid, local):
            fail += 1
            continue
        if _replace_on_pi(c, local, vid, dry_run=args.dry_run):
            ok += 1
        else:
            fail += 1
        if not args.dry_run and local.exists():
            local.unlink(missing_ok=True)

    _log.info("done in %.0fs: ok=%d fail=%d", time.monotonic() - t0, ok, fail)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
