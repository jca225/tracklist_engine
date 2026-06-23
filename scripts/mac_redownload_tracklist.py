#!/usr/bin/env python3
"""Mac-side list-driven re-source for an explicit set of track_ids.

Generalises ``mac_redownload_bb_remix.py`` (which only handled remix/rework
rows) to an arbitrary track_id list, and auto-routes each track to **replace**
(an existing ``track_audio`` row exists) or **add** (no row — the MISSING case)
based on canonical DB state. Used to make a set alignment-ready when the old
download path left gaps.

Pipeline per track (Mac yt-dlp works around pi-storage's bot detection):
  metadata from pi -> YT Music search by ``full_name`` -> Mac yt-dlp download
  (Safari cookies + EJS) -> scp to pi -> ``replace_track_audio.py`` (ledger-logged).

``replace_track_audio.py`` adds a fresh row when ``--track-audio-id`` is omitted
(ledger ``action=add``) and replaces destructively when it is supplied.

Usage:
  venvs/audio/bin/python scripts/mac_redownload_tracklist.py \\
      --set-id 2nvzlh2k --track-ids-file /tmp/bb11_missing.txt --dry-run
  venvs/audio/bin/python scripts/mac_redownload_tracklist.py \\
      --set-id 2nvzlh2k --track-ids 26b4gz6f,2m5wh0t5,...
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
from ingest.search_query import to_search_query_for_claim

PI_HOST = "pi-storage"
PI_DB = "/mnt/storage/data/db/music_database.db"
PI_REPO = "~/tracklist_engine"
PI_PY = "venvs/audio/bin/python"

NODE = shutil.which("node") or "/opt/homebrew/bin/node"
YTDLP = REPO / "venvs/audio/bin/yt-dlp"
YTDLP_BASE = [
    str(YTDLP),
    "--js-runtimes",
    f"node:{NODE}",
    "--remote-components",
    "ejs:github",
    "--cookies-from-browser",
    "safari",
    "-f",
    "ba[ext=m4a]/bestaudio[ext=m4a]/bestaudio/best",
]

_log = logging.getLogger("mac_redownload_tracklist")


@dataclass(frozen=True)
class Candidate:
    track_id: str
    full_name: str | None
    artists_csv: str
    title: str
    version: str | None
    claimed_stem: str
    set_id: str
    # Current reference row, if any. None => ADD a fresh row; else REPLACE it.
    ref_track_audio_id: int | None
    ref_platform: str | None

    @property
    def query(self) -> str:
        return to_search_query_for_claim(
            full_name=self.full_name,
            artists_csv=self.artists_csv,
            title=self.title,
            claimed_stem=self.claimed_stem,
            version=self.version,
        )

    @property
    def mode(self) -> str:
        return "replace" if self.ref_track_audio_id is not None else "add"


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


def load_candidates(track_ids: tuple[str, ...], set_id: str) -> tuple[Candidate, ...]:
    """Pull metadata + current reference row for each track_id, in input order."""
    csv = ",".join(f"'{t}'" for t in track_ids)
    sql = f"""
SELECT tm.track_id, tm.full_name, tm.artists_json, tm.title, tm.version,
  COALESCE(
    (SELECT sts.claimed_stem FROM set_track_slots sts
       WHERE sts.set_id = '{set_id}' AND sts.track_id = tm.track_id LIMIT 1),
    'regular'
  ) AS claimed_stem,
  (SELECT ta.track_audio_id FROM track_audio ta
     WHERE ta.track_id = tm.track_id
     ORDER BY ta.is_reference DESC, ta.track_audio_id DESC LIMIT 1) AS ref_taid,
  (SELECT ta.platform FROM track_audio ta
     WHERE ta.track_id = tm.track_id
     ORDER BY ta.is_reference DESC, ta.track_audio_id DESC LIMIT 1) AS ref_plat
FROM track_metadata tm
WHERE tm.track_id IN ({csv})
"""
    by_id: dict[str, Candidate] = {}
    for line in _ssh_sql(sql).splitlines():
        parts = line.split("|")
        if len(parts) < 8:
            continue
        (
            tid,
            full_name,
            artists_json,
            title,
            version,
            claimed_stem,
            ref_taid,
            ref_plat,
        ) = parts[:8]
        try:
            artists = json.loads(artists_json) if artists_json else []
        except json.JSONDecodeError:
            artists = []
        by_id[tid] = Candidate(
            track_id=tid,
            full_name=full_name or None,
            artists_csv=", ".join(a for a in artists if a),
            title=title or "",
            version=version or None,
            claimed_stem=claimed_stem or "regular",
            set_id=set_id,
            ref_track_audio_id=int(ref_taid) if ref_taid else None,
            ref_platform=ref_plat or None,
        )
    # preserve input order; warn on any track_id with no metadata row
    out: list[Candidate] = []
    for tid in track_ids:
        if tid in by_id:
            out.append(by_id[tid])
        else:
            _log.warning("%s: no track_metadata row — skipping", tid)
    return tuple(out)


def _pick_video(c: Candidate) -> str | None:
    sr = ytmusic_adapter.search_and_pick(c.query, limit=8)
    if isinstance(sr, Err):
        _log.warning(
            "%s pick refused/failed: %s — %s",
            c.track_id,
            sr.error.kind,
            sr.error.detail,
        )
        return None
    hit = sr.value
    _log.info(
        "%s pick %s (%ss): %s", c.track_id, hit.video_id, hit.duration_s, hit.title
    )
    return hit.video_id


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


def _apply_on_pi(c: Candidate, local: Path, video_id: str, *, dry_run: bool) -> bool:
    remote_tmp = f"/tmp/mac_redownload_{c.track_id}.m4a"
    reason = (
        f"mac_rescue:tracklist|mode:{c.mode}|version:{c.version or 'original'}|"
        f"set:{c.set_id}|query:{c.query[:80]}"
    )
    parts = [
        "scripts/replace_track_audio.py",
        "--db",
        PI_DB,
        "--audio-root",
        "/mnt/storage",
        "--track-id",
        c.track_id,
        "--file",
        remote_tmp,
        f"--player-id={video_id}",
        "--set-id",
        c.set_id,
        "--reason",
        reason,
    ]
    if c.ref_track_audio_id is not None:  # replace mode (destructive)
        parts.extend(["--track-audio-id", str(c.ref_track_audio_id)])
    remote_cmd = f"cd {PI_REPO} && {PI_PY} " + " ".join(shlex.quote(p) for p in parts)

    if dry_run:
        _log.info("DRY [%s] scp %s -> %s:%s", c.mode, local.name, PI_HOST, remote_tmp)
        _log.info("DRY %s", remote_cmd)
        return True

    scp = subprocess.run(
        ["scp", "-q", str(local), f"{PI_HOST}:{remote_tmp}"],
        capture_output=True,
        text=True,
    )
    if scp.returncode != 0:
        _log.error("scp failed: %s", scp.stderr)
        return False
    ssh = subprocess.run(["ssh", PI_HOST, remote_cmd], capture_output=True, text=True)
    if ssh.stdout.strip():
        _log.info(ssh.stdout.strip())
    if ssh.returncode != 0:
        _log.error("replace/add failed: %s", ssh.stderr[-600:])
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", required=True, help="set_id for ledger provenance")
    p.add_argument("--track-ids", default=None, help="comma-separated track_ids")
    p.add_argument(
        "--track-ids-file",
        type=Path,
        default=None,
        help="file with one track_id per line (or comma-separated)",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-tracks", type=int, default=None)
    p.add_argument("--work-dir", type=Path, default=Path("/tmp/mac_redownload_tl"))
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    raw = ""
    if args.track_ids_file:
        raw = args.track_ids_file.read_text()
    if args.track_ids:
        raw = raw + "," + args.track_ids
    ids = tuple(t.strip() for t in raw.replace("\n", ",").split(",") if t.strip())
    if not ids:
        _log.error("no track_ids provided")
        return 2

    candidates = load_candidates(ids, args.set_id)
    if args.max_tracks is not None:
        candidates = candidates[: args.max_tracks]
    n_add = sum(1 for c in candidates if c.mode == "add")
    _log.info(
        "loaded %d candidates (%d add / %d replace)",
        len(candidates),
        n_add,
        len(candidates) - n_add,
    )

    ok = fail = 0
    t0 = time.monotonic()
    for i, c in enumerate(candidates, 1):
        _log.info(
            "[%d/%d] %s mode=%s v=%s q=%r",
            i,
            len(candidates),
            c.track_id,
            c.mode,
            c.version,
            c.query,
        )
        vid = _pick_video(c)
        if not vid:
            fail += 1
            continue
        local = args.work_dir / f"{c.track_id}.m4a"
        if not args.dry_run and not _download(vid, local):
            fail += 1
            continue
        if _apply_on_pi(c, local, vid, dry_run=args.dry_run):
            ok += 1
        else:
            fail += 1
        if not args.dry_run and local.exists():
            local.unlink(missing_ok=True)

    _log.info("done in %.0fs: ok=%d fail=%d", time.monotonic() - t0, ok, fail)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
