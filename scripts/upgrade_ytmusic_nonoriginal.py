"""Re-source non-original youtube_music references via version-gated YT Music search.

The 2026-06-12 audit found ~239 non-original tracks whose ``is_reference`` row
came from unvalidated YT Music ``hits[0]`` rescue. Re-fetching via the scraped
1001tracklists YouTube URL often re-installs the Topic-channel *original* (same
failure mode). This script instead searches by ``full_name`` and applies the
same ``search_and_pick`` version gate as ``redownload_via_ytmusic``.

Run ON pi-storage (canonical DB + yt-dlp):

    venvs/audio/bin/python -m scripts.upgrade_ytmusic_nonoriginal --dry-run
    venvs/audio/bin/python -m scripts.upgrade_ytmusic_nonoriginal
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core.result import Err
from ingest.adapters import ytmusic_adapter
from ingest.search_query import TrackSearchMeta, to_search_query_for_meta

DB = "/mnt/storage/data/db/music_database.db"
SETS = (
    "1n81jy3k",
    "w1mgcjt",
    "2nvzlh2k",
    "1fsnxchk",
    "qj4v0wt",
    "1yl70ql1",
    "237tdqmk",
    "261s43wt",
    "zwf3n2t",
    "9l2wdv1",
    "z0mhsf1",
    "x5yyn4k",
    "21khc009",
    "2svckg31",
    "2vpur281",
    "1mpqt5wk",
    "2cxndfmk",
    "pwgrrb1",
    "2ws2y6h9",
)
REASON = "ytmusic-rescue mismatch: re-fetch via version-gated YT Music search"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("upgrade_ytmusic")


@dataclass(frozen=True)
class WorkItem:
    track_id: str
    track_audio_id: int
    query: str


def worklist(con: sqlite3.Connection) -> list[WorkItem]:
    ph = ",".join("?" * len(SETS))
    rows = con.execute(
        f"""
        SELECT DISTINCT ta.track_id, ta.track_audio_id,
               tm.full_name, tm.artists_json, tm.title, tm.version,
               COALESCE(sts.claimed_stem, 'regular') AS claimed_stem
        FROM track_audio ta
        JOIN track_metadata tm ON tm.track_id = ta.track_id
        JOIN set_track_slots sts ON sts.track_id = ta.track_id
        WHERE ta.platform = 'youtube_music' AND ta.is_reference = 1
          AND sts.set_id IN ({ph})
          AND sts.claimed_version IS NOT NULL AND sts.claimed_version <> 'original'
        """,
        SETS,
    ).fetchall()
    out: list[WorkItem] = []
    seen: set[str] = set()
    for tid, taid, full_name, artists_json, title, version, claimed_stem in rows:
        if tid in seen:
            continue
        seen.add(tid)
        try:
            artists = json.loads(artists_json) if artists_json else []
        except json.JSONDecodeError:
            artists = []
        meta = TrackSearchMeta(
            full_name=full_name,
            artists_csv=", ".join(a for a in artists if a),
            title=title or "",
            version=version,
            claimed_stem=claimed_stem or "regular",
        )
        query = to_search_query_for_meta(meta)
        out.append(WorkItem(track_id=tid, track_audio_id=int(taid), query=query))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="cap N (smoke test)")
    args = p.parse_args(argv)

    con = sqlite3.connect(DB)
    work = worklist(con)
    if args.limit:
        work = work[: args.limit]
    log.info(
        "non-original youtube_music tracks to re-fetch via gated search: %d", len(work)
    )

    ok = fail = skipped = gate_refused = 0
    for i, item in enumerate(work, 1):
        cur = con.execute(
            "SELECT platform FROM track_audio WHERE track_id=? AND is_reference=1",
            (item.track_id,),
        ).fetchone()
        if not cur or cur[0] != "youtube_music":
            skipped += 1
            continue

        pick_r = ytmusic_adapter.search_and_pick(item.query, limit=8)
        if isinstance(pick_r, Err):
            gate_refused += 1
            log.warning(
                "[%d/%d] %s gate refused q=%r: %s",
                i,
                len(work),
                item.track_id,
                item.query,
                pick_r.error.detail[:160],
            )
            continue

        hit = pick_r.value
        url = f"https://www.youtube.com/watch?v={hit.video_id}"
        log.info("[%d/%d] %s -> %s (%s)", i, len(work), item.track_id, url, hit.title)
        if args.dry_run:
            continue

        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.replace_track_audio",
                "--track-id",
                item.track_id,
                "--track-audio-id",
                str(item.track_audio_id),
                "--url",
                url,
                "--reason",
                REASON,
                "--log-level",
                "WARNING",
            ],
            cwd=str(REPO),
        )
        if r.returncode == 0:
            ok += 1
        else:
            fail += 1
            log.warning("  replace failed for %s (rc=%d)", item.track_id, r.returncode)

    log.info(
        "done: ok=%d fail=%d skipped=%d gate_refused=%d dry_run=%s",
        ok,
        fail,
        skipped,
        gate_refused,
        args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
