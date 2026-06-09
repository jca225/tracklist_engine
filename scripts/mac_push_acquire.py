#!/usr/bin/env python3
"""Mac-side: YT Music search + yt-dlp download + scp + pi replace_track_audio.

Used when pi-storage yt-dlp hits bot detection on full downloads but Mac works.

SoundCloud-only scrape rows (no YouTube link) are out of scope here — use
ingest.main or replace_track_audio --url with an api.soundcloud.com/tracks/<id>
URL instead. SC-only is not a skip condition for the corpus.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ingest.adapters import ytmusic_adapter
from ingest.search_query import to_search_query
from core.result import Err, Ok

YTDLP = REPO / "venvs/audio/bin/yt-dlp"
YTDLP_BASE = [
    str(YTDLP),
    "--js-runtimes", "node:/opt/homebrew/bin/node",
    "--remote-components", "ejs:github",
    "--cookies-from-browser", "safari",
    "-f", "ba[ext=m4a]/bestaudio[ext=m4a]/bestaudio/best",
]
PI_HOST = "pi-storage"
PI_REPO = "~/tracklist_engine"
PI_PY = "venvs/audio/bin/python"


@dataclass(frozen=True)
class Job:
    track_id: str
    track_audio_id: int | None  # None = acquire (add row)
    full_name: str
    artists_csv: str
    title: str
    exclude_vids: frozenset[str]
    set_id: str | None
    reason: str
    force_video_id: str | None = None  # skip search, use this video


def _search(job: Job) -> str | None:
    if job.force_video_id:
        return job.force_video_id
    q = to_search_query(job.full_name, job.artists_csv, job.title)
    r = ytmusic_adapter.search(q, limit=8)
    if isinstance(r, Err):
        print(f"  search failed: {r.error.detail}", file=sys.stderr)
        return None
    for hit in r.value:
        if hit.video_id not in job.exclude_vids:
            dur = hit.duration_s or 0
            # reject playlist-scale (> 20 min) unless no alternative
            if dur > 1200:
                print(f"  skip {hit.video_id} ({dur}s): {hit.title}")
                continue
            print(f"  pick {hit.video_id} ({dur}s): {hit.title}")
            return hit.video_id
    return None


def _download(vid: str, out: Path) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file() and out.stat().st_size > 100_000:
        print(f"  reuse {out}")
        return True
    cmd = [*YTDLP_BASE, "-o", str(out), f"https://www.youtube.com/watch?v={vid}"]
    print(f"  dl: {' '.join(cmd[-2:])}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-800:] if r.stderr else r.stdout[-800:], file=sys.stderr)
        return False
    return out.is_file() and out.stat().st_size > 100_000


def _push_and_register(job: Job, local: Path, video_id: str, *, dry_run: bool) -> int:
    remote_tmp = f"/tmp/mac_acquire_{job.track_id}.m4a"
    parts = [
        "scripts/replace_track_audio.py",
        "--db", "/mnt/storage/data/db/music_database.db",
        "--audio-root", "/mnt/storage",
        "--track-id", job.track_id,
        "--file", remote_tmp,
        f"--player-id={video_id}",  # may start with '-' (YouTube ids)
        "--reason", job.reason,
    ]
    if job.track_audio_id is not None:
        parts.extend(["--track-audio-id", str(job.track_audio_id)])
    if job.set_id:
        parts.extend(["--set-id", job.set_id])
    inner = " ".join(shlex.quote(p) for p in parts)
    remote_cmd = f"cd {PI_REPO} && {PI_PY} {inner}"

    if dry_run:
        print(f"  DRY scp {local} -> {PI_HOST}:{remote_tmp}")
        print(f"  DRY {remote_cmd}")
        return 0

    scp = subprocess.run(
        ["scp", str(local), f"{PI_HOST}:{remote_tmp}"],
        capture_output=True, text=True,
    )
    if scp.returncode != 0:
        print(scp.stderr, file=sys.stderr)
        return 1

    ssh = subprocess.run(
        ["ssh", PI_HOST, remote_cmd],
        capture_output=True, text=True,
    )
    print(ssh.stdout)
    if ssh.returncode != 0:
        print(ssh.stderr, file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--work-dir", type=Path, default=Path("/tmp/tracklist_acquire"))
    args = p.parse_args(argv)

    jobs: tuple[Job, ...] = (
        # acquire — reconcile pass1 emptied wrong files
        Job("4dtxu75", None,
            "Green Velvet - Flash (Nicky Romero Remix)",
            "Green Velvet", "Flash",
            frozenset({"3Qffe2hLRvo"}), "115btk8t",
            "re-acquire: pass1 removed 650s wrong Flash (3Qffe2hLRvo)"),
        Job("19bg4m9p", None,
            "Daft Punk - Around The World (Dimitri Vegas & Like Mike Remix)",
            "Daft Punk", "Around The World",
            frozenset({"Jb6gcoR266U"}), "14gkrs81",
            "re-acquire: pass1 removed ATW full (Jb6gcoR266U belongs on gnsjmf)",
            force_video_id="sWf-y-73fmM"),
        Job("1fw35fxp", None,
            "GTA & Astronomar - Heavy Thunder",
            "GTA, Astronomar", "Heavy Thunder",
            frozenset({"naNj8bXnySE", "A7nR703Komg"}), "1652uk1",
            "re-acquire: pass1 removed 4500s playlist (naNj8bXnySE)",
            force_video_id=None),  # SKIP below — no YT hit; SC-only scrape
        Job("1mlz2hg5", None,
            "KURA - Thunder",
            "KURA", "Thunder",
            frozenset({"naNj8bXnySE", "h5gNbrtlKQo"}), "2h9wg22t",
            "re-acquire: pass1 removed 4500s playlist (naNj8bXnySE)",
            force_video_id="wsItf9KArQc"),
        Job("1uz8820p", None,
            "deadmau5 - Strobe (Lane 8 Remix)",
            "deadmau5", "Strobe",
            frozenset({"AT0e3LGteoA"}), "11p05cmt",
            "re-acquire: pass1 removed wrong Lane 8 length (AT0e3LGteoA)"),
        Job("1wws7mtf", None,
            "Hardwell & Armin van Buuren - Boundaries (AMF 2017 Two Is One Official Anthem)",
            "Hardwell, Armin van Buuren", "Boundaries",
            frozenset({"aBq83ksKmUo", "LoRzim8svTA"}), "2uhd818t",
            "re-acquire: pass1 removed 4514s year-mix blob (aBq83ksKmUo)",
            force_video_id="nseGMfUUliM"),
        Job("hm0pvnp", None,
            "Armin van Buuren ft. Uni V. Sol - A State Of Trance Year Mix 2021 (Intro - Learn To Dance Again)",
            "Armin van Buuren", "A State Of Trance Year Mix 2021",
            frozenset({"aBq83ksKmUo"}), "15m01xu9",
            "re-acquire: pass1 removed year-mix blob; use 1001tl scrape URL",
            force_video_id="6owT-UpEG7c"),
        # resource — bare-query remix re-source (BB12)
        Job("100hm0dp", 20778,
            "Martin Garrix & Troye Sivan - There For You (Madison Mars Remix)",
            "Martin Garrix, Troye Sivan", "There For You",
            frozenset(), "1fsnxchk",
            "re-source: pre-2026-05-13 bare-query resolved to original"),
        Job("4gyw2tx", 20779,
            "Hardwell - Spaceman (Carnage Festival Trap Remix)",
            "Hardwell", "Spaceman",
            frozenset(), "1fsnxchk",
            "re-source: pre-2026-05-13 bare-query resolved to original"),
        Job("1lm1yrff", 20780,
            "Bingo Players - Mode (Jay Hardway Remix)",
            "Bingo Players", "Mode",
            frozenset(), "1fsnxchk",
            "re-source: pre-2026-05-13 bare-query resolved to original"),
        Job("23b7cvbx", 20781,
            "Mako - Smoke Filled Room (Elephante Remix)",
            "Mako", "Smoke Filled Room",
            frozenset(), "1fsnxchk",
            "re-source: pre-2026-05-13 bare-query resolved to original"),
    )

    failed = 0
    for job in jobs:
        print(f"\n=== {job.track_id} ({'replace' if job.track_audio_id else 'acquire'}) ===")
        if job.track_id == "1fw35fxp" and job.force_video_id is None:
            print("  SKIP here: SC-only — use ingest.main or replace_track_audio --url "
                  "(api.soundcloud.com/tracks/<id>)")
            failed += 1
            continue
        vid = _search(job)
        if not vid:
            print("  FAIL: no video", file=sys.stderr)
            failed += 1
            continue
        local = args.work_dir / f"{job.track_id}.m4a"
        if not args.dry_run and not _download(vid, local):
            failed += 1
            continue
        if _push_and_register(job, local, vid, dry_run=args.dry_run) != 0:
            failed += 1

    print(f"\nDone: {len(jobs) - failed}/{len(jobs)} ok, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
