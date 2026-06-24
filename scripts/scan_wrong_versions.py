#!/usr/bin/env python3
"""Scan corpus for wrong-version suspects (Topic original, live, wrong remix).

Emits CSV for human review. Uses oEmbed where available + metadata heuristics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sqlite3
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

_REMIX_IN_NAME = re.compile(
    r"\(([^)]*\b(?:remix|rework|bootleg|mashup|edit|flip|vip)\b[^)]*)\)",
    re.I,
)
_LIVE_HINT = re.compile(r"\blive\b|\bconcert\b|\bfestival\b|\b@ ", re.I)


@dataclass(frozen=True)
class Suspect:
    track_audio_id: int
    track_id: str
    platform: str
    player_id: str
    full_name: str
    oembed_title: str
    klass: str
    detail: str


def _oembed_title(video_id: str) -> str:
    url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        return str(data.get("title") or "")
    except Exception:
        return ""


def _classify_row(
    taid: int,
    track_id: str,
    platform: str,
    player_id: str,
    full_name: str | None,
    version: str | None,
) -> Suspect | None:
    name = full_name or ""
    remix_m = _REMIX_IN_NAME.search(name)
    named_remix = remix_m is not None and " remix" in remix_m.group(1).lower()

    oembed = ""
    if platform in ("youtube", "youtube_music", "manual") and player_id:
        oembed = _oembed_title(player_id)

    if named_remix and oembed:
        oembed_low = oembed.lower()
        remixer_bit = remix_m.group(1).lower() if remix_m else ""
        if "topic" in oembed_low and remixer_bit.split()[0] not in oembed_low:
            return Suspect(
                taid,
                track_id,
                platform,
                player_id or "",
                name,
                oembed,
                "topic_original",
                "named remix in metadata but Topic channel title",
            )
        if remixer_bit and remixer_bit.split()[0] not in oembed_low:
            return Suspect(
                taid,
                track_id,
                platform,
                player_id or "",
                name,
                oembed,
                "wrong_remix",
                "oEmbed title lacks named remixer",
            )

    if oembed and _LIVE_HINT.search(oembed) and not _LIVE_HINT.search(name):
        return Suspect(
            taid,
            track_id,
            platform,
            player_id or "",
            name,
            oembed,
            "live_suspect",
            "oEmbed suggests live performance",
        )

    if (
        (version or "") == "mashup"
        and "mashup" not in name.lower()
        and " vs" not in name.lower()
    ):
        return Suspect(
            taid,
            track_id,
            platform,
            player_id or "",
            name,
            oembed,
            "mashup_metadata_gap",
            "version=mashup but full_name lacks mashup hint",
        )

    return None


def scan(db_path: Path, *, limit: int | None = None) -> list[Suspect]:
    conn = sqlite3.connect(db_path)
    q = """
        SELECT ta.track_audio_id, ta.track_id, ta.platform, ta.player_id,
               tm.full_name, tm.version
        FROM track_audio ta
        LEFT JOIN track_metadata tm ON tm.track_id = ta.track_id
        WHERE ta.is_reference = 1
        ORDER BY ta.track_audio_id
    """
    if limit:
        q += f" LIMIT {int(limit)}"
    rows = conn.execute(q).fetchall()
    out: list[Suspect] = []
    for taid, tid, plat, pid, fn, ver in rows:
        s = _classify_row(int(taid), tid, plat or "", pid or "", fn, ver)
        if s:
            out.append(s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--db",
        type=Path,
        default=Path(
            os.environ.get("TRACKLIST_DB", "/mnt/storage/data/db/music_database.db")
        ),
    )
    ap.add_argument("--out", type=Path, default=Path("wrong_version_scan.csv"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    suspects = scan(args.db, limit=args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "track_audio_id",
                "track_id",
                "klass",
                "full_name",
                "oembed_title",
                "detail",
            ]
        )
        for s in suspects:
            w.writerow(
                [
                    s.track_audio_id,
                    s.track_id,
                    s.klass,
                    s.full_name,
                    s.oembed_title,
                    s.detail,
                ]
            )
    print(f"Wrote {len(suspects)} suspects -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
