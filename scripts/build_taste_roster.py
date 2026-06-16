#!/usr/bin/env python3
"""Build a validated taste-prior scrape roster from the 1001tracklists corpus.

For each artist in ROSTER, pull the SC-linked DJ sets from the canonical corpus
(dj_sets + dj_set_media_links on pi-storage), then **resolve every SoundCloud
link** against the live SC API and keep only the ones whose upload still EXISTS
(John's rule: "the SoundCloud DJ-set label must exist for the 1001tracklists sets
we scrape"). Dead/private re-uploads 404 and are dropped. Resolving also yields the
real like_count — a better cohort-size signal than 1001tl page-views — which we
rank by.

Output:
  - data/taste/roster_candidates.yaml   (mixes.yaml-shaped, keyed by 1001tl set_id)
  - a manifest table on stdout (per-artist kept/dropped + likes)

It does NOT touch personalization/config/mixes.yaml or kick off any scrape — review
the candidates file, then merge. Pure read + one read-only SC resolve per set.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field

import yaml

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from personalization.soundcloud_client import (  # noqa: E402
    RateLimiter,
    extract_client_id,
    resolve_track,
    sc_client,
)

PI_DB = "/mnt/storage/data/db/music_database.db"

# (display title, SQL LIKE needle). "etc." is editable — add a line to extend.
ROSTER: list[tuple[str, str]] = [
    ("Galantis", "galantis"),
    ("Porter Robinson", "porter robinson"),
    ("Chris Lake", "chris lake"),
    ("John Summit", "john summit"),
    ("Fisher", "fisher"),
    ("Avicii", "avicii"),
    ("Disco Lines", "disco lines"),
    ("It's Murph", "murph"),
    ("Two Friends", "two friends"),
    ("RÜFÜS DU SOL", "du sol"),  # SQLite lower() doesn't fold the umlaut in "RÜFÜS"
]

# pull more than we keep, to absorb 404s / private uploads / junk
CANDIDATES_PER_ARTIST = 30
KEEP_PER_ARTIST = 10
# floor drops wrong-artist false positives (e.g. "Fisherman") + tiny no-cohort
# radio bits that the LIKE needle sweeps in. Real DJ sets clear this easily.
MIN_LIKES = 500

# api.soundcloud.com/tracks/<id> OR the api-v2 resolve-able permalink in the embed
TRACK_ID_RE = re.compile(r"api\.soundcloud\.com/tracks/(\d+)")
SECRET_RE = re.compile(r"secret_token%3D(s-[A-Za-z0-9]+)")


@dataclass
class Candidate:
    set_id: str
    title: str
    views: int
    tracks: int
    sc_track_id: str
    secret: str | None = None
    # filled after resolve:
    exists: bool = False
    likes: int = 0
    plays: int = 0
    permalink: str = ""


def query_corpus(needle: str) -> list[Candidate]:
    """Top SC-linked sets for one artist, via ssh sqlite3 on pi-storage."""
    sql = f"""
SELECT s.set_id || '|' || COALESCE(s.views,0) || '|' || COALESCE(s.total_tracks,0)
       || '|' || (SELECT m.url FROM dj_set_media_links m
                  WHERE m.set_id=s.set_id AND m.platform='soundcloud' LIMIT 1)
       || '|' || REPLACE(substr(s.title,1,60),'|','/')
FROM dj_sets s
WHERE (lower(s.artists) LIKE '%{needle}%' OR lower(s.title) LIKE '%{needle}%')
  AND EXISTS (SELECT 1 FROM dj_set_media_links m
              WHERE m.set_id=s.set_id AND m.platform='soundcloud')
ORDER BY s.views DESC LIMIT {CANDIDATES_PER_ARTIST};
"""
    out = subprocess.run(
        ["ssh", "pi-storage", f"sqlite3 -noheader {PI_DB} \"{sql}\""],
        capture_output=True, text=True, timeout=60,
    ).stdout
    cands: list[Candidate] = []
    for line in out.splitlines():
        parts = line.split("|", 4)
        if len(parts) < 5:
            continue
        set_id, views, tracks, url, title = parts
        m = TRACK_ID_RE.search(url)
        if not m:
            continue
        sec = SECRET_RE.search(url)
        cands.append(Candidate(
            set_id=set_id, title=title, views=int(views or 0),
            tracks=int(tracks or 0), sc_track_id=m.group(1),
            secret=sec.group(1) if sec else None,
        ))
    return cands


def resolve_one(client, rl, cid: str, c: Candidate) -> None:
    url = f"https://api.soundcloud.com/tracks/{c.sc_track_id}"
    if c.secret:
        url += f"?secret_token={c.secret}"
    try:
        t = resolve_track(client, rl, cid, url)
        c.exists = True
        c.likes = int(t.get("likes_count") or t.get("favoritings_count") or 0)
        c.plays = int(t.get("playback_count") or 0)
        c.permalink = t.get("permalink_url") or ""
    except Exception:
        c.exists = False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rpm", type=int, default=30)
    ap.add_argument("--out", default="data/taste/roster_candidates.yaml")
    args = ap.parse_args()

    rl = RateLimiter(args.rpm)
    mixes: dict[str, dict] = {}
    seen: set[str] = set()  # dedup collab/B2B sets across artists
    print(f"{'artist':<16} {'kept':>4} {'dropped(404)':>12}  top set (likes)")
    print("-" * 72)
    with sc_client() as client:
        cid = extract_client_id(client, rl)
        for display, needle in ROSTER:
            cands = query_corpus(needle)
            for c in cands:
                resolve_one(client, rl, cid, c)
            existing = [c for c in cands if c.exists and c.likes >= MIN_LIKES and c.set_id not in seen]
            dropped = sum(1 for c in cands if not c.exists)
            existing.sort(key=lambda c: c.likes, reverse=True)
            kept = existing[:KEEP_PER_ARTIST]
            for c in kept:
                seen.add(c.set_id)
                mixes[c.set_id] = {
                    "set_id": c.set_id,
                    "title": c.title,
                    "soundcloud_url": c.permalink,
                    "_artist": display,
                    "_sc_likes": c.likes,
                    "_sc_plays": c.plays,
                    "_tl_views": c.views,
                    "_tracks": c.tracks,
                }
            top = f"{kept[0].title[:30]} ({kept[0].likes})" if kept else "—"
            print(f"{display:<16} {len(kept):>4} {dropped:>12}  {top}")

    out_path = __import__("pathlib").Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump({"mixes": mixes}, allow_unicode=True, sort_keys=False))
    print("-" * 72)
    print(f"total validated sets: {len(mixes)}  ->  {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
