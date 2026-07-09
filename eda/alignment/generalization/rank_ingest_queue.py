#!/usr/bin/env python3
"""Rank the set-audio ingest queue — the generalization bottleneck, ordered.

The aligner's runnable corpus is gated on set audio (2026-07-09 profile:
561/41,492 sets have it, while ~23k tokenized sets carry a YouTube/SoundCloud
media link the existing ingest stack can pull). This script turns that pipe
into a concrete, prioritized work order: one row per ingestable set, scored by

  - popularity      log10(views) + 0.5*log10(likes)   (1001tracklists counters)
  - data quality    ided_tracks/total_tracks (tokenizer ID rate) + slot count
  - runnability     fraction of the set's slot recordings that ALREADY have
                    track_audio on pi (high = the set becomes end-to-end
                    runnable almost for free once the mix lands)
  - link quality    SoundCloud preferred over YouTube-only (usually the DJ's
                    own full-length upload; fewer bot-detection stalls)
  - whitelist       Two Friends / It's Murph / John Summit / Disco Lines
                    (docs/alignment_objective.md corpus scope)

All inputs are read-only queries against the canonical DB on pi-storage.
Output: out/ingest_queue.tsv (rank, score, set_id, dj-ish title, url picks,
per-part scores) — feed the top slice to the ingest job runner.

Usage:
    venvs/audio/bin/python -m eda.alignment.generalization.rank_ingest_queue \
        [--top 1000]
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "out"
PI_HOST = "pi-storage"
PI_DB = "/mnt/storage/data/db/music_database.db"

WHITELIST = ("two friends", "john summit", "disco lines", "murph")

# One aggregated row per candidate set (no raw-table dumps):
#   set_id, title, views, likes, ided, total, n_slots, n_resolved,
#   n_with_audio, has_sc, has_yt, sc_url, yt_url
_SQL = """
WITH slots AS (
  SELECT set_id,
         COUNT(*) AS n_slots,
         SUM(CASE WHEN recording_id IS NOT NULL THEN 1 ELSE 0 END) AS n_resolved
  FROM set_track_slots GROUP BY set_id HAVING n_slots >= 10
),
have AS (SELECT DISTINCT recording_id FROM track_audio),
cov AS (
  SELECT s.set_id,
         SUM(CASE WHEN h.recording_id IS NOT NULL THEN 1 ELSE 0 END) AS n_with_audio
  FROM set_track_slots s LEFT JOIN have h ON h.recording_id = s.recording_id
  GROUP BY s.set_id
),
links AS (
  SELECT set_id,
         MAX(CASE WHEN platform='soundcloud' THEN COALESCE(url,'') END) AS sc_url,
         MAX(CASE WHEN platform='youtube'   THEN COALESCE(url,'') END) AS yt_url,
         MAX(CASE WHEN platform='soundcloud' THEN 1 ELSE 0 END) AS has_sc,
         MAX(CASE WHEN platform='youtube'   THEN 1 ELSE 0 END) AS has_yt
  FROM dj_set_media_links GROUP BY set_id
),
audio AS (SELECT DISTINCT set_id FROM set_audio)
SELECT d.set_id, REPLACE(d.title, char(9), ' '), COALESCE(d.views,0),
       COALESCE(d.likes,0), COALESCE(d.ided_tracks,0), COALESCE(d.total_tracks,0),
       sl.n_slots, sl.n_resolved, COALESCE(c.n_with_audio,0),
       l.has_sc, l.has_yt, COALESCE(l.sc_url,''), COALESCE(l.yt_url,'')
FROM dj_sets d
JOIN slots sl ON sl.set_id = d.set_id
JOIN links l  ON l.set_id = d.set_id AND (l.has_sc=1 OR l.has_yt=1)
JOIN cov c    ON c.set_id = d.set_id
LEFT JOIN audio a ON a.set_id = d.set_id
WHERE a.set_id IS NULL;
"""


def _fetch() -> list[list[str]]:
    proc = subprocess.run(
        ["ssh", PI_HOST, f'sqlite3 -readonly -separator "\t" {PI_DB} "{_SQL}"'],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.split("\t") for line in proc.stdout.strip().splitlines()]


def _score(row: list[str]) -> dict:
    (
        set_id,
        title,
        views,
        likes,
        ided,
        total,
        n_slots,
        n_resolved,
        n_with_audio,
        has_sc,
        has_yt,
        sc_url,
        yt_url,
    ) = row
    views, likes = int(views), int(likes)
    ided, total = int(ided), int(total)
    n_slots, n_resolved = int(n_slots), int(n_resolved)
    n_with_audio = int(n_with_audio)

    pop = math.log10(1 + views) + 0.5 * math.log10(1 + likes)  # ~0..7
    id_rate = ided / total if total else n_resolved / n_slots
    coverage = n_with_audio / n_slots
    link = 1.0 if int(has_sc) else 0.5  # SC preferred; YT-only still fine
    wl = 1.0 if any(w in title.lower() for w in WHITELIST) else 0.0

    # weights: popularity is the user-stated first-order signal; quality and
    # runnability keep the queue honest; whitelist rides on top per the
    # objective doc's curated-corpus clause.
    score = 1.0 * pop + 2.0 * id_rate + 2.0 * coverage + 1.0 * link + 3.0 * wl
    return {
        "set_id": set_id,
        "title": title[:80],
        "score": round(score, 3),
        "views": views,
        "likes": likes,
        "id_rate": round(id_rate, 3),
        "n_slots": n_slots,
        "audio_coverage": round(coverage, 3),
        "link": "soundcloud" if int(has_sc) else "youtube",
        "whitelist": int(wl),
        "url": sc_url or yt_url,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=1000, help="rows to write")
    args = ap.parse_args(argv)

    rows = [_score(r) for r in _fetch()]
    rows.sort(key=lambda r: -r["score"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / "ingest_queue.tsv"
    with dest.open("w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["rank"] + list(rows[0].keys()), delimiter="\t"
        )
        w.writeheader()
        for i, r in enumerate(rows[: args.top], 1):
            w.writerow({"rank": i, **r})

    n_sc = sum(1 for r in rows if r["link"] == "soundcloud")
    n_wl = sum(1 for r in rows if r["whitelist"])
    print(f"candidates: {len(rows)}  (soundcloud {n_sc}, whitelist {n_wl})")
    print(f"wrote top {min(args.top, len(rows))} -> {dest}")
    print("\ntop 15:")
    for i, r in enumerate(rows[:15], 1):
        print(
            f"  {i:>3}. [{r['score']:5.2f}] {r['title'][:56]:<56} "
            f"views={r['views']:<9} id={r['id_rate']:.2f} cov={r['audio_coverage']:.2f} {r['link']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
