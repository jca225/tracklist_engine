#!/usr/bin/env python3
"""Coverage → ingest job file — turn the grammar-coverage ranking into downloads.

``grammar_coverage.py`` fingerprints every scraped set, finds the starved
grammar corners, and ranks fetchable-but-undownloaded sets by how much each one
fills those corners (``Coverage.fill_weight``). This module is the *executor*:
it takes the top-N of that ranking and emits a **job file** the existing ingest
stack already consumes — the same drop-in pattern as
``eda/queries/tier1_plus_bb.sql`` → ``data/djs/*.json`` — so
``python -m ingest.main --job-file data/djs/grammar_coverage_batch.json`` picks
it up with NO glue code.

This is a *caller/emitter*, not a modifier of ingest internals:

  - It re-uses ``grammar_coverage.fetch_rows()`` + ``build_coverage()`` to get
    the ranked, live-PA-filtered, fetchable-undownloaded candidates (already
    sorted by starvation fill weight — the fingerprint is NOT re-implemented).
  - It fetches each candidate's rich metadata + media URL (SoundCloud preferred
    over YouTube) via ONE read-only aggregate query against the canonical DB, so
    the emitted rows match the existing job-file schema exactly.
  - It writes ``data/djs/grammar_coverage_batch.json`` and NEVER invokes a
    downloader or mutates the canonical DB.

Dry-run by default: prints a summary (count, grammar corners filled, SC-vs-YT
split, dropped-with-reason) + the first rows, and writes NOTHING. Pass
``--write`` to actually write the job file.

Read-only against pi-storage; ``encoding="utf-8"`` is pinned on the ssh/sqlite
subprocess (the pi locale is Latin-1 — unpinned text mojibakes unicode titles).

Usage:
    # dry-run: summary + preview, writes nothing
    venvs/audio/bin/python -m eda.alignment.generalization.download_from_coverage [--top 500]
    # actually emit the job file for ingest.main to consume
    venvs/audio/bin/python -m eda.alignment.generalization.download_from_coverage --top 500 --write
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from eda.alignment.generalization.grammar_coverage import (
    Coverage,
    SetFingerprint,
    build_coverage,
    fetch_rows,
    joint_cell,
)
from eda.alignment.generalization.rank_ingest_queue import PI_DB, PI_HOST

# Repo convention (eda/CLAUDE.md, core.acquisition_case.default_path): scripts run
# from the project root, so data paths are CWD-relative — not Path.parents[N].
JOB_DIR = Path("data") / "djs"
JOB_NAME = "grammar_coverage_batch.json"

# joint-cell corpus-mass floor for "is this a real starved corner" (mirrors
# grammar_coverage.MIN_CELL — a zero-coverage cell of 2 sets is scrape noise).
MIN_CELL = 20

# Job-file schema (matches data/djs/*.json produced by tier1_plus_bb.sql):
#   an array of objects with these keys. tracklist_id is the only field
#   ingest.main reads; the rest are descriptive metadata carried for parity
#   with the other job files (and human inspection of the batch).
JOB_FIELDS = (
    "tracklist_id",
    "title",
    "url",
    "date",
    "creator_name",
    "creator_url",
    "views",
    "ided_tracks",
    "total_tracks",
    "play_time",
    "likes",
    "styles",
)


@dataclass(frozen=True)
class SetMeta:
    """Rich per-set metadata + preferred media URL for a candidate set."""

    set_id: str
    title: str
    set_url: str
    date: str
    creator_name: str
    creator_url: str
    views: int
    ided_tracks: int
    total_tracks: int
    play_time: str
    likes: int
    styles: str
    media_url: str  # SoundCloud preferred over YouTube; "" if neither
    link_platform: str  # "soundcloud" | "youtube" | "none"


# ---------------------------------------------------------------------------
# I/O edge — one read-only aggregate query for the candidate metadata + URLs.
# ---------------------------------------------------------------------------

# GROUP the set's media links so SC is preferred over YT (SC is usually the DJ's
# own full-length upload; fewer bot-detection stalls — same policy as
# rank_ingest_queue). All the dj_sets descriptive columns ride along so the
# emitted rows match the tier1_plus_bb.sql job-file schema.
_META_SQL = """
WITH links AS (
  SELECT set_id,
         MAX(CASE WHEN platform='soundcloud' THEN COALESCE(url,'') END) AS sc_url,
         MAX(CASE WHEN platform='youtube'   THEN COALESCE(url,'') END) AS yt_url
  FROM dj_set_media_links GROUP BY set_id
)
SELECT d.set_id,
       REPLACE(COALESCE(d.title,''), char(9), ' '),
       COALESCE(d.set_url,''),
       COALESCE(d.date_played,''),
       REPLACE(COALESCE(d.creator_name,''), char(9), ' '),
       COALESCE(d.creator_url,''),
       COALESCE(d.views,0),
       COALESCE(d.ided_tracks,0),
       COALESCE(d.total_tracks,0),
       COALESCE(d.play_time,''),
       COALESCE(d.likes,0),
       REPLACE(COALESCE(d.styles,''), char(9), ' '),
       COALESCE(l.sc_url,''),
       COALESCE(l.yt_url,'')
FROM dj_sets d
LEFT JOIN links l ON l.set_id=d.set_id
WHERE d.set_id IN ({ids});
"""


def _quote_id(sid: str) -> str:
    """SQL-quote a set_id (they are hex-ish slugs, but be safe against ')."""
    return "'" + sid.replace("'", "''") + "'"


def _meta_row_to_setmeta(c: list[str]) -> SetMeta:
    sc_url, yt_url = c[12], c[13]
    if sc_url:
        media_url, platform = sc_url, "soundcloud"
    elif yt_url:
        media_url, platform = yt_url, "youtube"
    else:
        media_url, platform = "", "none"
    return SetMeta(
        set_id=c[0],
        title=c[1],
        set_url=c[2],
        date=c[3],
        creator_name=c[4],
        creator_url=c[5],
        views=int(c[6]) if c[6] else 0,
        ided_tracks=int(c[7]) if c[7] else 0,
        total_tracks=int(c[8]) if c[8] else 0,
        play_time=c[9],
        likes=int(c[10]) if c[10] else 0,
        styles=c[11],
        media_url=media_url,
        link_platform=platform,
    )


def fetch_meta(set_ids: list[str]) -> dict[str, SetMeta]:
    """Read-only fetch of rich metadata + preferred media URL for `set_ids`.

    Returns a dict keyed by set_id. `encoding="utf-8"` is load-bearing (pi
    locale is Latin-1). Empty input short-circuits (no query).
    """
    if not set_ids:
        return {}
    ids = ",".join(_quote_id(s) for s in set_ids)
    sql = _META_SQL.format(ids=ids)
    proc = subprocess.run(
        [
            "ssh",
            PI_HOST,
            f'sqlite3 -readonly -separator "\t" {PI_DB} ".timeout 60000" "{sql}"',
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
        check=True,
    )
    out: dict[str, SetMeta] = {}
    for line in proc.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) == 14:
            m = _meta_row_to_setmeta(parts)
            out[m.set_id] = m
    return out


# ---------------------------------------------------------------------------
# Pure core — candidate ranking → job rows (the unit-tested surface).
# ---------------------------------------------------------------------------


def rank_candidates(
    rows: list[SetFingerprint], cov: Coverage, live_pa_thresh: float
) -> list[SetFingerprint]:
    """The fetchable, non-live-PA candidates, sorted by fill weight (desc).

    Reuses ``grammar_coverage``'s ``Coverage.fill_weight`` fingerprint — this is
    the same ranking ``grammar_coverage.main`` produces, extracted so the
    executor and the analysis tool agree by construction.
    """
    candidates = [r for r in rows if r.fetchable and r.self_frac < live_pa_thresh]
    return sorted(
        candidates, key=lambda r: (-cov.fill_weight(r), r.self_frac, -r.n_slots)
    )


def candidate_to_job_row(fp: SetFingerprint, meta: SetMeta | None) -> dict:
    """Map a ranked candidate (+ its fetched metadata) to a job-file object.

    ``tracklist_id`` is always the fingerprint's set_id (what ingest.main
    reads). When metadata is present the descriptive fields come from it (and
    ``url`` = the preferred MEDIA url, SC>YT — the actual mix audio to pull, not
    the 1001tracklists page). When metadata is missing (set vanished between the
    two reads), fields fall back to the fingerprint's own title + nulls so the
    row is still a valid, downloadable job entry.
    """
    if meta is None:
        return {
            "tracklist_id": fp.set_id,
            "title": fp.title,
            "url": "",
            "date": None,
            "creator_name": None,
            "creator_url": None,
            "views": None,
            "ided_tracks": None,
            "total_tracks": fp.total_tracks or None,
            "play_time": None,
            "likes": None,
            "styles": fp.styles or None,
        }
    return {
        "tracklist_id": meta.set_id,
        "title": meta.title or fp.title,
        "url": meta.media_url,
        "date": meta.date or None,
        "creator_name": meta.creator_name or None,
        "creator_url": meta.creator_url or None,
        "views": meta.views,
        "ided_tracks": meta.ided_tracks,
        "total_tracks": meta.total_tracks,
        "play_time": meta.play_time or None,
        "likes": meta.likes,
        "styles": meta.styles or None,
    }


def build_job_rows(
    ranked: list[SetFingerprint], meta: dict[str, SetMeta], top: int
) -> list[dict]:
    """Top-N ranked candidates → job-file objects, in rank order."""
    return [candidate_to_job_row(fp, meta.get(fp.set_id)) for fp in ranked[:top]]


# ---------------------------------------------------------------------------
# Summary / reporting — no silent caps: everything dropped is counted + named.
# ---------------------------------------------------------------------------


def summarize(
    rows: list[SetFingerprint],
    ranked: list[SetFingerprint],
    selected: list[SetFingerprint],
    meta: dict[str, SetMeta],
    cov: Coverage,
    live_pa_thresh: float,
) -> list[str]:
    """Human-readable selection summary. Reports every drop and its reason."""
    n_total = len(rows)
    n_downloaded = sum(1 for r in rows if r.has_audio)
    n_fetchable = sum(1 for r in rows if r.fetchable)
    n_live_pa = sum(1 for r in rows if r.fetchable and r.self_frac >= live_pa_thresh)
    n_candidates = len(ranked)

    # SC-vs-YT split + no-link count over the SELECTED slice.
    sc = yt = none = 0
    for fp in selected:
        m = meta.get(fp.set_id)
        plat = m.link_platform if m else "none"
        if plat == "soundcloud":
            sc += 1
        elif plat == "youtube":
            yt += 1
        else:
            none += 1

    # Grammar corners the selected slice fills: which starved joint cells they
    # land in (the same (w,version,stem) cells grammar_coverage reports as thin).
    corner_hits: Counter = Counter()
    for fp in selected:
        cell = joint_cell(fp)
        if cov.joint_n.get(cell, 0) >= MIN_CELL:
            corner_hits[(cell, round(cov.joint.get(cell, 0.0), 3))] += 1

    lines: list[str] = []
    lines.append("=== grammar-coverage download selection ===")
    lines.append(
        f"corpus sets: {n_total}  downloaded: {n_downloaded}  fetchable: {n_fetchable}"
    )
    lines.append("--- drops (no silent caps) ---")
    lines.append(f"  already downloaded (has set_audio):  {n_downloaded}")
    lines.append(
        f"  not fetchable (no SC/YT link or already downloaded): "
        f"{n_total - n_fetchable}"
    )
    lines.append(f"  live-PA excluded (self_frac >= {live_pa_thresh}):    {n_live_pa}")
    n_missing_meta = sum(1 for fp in selected if fp.set_id not in meta)
    if n_missing_meta:
        lines.append(
            f"  selected but metadata vanished (row kept w/ fallback): {n_missing_meta}"
        )
    if none:
        lines.append(f"  selected but NO media url after re-fetch (url=''): {none}")
    lines.append("--- selected ---")
    lines.append(
        f"  ranked candidates: {n_candidates}  selected (top-N): {len(selected)}"
    )
    lines.append(f"  media-link split: soundcloud={sc}  youtube={yt}  none={none}")
    lines.append(f"--- grammar corners filled (starved joint cells, n>={MIN_CELL}) ---")
    if corner_hits:
        for (cell, cov_frac), n in corner_hits.most_common(8):
            lines.append(
                f"  (w,ver,stem)={cell}  coverage={cov_frac:.3f}  selected_here={n}"
            )
    else:
        lines.append("  (no selected sets land in a mass-gated starved cell)")
    return lines


def preview(
    selected: list[SetFingerprint],
    meta: dict[str, SetMeta],
    cov: Coverage,
    n: int = 15,
) -> list[str]:
    lines = [f"--- first {min(n, len(selected))} selected ---"]
    for i, fp in enumerate(selected[:n], 1):
        m = meta.get(fp.set_id)
        plat = m.link_platform if m else "none"
        title = (m.title if m else fp.title) or fp.set_id
        lines.append(
            f"  {i:>2}. fill={cov.fill_weight(fp):5.2f}  self={fp.self_frac:.2f}  "
            f"w={fp.w_frac:.2f} ver={fp.version_frac:.2f} stem={fp.stem_frac:.2f}  "
            f"[{plat:>10}]  {title[:44]}  ({fp.set_id})"
        )
    return lines


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--top",
        type=int,
        default=500,
        help="number of candidates to emit (default 500)",
    )
    ap.add_argument(
        "--live-pa-thresh",
        type=float,
        default=0.7,
        help="exclude sets with self_frac >= this (own-material live PAs)",
    )
    ap.add_argument(
        "--write",
        action="store_true",
        help="actually write the job file (default: dry-run, print only)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=JOB_DIR / JOB_NAME,
        help=f"job-file path (default: {JOB_DIR / JOB_NAME})",
    )
    args = ap.parse_args(argv)

    rows = fetch_rows()
    cov = build_coverage(rows)
    ranked = rank_candidates(rows, cov, args.live_pa_thresh)
    selected = ranked[: args.top]

    # Fetch rich metadata + media URLs only for the selected slice (read-only).
    meta = fetch_meta([fp.set_id for fp in selected])

    print("\n".join(summarize(rows, ranked, selected, meta, cov, args.live_pa_thresh)))
    print()
    print("\n".join(preview(selected, meta, cov)))

    job_rows = build_job_rows(ranked, meta, args.top)

    if args.write:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(job_rows, indent=2) + "\n", encoding="utf-8")
        print(f"\nWROTE {len(job_rows)} job rows -> {args.out}")
        print(f"  feed to ingest: python -m ingest.main --job-file {args.out}")
    else:
        print(
            f"\n[dry-run] would write {len(job_rows)} job rows -> {args.out}  "
            "(pass --write to emit)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
