"""One-shot: re-source yt-dlp `track_audio` rows via YT Music search.

Mirror of `redownload_via_spotdl.py` but uses `ytmusic_adapter` instead of
`spotdl_adapter`. Same end state — replace noisy 1001tracklists YT scrapes
with the clean studio audio YT Music surfaces under filter='songs' (which
maps to Topic-channel + label-uploaded album tracks).

Why this exists alongside redownload_via_spotdl.py:
  spotdl goes via Spotify Web API → ytmusicapi → yt-dlp. Two issues:
  (a) Spotify app credentials hit rate limits routinely (we burned
      ~13.6 hours of our daily quota on Day 1 of this work);
  (b) The Spotify hop is unnecessary — we already have artist + title
      in `track_metadata` (populated by web_crawler/tokenizer/materialize).

  This script cuts Spotify out: query YT Music directly with our local
  metadata, take the top 'songs' result, download via yt-dlp.

Two phases (matches the spotdl variant):
  Phase 1 (additive): yt-dlp downloads. Each success inserts a new
    track_audio row with platform='youtube_music'. Coexists temporarily
    with the existing yt-dlp row.
  Phase 2 (destructive): for each Phase-1 success, delete the yt-dlp
    track_audio row by track_audio_id. ON DELETE CASCADE removes
    track_analysis, track_stems, track_audio_features, track_mert_measures.
    Also unlinks the on-disk m4a and the stems dir.

Vast knock-on: the 42 already-analyzed yt-dlp tracks lose their
track_analysis row in the cascade. Vast's next_task() picks up the new
youtube_music track_audio_id automatically.

Usage:
  # Smoke
  venvs/audio/bin/python -m scripts.redownload_via_ytmusic --dry-run --max-tracks 5

  # Full run on pi-storage
  venvs/audio/bin/python -m scripts.redownload_via_ytmusic
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass, replace as dc_replace
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core import db as db_adapter
from ingest.adapters import ytmusic_adapter
from ingest.errors import DownloadError
from core.result import Err, Ok

_log = logging.getLogger("redownload_via_ytmusic")


_BB_SETS: frozenset[str] = frozenset((
    "w1mgcjt", "2nvzlh2k", "1fsnxchk", "qj4v0wt", "1yl70ql1", "237tdqmk",
))


# 1001tracklists uses "ID" as a placeholder when the remixer/track is
# unknown — "(ID Remix)", "(ID Bootleg)", "(ID)", "ID - ID", etc. The
# literal word "ID" inside a YT Music query derails search to random
# uploads, so we strip the parenthetical and fall back to the bare
# "Artist - Title" form for these.
_ID_PLACEHOLDER_RE: re.Pattern[str] = re.compile(
    r"\bID\b\s*(?:Remix|Bootleg|Edit|Mashup|VIP|Rework|Mix|Flip)?\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Candidate:
    """A track to feed through YT Music. Two flavors:

    - REPLACE: track already has a `track_audio` row sourced from raw
      yt-dlp (1001tracklists scrape, often noisy). `yt_track_audio_id`
      and `yt_audio_path` are the existing row + file. Phase 2 deletes
      them after the new YT Music row inserts.
    - ACQUIRE: no `track_audio` row exists yet. `yt_track_audio_id` is
      None. Phase 1 inserts the YT Music row; Phase 2 is a no-op for
      this candidate.
    """
    yt_track_audio_id: int | None  # None = acquire mode (no row to replace)
    yt_audio_path: str | None      # None = acquire mode
    track_id: str
    title: str
    artists_csv: str               # 'Daft Punk' or 'Artist1, Artist2'
    set_id: str
    is_bb: int
    version_tag: str | None        # Remix | Rework | Acappella | AltVersion | None
    full_name: str | None          # canonical scraped 'Artist - Title (Remixer Remix)'

    @property
    def query(self) -> str:
        # Always use the canonical scraped full_name when available — it
        # carries the remix qualifier ("(Madison Mars Remix)") and any
        # (Acappella)/(Instrumental Mix) marker that the tokenizer-split
        # `title` field drops. YT Music search tolerates these qualifiers
        # gracefully (returns the studio variant), while a bare
        # "Artist - Title" silently resolves to the original cut and is
        # the root cause of the variant-bleed bug observed in the corpus.
        # Exception: skip full_name when it carries 1001tracklists' "ID"
        # placeholder (e.g. "(ID Remix)") — the literal word "ID" in the
        # search corrupts results.
        if self.full_name and not _ID_PLACEHOLDER_RE.search(self.full_name):
            return self.full_name
        if self.artists_csv:
            return f"{self.artists_csv} - {self.title}"
        return self.title

    @property
    def needs_replace(self) -> bool:
        return self.yt_track_audio_id is not None


@dataclass(frozen=True)
class RunStats:
    candidates: int = 0
    skipped_no_metadata: int = 0
    phase1_ok: int = 0
    phase1_failed_dl: int = 0
    phase1_failed_search: int = 0
    phase1_failed_insert: int = 0
    phase2_replaced: int = 0
    phase2_skipped: int = 0
    phase2_failed: int = 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--db", type=Path,
                   default=Path(os.environ.get("TRACKLIST_DB",
                                               "/mnt/storage/data/db/music_database.db")))
    p.add_argument("--audio-root", type=Path,
                   default=Path(os.environ.get("TRACKLIST_AUDIO_ROOT", "/mnt/storage")))
    p.add_argument("--threads", type=int, default=4,
                   help="Parallel yt-dlp processes (4 is conservative; bump if "
                        "no rate-limit signs after a few hundred tracks).")
    p.add_argument("--timeout-per-track", type=float, default=120.0)
    p.add_argument("--max-tracks", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-replace", action="store_true",
                   help="Skip Phase 2 destructive cleanup; both rows coexist.")
    p.add_argument("--mode", choices=("replace", "acquire", "all"),
                   default="all",
                   help="replace=only existing yt-dlp rows (the original "
                        "redownload behavior); acquire=only tracks with no "
                        "track_audio row yet; all=both, replace candidates "
                        "first then acquire (default).")
    p.add_argument("--job-file", type=Path, default=None,
                   help="Optional JSON job file (e.g. data/djs/tier1_plus_bb.json) "
                        "to restrict candidates to tracks that appear in any of "
                        "those sets. Without this flag, the script considers "
                        "the entire corpus.")
    p.add_argument("--audio-format", default="m4a")
    p.add_argument("--cookies", type=Path,
                   default=Path(os.environ["TRACKLIST_YT_COOKIES"])
                       if os.environ.get("TRACKLIST_YT_COOKIES") else None,
                   help="yt-dlp cookies for age-gated tracks.")
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p.parse_args(argv)


def _row_to_candidate(r: sqlite3.Row) -> Candidate:
    try:
        artists = json.loads(r["artists_json"]) if r["artists_json"] else []
    except (json.JSONDecodeError, TypeError):
        artists = []
    artists_csv = ", ".join(a for a in artists if a)
    # yt_track_audio_id and yt_audio_path are NULL for ACQUIRE candidates.
    yt_taid = r["yt_track_audio_id"] if "yt_track_audio_id" in r.keys() else None
    yt_path = r["yt_audio_path"] if "yt_audio_path" in r.keys() else None
    return Candidate(
        yt_track_audio_id=yt_taid,
        yt_audio_path=yt_path,
        track_id=r["track_id"],
        title=r["title"],
        artists_csv=artists_csv,
        set_id=r["set_id"] or "",
        is_bb=r["is_bb"],
        version_tag=r["version_tag"] if "version_tag" in r.keys() else None,
        full_name=r["full_name"] if "full_name" in r.keys() else None,
    )


def _load_candidates(
    db_path: Path,
    mode: str,                            # 'replace' | 'acquire' | 'all'
    job_set_ids: frozenset[str] | None,   # None = whole corpus
) -> tuple[Candidate, ...]:
    """Loads candidates per `mode`:
      - 'replace': existing track_audio rows where platform='youtube'
      - 'acquire': tracks with no track_audio row yet
      - 'all': both, replace-first then acquire (Phase 2 only fires on
        replace candidates).

    `job_set_ids`, if provided, restricts to tracks appearing in any of
    those sets (e.g. tier1_plus_bb).

    Ordering: BB-first, then by set_id, then track_audio_id (or track_id
    for acquire candidates with no audio_id).
    """
    bb_csv = ",".join(f"'{s}'" for s in _BB_SETS)
    job_filter = ""
    if job_set_ids is not None and job_set_ids:
        job_csv = ",".join(f"'{s}'" for s in job_set_ids)
        job_filter = (
            "AND EXISTS ("
            "SELECT 1 FROM dj_set_track_media_links jm "
            f"WHERE jm.track_id = base.track_id AND jm.set_id IN ({job_csv})"
            ")"
        )

    out: list[Candidate] = []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        if mode in ("replace", "all"):
            replace_q = f"""
                WITH base AS (
                    SELECT ta.track_audio_id, ta.path, ta.track_id
                    FROM track_audio ta
                    WHERE ta.platform = 'youtube'
                )
                SELECT
                  base.track_audio_id   AS yt_track_audio_id,
                  base.path             AS yt_audio_path,
                  base.track_id         AS track_id,
                  tm.title              AS title,
                  tm.artists_json       AS artists_json,
                  tm.version_tag        AS version_tag,
                  tm.full_name          AS full_name,
                  (
                    SELECT m.set_id FROM dj_set_track_media_links m
                    WHERE m.track_id = base.track_id
                    ORDER BY (CASE WHEN m.set_id IN ({bb_csv}) THEN 0 ELSE 1 END), m.set_id
                    LIMIT 1
                  ) AS set_id,
                  (CASE WHEN EXISTS (
                    SELECT 1 FROM dj_set_track_media_links m
                    WHERE m.track_id = base.track_id AND m.set_id IN ({bb_csv})
                  ) THEN 1 ELSE 0 END) AS is_bb
                FROM base
                JOIN track_metadata tm ON tm.track_id = base.track_id
                WHERE tm.title IS NOT NULL AND tm.title != ''
                  {job_filter}
                ORDER BY is_bb DESC, set_id, base.track_audio_id
            """
            for r in conn.execute(replace_q).fetchall():
                out.append(_row_to_candidate(r))

        if mode in ("acquire", "all"):
            acquire_q = f"""
                WITH base AS (
                    SELECT tm.track_id
                    FROM track_metadata tm
                    LEFT JOIN track_audio ta ON ta.track_id = tm.track_id
                    WHERE ta.track_audio_id IS NULL
                      AND tm.title IS NOT NULL AND tm.title != ''
                )
                SELECT
                  NULL                  AS yt_track_audio_id,
                  NULL                  AS yt_audio_path,
                  base.track_id         AS track_id,
                  tm.title              AS title,
                  tm.artists_json       AS artists_json,
                  tm.version_tag        AS version_tag,
                  tm.full_name          AS full_name,
                  (
                    SELECT m.set_id FROM dj_set_track_media_links m
                    WHERE m.track_id = base.track_id
                    ORDER BY (CASE WHEN m.set_id IN ({bb_csv}) THEN 0 ELSE 1 END), m.set_id
                    LIMIT 1
                  ) AS set_id,
                  (CASE WHEN EXISTS (
                    SELECT 1 FROM dj_set_track_media_links m
                    WHERE m.track_id = base.track_id AND m.set_id IN ({bb_csv})
                  ) THEN 1 ELSE 0 END) AS is_bb
                FROM base
                JOIN track_metadata tm ON tm.track_id = base.track_id
                WHERE 1=1
                  {job_filter}
                ORDER BY is_bb DESC, set_id, base.track_id
            """
            for r in conn.execute(acquire_q).fetchall():
                out.append(_row_to_candidate(r))

    return tuple(out)


def _phase1_download(
    candidates: tuple[Candidate, ...],
    args: argparse.Namespace,
) -> tuple[RunStats, dict[str, int]]:
    """Pool yt-dlp downloads over `args.threads` workers. Returns (stats,
    {track_id → new track_audio_id}).
    """
    stats = RunStats(candidates=len(candidates))
    ok_map: dict[str, int] = {}
    objects_root = args.audio_root / "objects"

    items = tuple(
        ytmusic_adapter.BatchItem(track_id=c.track_id, query=c.query)
        for c in candidates
    )

    _log.info("Phase 1: ytmusic-search + yt-dlp, %d items, threads=%d, "
              "per-track-timeout=%.0fs", len(items), args.threads,
              args.timeout_per_track)

    # Slice into shards so we can log + dedupe per-shard, but workers
    # actually parallelize across each shard.
    shard = max(args.threads * 5, 20)
    t_start = time.monotonic()
    for i in range(0, len(items), shard):
        chunk = items[i : i + shard]
        chunk_cands = candidates[i : i + shard]
        t0 = time.monotonic()
        results = ytmusic_adapter.download_batch(
            chunk, objects_root, args.audio_format,
            threads=args.threads,
            timeout_s_per_track=args.timeout_per_track,
            cookies_path=args.cookies,
        )
        elapsed = time.monotonic() - t0

        ok_in_shard = 0
        for c, r in zip(chunk_cands, results):
            match r.result:
                case Ok(asset):
                    ins = db_adapter.insert_audio_or_reap(args.db, asset)
                    match ins:
                        case Ok(new_taid):
                            stats = dc_replace(stats, phase1_ok=stats.phase1_ok + 1)
                            ok_map[c.track_id] = new_taid
                            ok_in_shard += 1
                            supersedes = (str(c.yt_track_audio_id)
                                          if c.yt_track_audio_id is not None
                                          else "—")
                            _log.info("OK    %s -> %s (taid=%d, supersedes %s)",
                                      c.track_id, asset.path, new_taid, supersedes)
                        case Err(e):
                            stats = dc_replace(stats, phase1_failed_insert=stats.phase1_failed_insert + 1)
                            _log.error("DB    %s insert: %s", c.track_id, e.detail)
                case Err(err):
                    if "search failed" in (err.detail or "").lower() or err.kind == "unavailable":
                        stats = dc_replace(stats, phase1_failed_search=stats.phase1_failed_search + 1)
                    else:
                        stats = dc_replace(stats, phase1_failed_dl=stats.phase1_failed_dl + 1)
                    _log.warning("FAIL  %s [%s]: %s",
                                 c.track_id, err.kind, (err.detail or "")[:140])
        _log.info("shard %d-%d/%d: %d/%d ok in %.0fs (%.1fs/track wall)",
                  i + 1, i + len(chunk), len(items),
                  ok_in_shard, len(chunk), elapsed,
                  elapsed / max(len(chunk), 1))
    _log.info("Phase 1 total wall: %.0fs", time.monotonic() - t_start)
    return stats, ok_map


def _phase2_replace(
    candidates: tuple[Candidate, ...],
    ok_map: dict[str, int],
    audio_root: Path,
    db_path: Path,
    stats: RunStats,
) -> RunStats:
    """For each Phase-1 success, sanity-check the new file then DELETE the
    yt-dlp row (cascade kills downstream analysis tables). Unlink the
    yt-dlp m4a and the old stems dir.
    """
    _log.info("Phase 2: replacing %d yt-dlp rows", len(ok_map))
    stems_root = audio_root / "stems"

    for c in candidates:
        if not c.needs_replace:
            # ACQUIRE candidate — Phase 1 inserted a fresh row, nothing to
            # delete in Phase 2. (Counted in phase2_skipped to avoid
            # over-reporting "actual" failures.)
            stats = dc_replace(stats, phase2_skipped=stats.phase2_skipped + 1)
            continue
        new_taid = ok_map.get(c.track_id)
        if new_taid is None:
            stats = dc_replace(stats, phase2_skipped=stats.phase2_skipped + 1)
            continue

        # Sanity check: new file must exist and be reasonable size.
        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT path FROM track_audio WHERE track_audio_id = ?",
                    (new_taid,),
                ).fetchone()
        except sqlite3.DatabaseError as e:
            stats = dc_replace(stats, phase2_failed=stats.phase2_failed + 1)
            _log.error("[skip] %s lookup failed: %s", c.track_id, e)
            continue
        new_path = Path(row["path"]) if row and row["path"] else None
        if new_path is None or not new_path.is_file() or new_path.stat().st_size < 100_000:
            stats = dc_replace(stats, phase2_skipped=stats.phase2_skipped + 1)
            _log.warning("[skip-unsafe] %s new file missing/tiny (%s); "
                         "leaving yt-dlp row in place",
                         c.track_id, new_path)
            continue

        try:
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                conn.execute("DELETE FROM track_audio WHERE track_audio_id = ?",
                             (c.yt_track_audio_id,))
                conn.commit()

            yt_path = Path(c.yt_audio_path)
            if yt_path.is_file():
                yt_path.unlink()

            old_stems = stems_root / str(c.yt_track_audio_id)
            if old_stems.exists():
                shutil.rmtree(old_stems, ignore_errors=True)

            stats = dc_replace(stats, phase2_replaced=stats.phase2_replaced + 1)
            _log.info("replaced %s yt_taid=%d -> ytm_taid=%d",
                      c.track_id, c.yt_track_audio_id, new_taid)
        except (sqlite3.DatabaseError, OSError) as e:
            stats = dc_replace(stats, phase2_failed=stats.phase2_failed + 1)
            _log.error("replace failed for %s: %s", c.track_id, e)
    return stats


def _run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    job_set_ids: frozenset[str] | None = None
    if args.job_file is not None:
        try:
            rows = json.loads(args.job_file.read_text())
            ids = [r["tracklist_id"] for r in rows
                   if isinstance(r, dict) and r.get("tracklist_id")]
            job_set_ids = frozenset(ids)
            _log.info("job-file %s: %d set_ids", args.job_file, len(job_set_ids))
        except (OSError, json.JSONDecodeError, KeyError) as e:
            _log.error("failed to read job file %s: %s", args.job_file, e)
            return 1

    candidates = _load_candidates(args.db, args.mode, job_set_ids)
    if args.max_tracks is not None:
        candidates = candidates[: args.max_tracks]
    if not candidates:
        _log.info("no candidates")
        return 0

    bb_count = sum(1 for c in candidates if c.is_bb)
    replace_count = sum(1 for c in candidates if c.needs_replace)
    acquire_count = len(candidates) - replace_count
    _log.info("loaded %d candidates (mode=%s, replace=%d, acquire=%d, BB-first=%d, "
              "dry_run=%s, no_replace=%s)",
              len(candidates), args.mode, replace_count, acquire_count,
              bb_count, args.dry_run, args.no_replace)

    if args.dry_run:
        for c in candidates[:10]:
            kind = "REPL" if c.needs_replace else "ACQU"
            taid = str(c.yt_track_audio_id) if c.needs_replace else "—"
            _log.info("DRY  %s  yt_taid=%s  set=%s  bb=%d  vtag=%s  query=%r",
                      kind, taid, c.set_id, c.is_bb,
                      c.version_tag or "—", c.query)
        if len(candidates) > 10:
            _log.info("... and %d more", len(candidates) - 10)
        return 0

    t0 = time.monotonic()
    stats, ok_map = _phase1_download(candidates, args)
    _log.info("Phase 1 done in %.0fs: %d ok, %d search-fail, %d dl-fail, %d insert-fail",
              time.monotonic() - t0, stats.phase1_ok,
              stats.phase1_failed_search, stats.phase1_failed_dl,
              stats.phase1_failed_insert)

    if not args.no_replace and ok_map:
        t1 = time.monotonic()
        stats = _phase2_replace(candidates, ok_map, args.audio_root, args.db, stats)
        _log.info("Phase 2 done in %.0fs: %d replaced, %d skipped, %d failed",
                  time.monotonic() - t1,
                  stats.phase2_replaced, stats.phase2_skipped, stats.phase2_failed)

    _log.info(
        "DONE in %.0fs | candidates=%d phase1_ok=%d phase2_replaced=%d "
        "(failures: search=%d, dl=%d, insert=%d, replace=%d)",
        time.monotonic() - t0,
        stats.candidates, stats.phase1_ok, stats.phase2_replaced,
        stats.phase1_failed_search, stats.phase1_failed_dl,
        stats.phase1_failed_insert, stats.phase2_failed,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    return _run(_parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
