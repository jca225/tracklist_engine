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
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass, replace as dc_replace
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from audio_pipeline.adapters import db as db_adapter
from audio_pipeline.adapters import ytmusic_adapter
from audio_pipeline.errors import DownloadError
from audio_pipeline.result import Err, Ok

_log = logging.getLogger("redownload_via_ytmusic")


_BB_SETS: frozenset[str] = frozenset((
    "w1mgcjt", "2nvzlh2k", "1fsnxchk", "qj4v0wt", "1yl70ql1", "237tdqmk",
))


@dataclass(frozen=True)
class Candidate:
    """A yt-dlp-sourced track that has artist+title metadata available."""
    yt_track_audio_id: int
    yt_audio_path: str
    track_id: str
    title: str
    artists_csv: str               # 'Daft Punk' or 'Artist1, Artist2'
    set_id: str
    is_bb: int

    @property
    def query(self) -> str:
        if self.artists_csv:
            return f"{self.artists_csv} - {self.title}"
        return self.title


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
    p.add_argument("--audio-format", default="m4a")
    p.add_argument("--cookies", type=Path,
                   default=Path(os.environ["TRACKLIST_YT_COOKIES"])
                       if os.environ.get("TRACKLIST_YT_COOKIES") else None,
                   help="yt-dlp cookies for age-gated tracks.")
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p.parse_args(argv)


def _load_candidates(db_path: Path) -> tuple[Candidate, ...]:
    """Tracks where:
      - track_audio.platform = 'youtube' (the noisy scraped version)
      - track_metadata has a non-empty title (so we have something to search)
    Order: BB-first, then by set_id grouping (so Vast can re-analyze BB
    quickly).
    """
    bb_csv = ",".join(f"'{s}'" for s in _BB_SETS)
    query = f"""
        SELECT
          ta.track_audio_id   AS yt_track_audio_id,
          ta.path             AS yt_audio_path,
          ta.track_id         AS track_id,
          tm.title            AS title,
          tm.artists_json     AS artists_json,
          (
            SELECT m.set_id
            FROM dj_set_track_media_links m
            WHERE m.track_id = ta.track_id
            ORDER BY (CASE WHEN m.set_id IN ({bb_csv}) THEN 0 ELSE 1 END), m.set_id
            LIMIT 1
          ) AS set_id,
          (
            CASE WHEN EXISTS (
              SELECT 1 FROM dj_set_track_media_links m
              WHERE m.track_id = ta.track_id AND m.set_id IN ({bb_csv})
            ) THEN 1 ELSE 0 END
          ) AS is_bb
        FROM track_audio ta
        JOIN track_metadata tm ON tm.track_id = ta.track_id
        WHERE ta.platform = 'youtube'
          AND tm.title IS NOT NULL AND tm.title != ''
        ORDER BY is_bb DESC, set_id, ta.track_audio_id
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query).fetchall()
    out: list[Candidate] = []
    for r in rows:
        try:
            artists = json.loads(r["artists_json"]) if r["artists_json"] else []
        except (json.JSONDecodeError, TypeError):
            artists = []
        artists_csv = ", ".join(a for a in artists if a)
        out.append(Candidate(
            yt_track_audio_id=r["yt_track_audio_id"],
            yt_audio_path=r["yt_audio_path"],
            track_id=r["track_id"],
            title=r["title"],
            artists_csv=artists_csv,
            set_id=r["set_id"] or "",
            is_bb=r["is_bb"],
        ))
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
                    ins = db_adapter.insert_audio(args.db, asset)
                    match ins:
                        case Ok(new_taid):
                            stats = dc_replace(stats, phase1_ok=stats.phase1_ok + 1)
                            ok_map[c.track_id] = new_taid
                            ok_in_shard += 1
                            _log.info("OK    %s -> %s (taid=%d, supersedes %d)",
                                      c.track_id, asset.path, new_taid,
                                      c.yt_track_audio_id)
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
    candidates = _load_candidates(args.db)
    if args.max_tracks is not None:
        candidates = candidates[: args.max_tracks]
    if not candidates:
        _log.info("no candidates")
        return 0

    bb_count = sum(1 for c in candidates if c.is_bb)
    _log.info("loaded %d candidates (BB-first count=%d, dry_run=%s, no_replace=%s)",
              len(candidates), bb_count, args.dry_run, args.no_replace)

    if args.dry_run:
        for c in candidates[:10]:
            _log.info("DRY  yt_taid=%d  set=%s  bb=%d  query=%r",
                      c.yt_track_audio_id, c.set_id, c.is_bb, c.query)
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
