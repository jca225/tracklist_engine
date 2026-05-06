"""One-shot: re-source yt-dlp `track_audio` rows via pooled spotdl.

Why:
  Raw YT URLs scraped from 1001tracklists frequently point to fan re-uploads,
  DJ live edits, or videos with intros / sponsor reads / talkover noise that
  contaminates downstream MIR features (MERT embeddings pick up the talkover,
  cue-detr fires on radio drops, etc.). spotdl seeded by a Spotify track ID
  resolves to the official mastered release on YT Music — clean track audio.

What:
  Two-phase replacement for the ~1,946 yt-dlp tracks that have a Spotify URL
  available in `dj_set_track_media_links`:

  Phase 1 (additive, safe): pooled spotdl downloads. Each success inserts a
  new track_audio row with platform='spotify'. Coexists with the existing
  yt-dlp row temporarily — that's deliberate, so a spotdl failure doesn't
  leave a track audioless.

  Phase 2 (destructive): for tracks where Phase 1 inserted a spotify row,
  delete the yt-dlp track_audio row by track_audio_id. ON DELETE CASCADE
  removes its track_analysis, track_stems, track_mert_measures,
  track_audio_features. Also unlinks the on-disk m4a and the stems dir.

Side effect on Vast: the 42 yt-dlp tracks that were already analyzed lose
their track_analysis row when we delete the yt-dlp track_audio. The new
spotify track_audio_id has no analysis, so Vast's next_task() picks it up
on the next pass. Net result: the 42 already-analyzed tracks get
re-analyzed (~63 min on Vast post-FLAC+pipeline).

Usage:
  # Smoke test (no deletes, no downloads — just print what would happen)
  venvs/audio/bin/python -m scripts.redownload_via_spotdl --dry-run \\
      --max-tracks 5

  # Real run, 5 tracks for sanity check
  venvs/audio/bin/python -m scripts.redownload_via_spotdl --max-tracks 5

  # Full run on pi-storage in tmux
  venvs/audio/bin/python -m scripts.redownload_via_spotdl

Skips Phase 2 with --no-replace if you want to leave the yt-dlp rows in
place (e.g. for A/B comparison of audio sources during validation).
"""
from __future__ import annotations

import argparse
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
from audio_pipeline.adapters import spotdl_adapter
from audio_pipeline.errors import DownloadError
from audio_pipeline.models import MediaSource, spotify_track_url
from audio_pipeline.result import Err, Ok

_log = logging.getLogger("redownload_via_spotdl")


@dataclass(frozen=True)
class Candidate:
    """A yt-dlp-sourced track that has a Spotify URL available for re-download."""
    yt_track_audio_id: int       # the track_audio row we're replacing
    yt_audio_path: str           # /mnt/storage/objects/<tid>/<tid>__youtube__<vid>.m4a
    track_id: str                # 1001tracklists canonical track_id
    spotify_player_id: str       # Spotify track ID for spotdl


@dataclass(frozen=True)
class RunStats:
    candidates: int = 0
    phase1_ok: int = 0
    phase1_failed_to_dl: int = 0
    phase1_failed_to_insert: int = 0
    phase2_replaced: int = 0
    phase2_skipped: int = 0       # spotify row missing → skip the delete
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
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--timeout", type=float, default=60.0,
                   help="Per-track spotdl timeout, scaled by batch_size/threads internally.")
    p.add_argument("--max-tracks", type=int, default=None,
                   help="Cap candidates for smoke testing.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print candidates and intended actions without "
                        "calling spotdl, inserting, or deleting.")
    p.add_argument("--no-replace", action="store_true",
                   help="Skip Phase 2 (destructive yt-dlp row deletion). "
                        "Leaves both rows in DB; useful for A/B comparison.")
    p.add_argument("--audio-format", default="m4a")
    p.add_argument("--client-id", default=os.environ.get("SPOTIFY_CLIENT_ID"))
    p.add_argument("--client-secret", default=os.environ.get("SPOTIFY_CLIENT_SECRET"))
    p.add_argument("--log-level", default="INFO",
                   choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p.parse_args(argv)


def _load_candidates(db_path: Path) -> tuple[Candidate, ...]:
    """Tracks where:
      - track_audio.platform = 'youtube'
      - dj_set_track_media_links has a non-empty spotify entry for the track_id
    Picks ONE spotify player_id per track (first by lex order — deterministic).
    """
    query = """
        SELECT
          ta.track_audio_id     AS yt_track_audio_id,
          ta.path               AS yt_audio_path,
          ta.track_id           AS track_id,
          (
            SELECT m.player_id
            FROM dj_set_track_media_links m
            WHERE m.track_id = ta.track_id
              AND m.platform = 'spotify'
              AND m.player_id IS NOT NULL AND m.player_id != ''
            ORDER BY m.player_id LIMIT 1
          ) AS spotify_player_id
        FROM track_audio ta
        WHERE ta.platform = 'youtube'
          AND EXISTS (
            SELECT 1 FROM dj_set_track_media_links m
            WHERE m.track_id = ta.track_id
              AND m.platform = 'spotify'
              AND m.player_id IS NOT NULL AND m.player_id != ''
          )
        ORDER BY ta.track_audio_id
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query).fetchall()
    return tuple(
        Candidate(
            yt_track_audio_id=r["yt_track_audio_id"],
            yt_audio_path=r["yt_audio_path"],
            track_id=r["track_id"],
            spotify_player_id=r["spotify_player_id"],
        )
        for r in rows
    )


def _phase1_download(
    candidates: tuple[Candidate, ...],
    args: argparse.Namespace,
) -> tuple[RunStats, dict[str, int]]:
    """Pooled spotdl download of all candidates. Returns (stats, ok_map).

    ok_map: {track_id → new spotify track_audio_id} for tracks Phase 2 should
    treat as "spotify row exists, safe to delete the yt-dlp row".
    """
    stats = RunStats(candidates=len(candidates))
    ok_map: dict[str, int] = {}
    objects_root = args.audio_root / "objects"

    # Per-batch timeout scaled like main_retry's _run_batched.
    batch_timeout = max(args.timeout * 2, (args.batch_size / args.threads) * args.timeout * 2)
    _log.info("Phase 1: pooled spotdl, %d candidates, batch=%d threads=%d batch_timeout=%.0fs",
              len(candidates), args.batch_size, args.threads, batch_timeout)

    batches = [candidates[i : i + args.batch_size]
               for i in range(0, len(candidates), args.batch_size)]
    for bi, batch in enumerate(batches, 1):
        items = tuple(
            spotdl_adapter.BatchItem(
                track_id=c.track_id,
                source=MediaSource(
                    platform="spotify",
                    player_id=c.spotify_player_id,
                    url=spotify_track_url(c.spotify_player_id),
                ),
            )
            for c in batch
        )
        t_batch = time.monotonic()
        results = spotdl_adapter.download_batch(
            items, objects_root, args.audio_format,
            threads=args.threads, timeout_s=batch_timeout,
            client_id=args.client_id, client_secret=args.client_secret,
        )
        elapsed = time.monotonic() - t_batch

        ok_in_batch = 0
        for c, r in zip(batch, results):
            match r.result:
                case Ok(asset):
                    ins = db_adapter.insert_audio(args.db, asset)
                    match ins:
                        case Ok(new_taid):
                            stats = dc_replace(stats, phase1_ok=stats.phase1_ok + 1)
                            ok_map[c.track_id] = new_taid
                            ok_in_batch += 1
                            _log.info("[%d] OK    %s spotify:%s -> %s (taid=%d, supersedes %d)",
                                      bi, c.track_id, c.spotify_player_id,
                                      asset.path, new_taid, c.yt_track_audio_id)
                        case Err(e):
                            stats = dc_replace(stats, phase1_failed_to_insert=stats.phase1_failed_to_insert + 1)
                            _log.error("[%d] DB    %s insert_audio: %s",
                                       bi, c.track_id, e.detail)
                case Err(err):
                    stats = dc_replace(stats, phase1_failed_to_dl=stats.phase1_failed_to_dl + 1)
                    _log.warning("[%d] FAIL  %s %s: %s",
                                 bi, c.track_id, err.kind, (err.detail or "")[:120])
        _log.info("batch %d/%d: %d/%d ok in %.0fs (%.1fs/track wall)",
                  bi, len(batches), ok_in_batch, len(batch), elapsed,
                  elapsed / max(len(batch), 1))
    return stats, ok_map


def _phase2_replace(
    candidates: tuple[Candidate, ...],
    ok_map: dict[str, int],
    audio_root: Path,
    db_path: Path,
    stats: RunStats,
) -> RunStats:
    """For each track where Phase 1 inserted a spotify row, delete the yt-dlp
    track_audio row (cascades to analysis tables) and unlink files on disk.

    Stems: track_stems rows are deleted by cascade. The on-disk stems dir
    /mnt/storage/stems/<old_track_audio_id>/ also needs manual rmtree —
    cascade only handles DB.
    """
    _log.info("Phase 2: replacing %d yt-dlp rows with their spotify replacements",
              len(ok_map))
    stems_root = audio_root / "stems"

    for c in candidates:
        if c.track_id not in ok_map:
            stats = dc_replace(stats, phase2_skipped=stats.phase2_skipped + 1)
            _log.debug("[skip] %s no spotify replacement (Phase 1 did not insert)",
                       c.track_id)
            continue

        try:
            # 1. Delete the yt-dlp DB row → cascade kills downstream tables.
            with sqlite3.connect(db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON;")
                conn.execute("DELETE FROM track_audio WHERE track_audio_id = ?",
                             (c.yt_track_audio_id,))
                conn.commit()

            # 2. Unlink the yt-dlp m4a file.
            yt_path = Path(c.yt_audio_path)
            if yt_path.is_file():
                yt_path.unlink()

            # 3. Remove the old stems directory keyed by the deleted track_audio_id.
            old_stems = stems_root / str(c.yt_track_audio_id)
            if old_stems.exists():
                shutil.rmtree(old_stems, ignore_errors=True)

            stats = dc_replace(stats, phase2_replaced=stats.phase2_replaced + 1)
            _log.info("replaced track_id=%s yt_taid=%d -> spotify_taid=%d",
                      c.track_id, c.yt_track_audio_id, ok_map[c.track_id])
        except (sqlite3.DatabaseError, OSError) as e:
            stats = dc_replace(stats, phase2_failed=stats.phase2_failed + 1)
            _log.error("replace failed for %s (taid=%d): %s",
                       c.track_id, c.yt_track_audio_id, e)
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
        _log.info("no candidates — every yt-dlp track without a spotify alt or already replaced")
        return 0

    _log.info("loaded %d candidates (db=%s, dry_run=%s, no_replace=%s)",
              len(candidates), args.db, args.dry_run, args.no_replace)

    if args.dry_run:
        for c in candidates[:10]:
            _log.info("DRY  yt_taid=%d  track=%s  spotify=%s  yt_path=%s",
                      c.yt_track_audio_id, c.track_id, c.spotify_player_id,
                      c.yt_audio_path)
        if len(candidates) > 10:
            _log.info("... and %d more", len(candidates) - 10)
        _log.info("DRY total: %d Phase 1 downloads + (if --no-replace not set) "
                  "%d Phase 2 deletes", len(candidates), len(candidates))
        return 0

    t0 = time.monotonic()
    stats, ok_map = _phase1_download(candidates, args)
    _log.info("Phase 1 done in %.0fs: %d/%d ok, %d dl-fail, %d insert-fail",
              time.monotonic() - t0,
              stats.phase1_ok, stats.candidates,
              stats.phase1_failed_to_dl, stats.phase1_failed_to_insert)

    if not args.no_replace and ok_map:
        t1 = time.monotonic()
        stats = _phase2_replace(candidates, ok_map, args.audio_root, args.db, stats)
        _log.info("Phase 2 done in %.0fs: %d replaced, %d skipped, %d failed",
                  time.monotonic() - t1,
                  stats.phase2_replaced, stats.phase2_skipped, stats.phase2_failed)
    elif args.no_replace:
        _log.info("Phase 2 skipped (--no-replace). Spotify rows coexist with yt-dlp rows.")

    _log.info(
        "DONE in %.0fs | candidates=%d phase1_ok=%d phase2_replaced=%d "
        "(failures: dl=%d, insert=%d, replace=%d)",
        time.monotonic() - t0,
        stats.candidates, stats.phase1_ok, stats.phase2_replaced,
        stats.phase1_failed_to_dl, stats.phase1_failed_to_insert, stats.phase2_failed,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    return _run(_parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
