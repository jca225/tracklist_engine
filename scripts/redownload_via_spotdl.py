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
import sys
import time
from dataclasses import dataclass, replace as dc_replace
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core import db as db_adapter
from core.db import connect
from scripts.rescue_common import bb_set_ids_sql, run_two_phase
from ingest.adapters import spotdl_adapter
from ingest.errors import DownloadError
from core.models import MediaSource, spotify_track_url
from core.result import Err, Ok

_log = logging.getLogger("redownload_via_spotdl")


@dataclass(frozen=True)
class Candidate:
    """A yt-dlp-sourced track that has a Spotify URL available for re-download."""
    yt_track_audio_id: int       # the track_audio row we're replacing
    yt_audio_path: str           # /mnt/storage/objects/<tid>/<tid>__youtube__<vid>.m4a
    track_id: str                # 1001tracklists canonical track_id
    spotify_player_id: str       # Spotify track ID for spotdl
    set_id: str                  # any DJ set this track appears in (used for ordering)
    is_bb: int                   # 1 if track appears in any BB10-15 set, 0 otherwise


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

    Output ordering: BB tracks first (so Vast can pick up re-analysis quickly),
    then by set_id grouping (so tracks from the same DJ set tend to land in
    the same spotdl batch — minor spotipy/metadata locality benefits), then
    by track_audio_id within a set.

    Picks ONE spotify player_id per track and ONE set_id per track
    (deterministic, lex-first) — a track that appears in many sets just
    inherits whichever set comes first alphabetically. This deduplicates
    the candidate list (one row per unique track_audio_id).
    """
    bb_csv = bb_set_ids_sql()
    query = f"""
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
          ) AS spotify_player_id,
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
        WHERE ta.platform = 'youtube'
          AND EXISTS (
            SELECT 1 FROM dj_set_track_media_links m
            WHERE m.track_id = ta.track_id
              AND m.platform = 'spotify'
              AND m.player_id IS NOT NULL AND m.player_id != ''
          )
        ORDER BY is_bb DESC, set_id, ta.track_audio_id
    """
    with connect(db_path) as conn:
        rows = conn.execute(query).fetchall()
    return tuple(
        Candidate(
            yt_track_audio_id=r["yt_track_audio_id"],
            yt_audio_path=r["yt_audio_path"],
            track_id=r["track_id"],
            spotify_player_id=r["spotify_player_id"],
            set_id=r["set_id"] or "",
            is_bb=r["is_bb"],
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
                    ins = db_adapter.insert_audio_or_reap(args.db, asset)
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
        bb_count = sum(1 for c in candidates if c.is_bb)
        _log.info("DRY  ordering: BB tracks first (count=%d), then by set_id, "
                  "then by track_audio_id", bb_count)
        for c in candidates[:10]:
            _log.info("DRY  yt_taid=%d  set=%s  bb=%d  track=%s  spotify=%s",
                      c.yt_track_audio_id, c.set_id, c.is_bb,
                      c.track_id, c.spotify_player_id)
        if len(candidates) > 10:
            _log.info("... and %d more", len(candidates) - 10)
        _log.info("DRY total: %d Phase 1 downloads + (if --no-replace not set) "
                  "%d Phase 2 deletes", len(candidates), len(candidates))
        return 0

    return run_two_phase(
        candidates=candidates,
        args=args,
        phase1_fn=_phase1_download,
        stats_cls=RunStats,
        log=_log,
        phase2_replacement_label="spotify",
        phase1_failure_fields=("phase1_failed_to_dl", "phase1_failed_to_insert"),
    )


def main(argv: list[str] | None = None) -> int:
    return _run(_parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
