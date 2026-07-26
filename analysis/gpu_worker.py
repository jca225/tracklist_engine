"""GPU worker — analyze_track on one audio file at a time.

Runs on a GPU box rented + managed by `gpubox` (the only sanctioned way to
rent GPUs here; gpubox rents from Vast.ai under the hood). Designed for the
corpus-scale analysis loop:
  1. Caller provides one audio file + the track_audio_id it corresponds to
  2. We load (cached) analyzers, run the full pipeline (Demucs + beat_this
     + cue-detr + per-measure MERT + Essentia), and persist results to the
     given SQLite DB
  3. Exit 0 on success; nonzero on error with stderr describing what failed

Resumable design — the worker exits after one track. The orchestrator
loops it. State lives in the DB, so interruption is safe.

Stems are written to `<stems-dir>/<track_audio_id>/{vocals,drums,bass,
other,instrumental}.<ext>`. The DB row references those paths, so the
caller is responsible for keeping the stems-dir reachable to consumers
(mount pi-storage there, or rsync after each track).

CLI:
    python -m analysis.gpu_worker \\
        --audio /mnt/audio/X.m4a \\
        --track-audio-id 999 \\
        --db /mnt/pi-storage/data/db/music_database.db \\
        --stems-dir /mnt/pi-storage/stems

For batch / loop mode (poll DB for next unanalyzed track until empty):
    python -m analysis.gpu_worker \\
        --db /mnt/pi-storage/data/db/music_database.db \\
        --audio-root /mnt/pi-storage \\
        --stems-dir /mnt/pi-storage/stems \\
        --loop
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

from core import db as db_adapter
from .pipeline import Analyzers, analyze_track, load_analyzers
from . import persistence
from core.models import AudioAsset
from core.result import Err, Ok, Result

_log = logging.getLogger("analysis.gpu_worker")

_BIG_BOOTIE_10_15: tuple[str, ...] = (
    "w1mgcjt",
    "2nvzlh2k",
    "1fsnxchk",
    "qj4v0wt",
    "1yl70ql1",
    "237tdqmk",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--db",
        type=Path,
        required=True,
        help="SQLite DB path (canonical or sshfs-mounted pi-storage DB)",
    )
    p.add_argument(
        "--stems-dir",
        type=Path,
        required=True,
        help="Where Demucs stems go. Layout: <stems-dir>/<track_audio_id>/<stem>.<ext>",
    )
    p.add_argument(
        "--audio",
        type=Path,
        help="Single audio file to analyze (paired with --track-audio-id)",
    )
    p.add_argument(
        "--track-audio-id",
        type=int,
        help="track_audio_id matching --audio (DB foreign key)",
    )
    p.add_argument(
        "--audio-root",
        type=Path,
        help="In --loop mode: the root that <track_audio.path> is relative to or "
        "where audio is mirrored. Used as a join prefix when track_audio.path "
        "still references the obsolete laptop drive.",
    )
    p.add_argument(
        "--loop",
        action="store_true",
        help="Poll DB for next unanalyzed track and process repeatedly until empty",
    )
    p.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="Stop after analyzing N tracks (smoke testing)",
    )
    p.add_argument(
        "--set-ids",
        type=str,
        default=None,
        help="Comma-separated set_ids: only process tracks that "
        "appear in any of these sets. Useful for prioritizing "
        "a small subset (e.g. Big Bootie 10-15) before "
        "draining the rest of the corpus.",
    )
    p.add_argument(
        "--bb-only",
        action="store_true",
        help="Shortcut for --set-ids matching the 6 Big Bootie 10-15 set_ids.",
    )
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    p.add_argument(
        "--separator",
        default="demucs",
        choices=("demucs", "uvr", "roformer"),
        help="Stem-separation backend (default: demucs).",
    )
    p.add_argument(
        "--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR")
    )
    return p.parse_args(argv)


def _next_unanalyzed(
    db_path: Path,
    set_ids: tuple[str, ...] | None = None,
    exclude: frozenset[int] | None = None,
) -> Result[tuple[int, str] | None, str]:
    """Return (track_audio_id, audio_path) for the next track that has
    a track_audio row but no track_analysis row. When set_ids is non-empty,
    restrict to tracks that appear in any of those sets (via
    set_track_slots, the canonical per-set spine — NOT
    dj_set_track_media_links, whose scrape-only rows silently drop gap
    tracks that were manually ingested without a scraped media link).
    ``exclude`` quarantines track_audio_ids that
    already failed this run so a poison track can't be re-selected forever
    (it has no track_analysis row, so the IS NULL predicate keeps returning
    it otherwise — the infinite-spin class fixed in the loop driver). Returns
    None when the queue is drained.
    """
    exclude = exclude or frozenset()
    excl_params = tuple(exclude)
    excl_clause = (
        f" AND ta.track_audio_id NOT IN ({','.join('?' * len(excl_params))})"
        if excl_params
        else ""
    )
    try:
        conn = sqlite3.connect(db_path)
        try:
            if set_ids:
                placeholders = ",".join("?" * len(set_ids))
                row = conn.execute(
                    f"""
                    SELECT ta.track_audio_id, ta.path
                    FROM track_audio ta
                    LEFT JOIN track_analysis tan
                      ON tan.track_audio_id = ta.track_audio_id
                    WHERE tan.track_audio_id IS NULL
                      AND ta.track_id IN (
                          SELECT DISTINCT track_id FROM set_track_slots
                          WHERE set_id IN ({placeholders})
                            AND track_id IS NOT NULL
                      ){excl_clause}
                    ORDER BY ta.track_audio_id
                    LIMIT 1
                    """,
                    set_ids + excl_params,
                ).fetchone()
            else:
                row = conn.execute(
                    f"""
                    SELECT ta.track_audio_id, ta.path
                    FROM track_audio ta
                    LEFT JOIN track_analysis tan
                      ON tan.track_audio_id = ta.track_audio_id
                    WHERE tan.track_audio_id IS NULL{excl_clause}
                    ORDER BY ta.track_audio_id
                    LIMIT 1
                    """,
                    excl_params,
                ).fetchone()
            return Ok((int(row[0]), str(row[1])) if row else None)
        finally:
            conn.close()
    except sqlite3.DatabaseError as e:
        return Err(f"db query failed: {e}")


def _load_audio_asset(db_path: Path, track_audio_id: int) -> Result[AudioAsset, str]:
    """Reconstruct an AudioAsset from the existing track_audio row. We
    only need a few fields populated for analyze_track; the heavy
    metadata is preserved as-is by the existing row."""
    try:
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(
                """
                SELECT track_audio_id, track_id, platform, source_url, player_id, path,
                       sha256, duration_s, sample_rate, codec, bitrate_kbps
                FROM track_audio WHERE track_audio_id = ?
                """,
                (track_audio_id,),
            ).fetchone()
            if row is None:
                return Err(f"no track_audio row for track_audio_id={track_audio_id}")
        finally:
            conn.close()
    except sqlite3.DatabaseError as e:
        return Err(f"db query failed: {e}")

    return Ok(
        AudioAsset(
            track_audio_id=int(row[0]),
            track_id=str(row[1]),
            platform=str(row[2]),
            source_url=str(row[3]) if row[3] else "",
            player_id=str(row[4]) if row[4] else "",
            path=str(row[5]),
            sha256=str(row[6]) if row[6] else None,
            duration_s=float(row[7]) if row[7] is not None else None,
            sample_rate=int(row[8]) if row[8] is not None else None,
            codec=str(row[9]) if row[9] else None,
            bitrate_kbps=int(row[10]) if row[10] is not None else None,
        )
    )


def _process_one(
    a: Analyzers,
    db_path: Path,
    asset: AudioAsset,
    audio_override_path: Path | None,
    stems_dir: Path,
) -> Result[float, str]:
    """Run analyze_track + persist_analysis for one track. Returns elapsed
    seconds on success."""
    if audio_override_path is not None:
        # Replace the asset path so analyze_track reads the actual file we have.
        from dataclasses import replace

        asset = replace(asset, path=str(audio_override_path))
    if not Path(asset.path).exists():
        return Err(f"audio file missing: {asset.path}")

    per_track_stems = stems_dir / str(asset.track_audio_id)
    per_track_stems.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    r = analyze_track(a, asset, stems_dir=per_track_stems)
    if not r.is_ok():
        return Err(f"analyze_track failed: {r.error.kind} — {r.error.detail}")

    p = persistence.persist_analysis(db_path, r.value)
    if not p.is_ok():
        return Err(f"persist_analysis failed: {p.error.kind} — {p.error.detail}")
    return Ok(time.time() - t0)


def _run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.loop:
        if args.audio is None or args.track_audio_id is None:
            _log.error("non-loop mode requires --audio and --track-audio-id")
            return 2

    _log.info(
        "loading analyzers (device=%s, separator=%s)…", args.device, args.separator
    )
    t0 = time.time()
    ar = load_analyzers(device=args.device, separator=args.separator)
    if not ar.is_ok():
        _log.error("load_analyzers failed: %s", ar.error)
        return 1
    a = ar.value
    _log.info(
        "analyzers loaded in %.1fs (with_essentia=%s)",
        time.time() - t0,
        a.with_essentia,
    )

    if not args.loop:
        # Single-track mode
        ar2 = _load_audio_asset(args.db, args.track_audio_id)
        if not ar2.is_ok():
            _log.error(ar2.error)
            return 1
        r = _process_one(a, args.db, ar2.value, args.audio, args.stems_dir)
        if not r.is_ok():
            _log.error(r.error)
            return 1
        _log.info("OK track_audio_id=%s in %.1fs", args.track_audio_id, r.value)
        return 0

    # Resolve --set-ids / --bb-only into a tuple
    set_filter: tuple[str, ...] | None = None
    if args.bb_only:
        set_filter = _BIG_BOOTIE_10_15
    elif args.set_ids:
        set_filter = tuple(s.strip() for s in args.set_ids.split(",") if s.strip())
    if set_filter:
        _log.info("filtering to %d set_ids: %s", len(set_filter), ",".join(set_filter))

    # Loop mode: drain unanalyzed tracks one at a time.
    # failed_tids quarantines tracks that error this run. Without it, a track
    # that never lands a track_analysis row (bad audio, poison file) is re-
    # selected by _next_unanalyzed every iteration — an infinite spin, the
    # exact class that burned 496 consecutive failures in the loop driver before
    # it grew the same skip set.
    n_done = 0
    failed_tids: set[int] = set()
    while args.max_tracks is None or n_done < args.max_tracks:
        nxt = _next_unanalyzed(args.db, set_filter, frozenset(failed_tids))
        if not nxt.is_ok():
            _log.error(nxt.error)
            return 1
        if nxt.value is None:
            _log.info(
                "no unanalyzed tracks remain — done (processed %d, quarantined %d)",
                n_done,
                len(failed_tids),
            )
            return 0
        track_audio_id, audio_path = nxt.value

        # If audio_path doesn't resolve and we have an audio-root override, try it
        resolved = Path(audio_path)
        if not resolved.exists() and args.audio_root is not None:
            # Strip the original prefix and join with our root.
            # track_audio.path looks like /mnt/storage/objects/<track_id>/<file>
            for candidate_prefix in ("/mnt/storage/", "/mnt/pi-storage/"):
                if audio_path.startswith(candidate_prefix):
                    rel = audio_path[len(candidate_prefix) :]
                    candidate = args.audio_root / rel
                    if candidate.exists():
                        resolved = candidate
                        break

        ar2 = _load_audio_asset(args.db, track_audio_id)
        if not ar2.is_ok():
            _log.warning("skip track_audio_id=%s: %s", track_audio_id, ar2.error)
            failed_tids.add(track_audio_id)
            continue
        r = _process_one(a, args.db, ar2.value, resolved, args.stems_dir)
        if not r.is_ok():
            _log.warning("FAIL track_audio_id=%s: %s", track_audio_id, r.error)
            # Continue rather than abort — one bad file shouldn't kill the loop —
            # but quarantine it so we don't re-select the same poison track.
            failed_tids.add(track_audio_id)
            n_done += 1
            continue
        _log.info(
            "OK [%d] track_audio_id=%s in %.1fs", n_done + 1, track_audio_id, r.value
        )
        n_done += 1

    _log.info("hit --max-tracks=%d, stopping", args.max_tracks)
    return 0


def main(argv: list[str] | None = None) -> int:
    return _run(_parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
