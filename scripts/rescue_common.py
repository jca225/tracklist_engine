"""Shared helpers for redownload_via_spotdl / redownload_via_ytmusic."""
from __future__ import annotations

import logging
import shutil
import sqlite3
import time
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

from core.db import connect

# BB10-15 set IDs — tracks appearing in any of these get redownloaded first so
# Vast can start re-analyzing them quickly instead of waiting for the global
# queue to roll past.
BB_SETS: frozenset[str] = frozenset((
    "w1mgcjt", "2nvzlh2k", "1fsnxchk", "qj4v0wt", "1yl70ql1", "237tdqmk",
))

TStats = TypeVar("TStats")


def bb_set_ids_sql() -> str:
    return ",".join(f"'{s}'" for s in BB_SETS)


def _needs_replace(candidate: object) -> bool:
    nr = getattr(candidate, "needs_replace", True)
    return nr() if callable(nr) else bool(nr)


def phase2_replace(
    candidates: Iterable[Any],
    ok_map: dict[str, int],
    audio_root: Path,
    db_path: Path,
    stats: TStats,
    log: logging.Logger,
    *,
    replacement_label: str = "new",
) -> TStats:
    """For each Phase-1 success, sanity-check the new file then DELETE the
    old yt-dlp row (cascade kills downstream analysis tables). Unlink the
    old m4a and stems dir."""
    log.info("Phase 2: replacing %d rows", len(ok_map))
    stems_root = audio_root / "stems"

    for c in candidates:
        if not _needs_replace(c):
            stats = dc_replace(stats, phase2_skipped=stats.phase2_skipped + 1)
            continue
        new_taid = ok_map.get(c.track_id)
        if new_taid is None:
            stats = dc_replace(stats, phase2_skipped=stats.phase2_skipped + 1)
            continue

        try:
            with connect(db_path) as conn:
                row = conn.execute(
                    "SELECT path FROM track_audio WHERE track_audio_id = ?",
                    (new_taid,),
                ).fetchone()
        except sqlite3.DatabaseError as e:
            stats = dc_replace(stats, phase2_failed=stats.phase2_failed + 1)
            log.error("[skip] %s lookup failed: %s", c.track_id, e)
            continue
        new_path = Path(row["path"]) if row and row["path"] else None
        if new_path is None or not new_path.is_file() or new_path.stat().st_size < 100_000:
            stats = dc_replace(stats, phase2_skipped=stats.phase2_skipped + 1)
            log.warning(
                "[skip-unsafe] %s %s file missing/tiny (%s); leaving old row in place",
                c.track_id, replacement_label, new_path,
            )
            continue

        old_taid = c.yt_track_audio_id
        old_path_str = c.yt_audio_path
        try:
            with connect(db_path) as conn:
                conn.execute(
                    "DELETE FROM track_audio WHERE track_audio_id = ?",
                    (old_taid,),
                )
                conn.commit()

            if old_path_str:
                yt_path = Path(old_path_str)
                if yt_path.is_file():
                    yt_path.unlink()

            old_stems = stems_root / str(old_taid)
            if old_stems.exists():
                shutil.rmtree(old_stems, ignore_errors=True)

            stats = dc_replace(stats, phase2_replaced=stats.phase2_replaced + 1)
            log.info(
                "replaced track_id=%s yt_taid=%s -> %s_taid=%d",
                c.track_id, old_taid, replacement_label, new_taid,
            )
        except (sqlite3.DatabaseError, OSError) as e:
            stats = dc_replace(stats, phase2_failed=stats.phase2_failed + 1)
            log.error("replace failed for %s (taid=%s): %s", c.track_id, old_taid, e)
    return stats


def run_two_phase(
    *,
    candidates: tuple[Any, ...],
    args: Any,
    phase1_fn: Callable[[tuple[Any, ...], Any], tuple[TStats, dict[str, int]]],
    stats_cls: type[TStats],
    log: logging.Logger,
    phase2_replacement_label: str,
    phase1_failure_fields: tuple[str, ...],
) -> int:
    """Shared orchestration: Phase 1 download → optional Phase 2 replace."""
    if not candidates:
        return 0

    t0 = time.monotonic()
    stats, ok_map = phase1_fn(candidates, args)
    phase1_ok = getattr(stats, "phase1_ok")
    failures = " ".join(
        f"{name}={getattr(stats, name)}"
        for name in phase1_failure_fields
        if hasattr(stats, name)
    )
    log.info(
        "Phase 1 done in %.0fs: %d/%d ok%s",
        time.monotonic() - t0,
        phase1_ok,
        stats.candidates,
        f" ({failures})" if failures else "",
    )

    if not args.no_replace and ok_map:
        t1 = time.monotonic()
        stats = phase2_replace(
            candidates, ok_map, args.audio_root, args.db, stats, log,
            replacement_label=phase2_replacement_label,
        )
        log.info(
            "Phase 2 done in %.0fs: %d replaced, %d skipped, %d failed",
            time.monotonic() - t1,
            stats.phase2_replaced,
            stats.phase2_skipped,
            stats.phase2_failed,
        )
    elif getattr(args, "no_replace", False):
        log.info("Phase 2 skipped (--no-replace). New rows coexist with old ones.")

    fail_parts = ", ".join(
        f"{name.split('_')[-1]}={getattr(stats, name)}"
        for name in (*phase1_failure_fields, "phase2_failed")
        if hasattr(stats, name)
    )
    log.info(
        "DONE in %.0fs | candidates=%d phase1_ok=%d phase2_replaced=%d (%s)",
        time.monotonic() - t0,
        stats.candidates,
        stats.phase1_ok,
        stats.phase2_replaced,
        fail_parts,
    )
    return 0
