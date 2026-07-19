#!/usr/bin/env python3
"""Warm the per-mix fingerprint cache ahead of the corpus harvest.

Streaming (memory-bounded) builds, resumable (skips cached), parallel-safe — run
several shards concurrently. Selects the same eligible mixes as the harvest.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Callable, Sequence

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from workspaces.pws_aligner.corpus_harvest import query_corpus_slots  # noqa: E402
from workspaces.pws_aligner.mix_fp_store import load_or_build  # noqa: E402

DEFAULT_CACHE_ROOT = Path("/mnt/storage/data/mix_fp_cache")


def distinct_mixes(slots: Sequence) -> list[tuple[int, str]]:
    """Distinct ``(set_audio_id, mix_full_path)`` over the slots (one per set)."""
    seen: dict[int, str] = {}
    for s in slots:
        seen.setdefault(int(s.set_audio_id), str(s.mix_full_path))
    return list(seen.items())


def warm_cache(
    mixes: Sequence[tuple[int, str]],
    cache_root: str | Path,
    *,
    build: Callable[..., object] | None = None,
) -> tuple[int, int, int]:
    """Build+persist a fingerprint per mix; skip cached. Returns (built, skipped, failed)."""
    built = skipped = failed = 0
    for set_audio_id, mix_path in mixes:
        cache_file = Path(cache_root) / f"{set_audio_id}.fp"
        if cache_file.is_file() and cache_file.stat().st_size > 0:
            skipped += 1
            continue
        try:
            load_or_build(cache_root, str(set_audio_id), mix_path, build=build)
            built += 1
        except Exception as exc:  # undecodable mix etc. — count, don't crash the batch
            print(f"FAILED set_audio_id={set_audio_id}: {exc}", file=sys.stderr)
            failed += 1
    return built, skipped, failed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--db",
        default="/mnt/storage/data/db/music_database.db",
        help="canonical DB path (file:...?immutable=1 for read-only NFS)",
    )
    ap.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    ap.add_argument("--stem", default="regular")
    ap.add_argument(
        "--set-ids-file", default=None, help="restrict to set_ids one-per-line (shard)"
    )
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    set_ids = None
    if args.set_ids_file:
        set_ids = [
            ln.strip()
            for ln in Path(args.set_ids_file).read_text().splitlines()
            if ln.strip()
        ]

    conn = sqlite3.connect(args.db, uri=args.db.startswith("file:"))
    conn.row_factory = sqlite3.Row
    try:
        slots = query_corpus_slots(
            conn, policy_stems=(args.stem,), limit=args.limit, set_ids=set_ids
        )
    finally:
        conn.close()

    mixes = distinct_mixes(slots)
    built, skipped, failed = warm_cache(mixes, args.cache_root)
    print(
        f"mixes={len(mixes)} built={built} skipped={skipped} failed={failed} "
        f"cache_root={args.cache_root}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
