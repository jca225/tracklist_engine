"""CLI: run beat_this + Demucs analysis on one DJ set's full mix.

    python -m audio_pipeline.analysis_set_main --set-id 2nvzlh2k

Idempotent: skips the set if a set_analysis row already exists.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

from .adapters import db as db_adapter
from .analysis import pipeline as ap
from .analysis.set_analysis import analyze_set
from .models import SetAudioAsset
from .result import Err, Ok


DEFAULT_DB = Path("data/db/music_database.db")
DEFAULT_STEMS_DIR = Path.home() / "Desktop" / "tracklist_audio_drive" / "stems"


def _load_set_asset(db_path: Path, set_id: str) -> SetAudioAsset | None:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT sa.*
            FROM set_audio sa
            LEFT JOIN set_analysis san ON san.set_audio_id = sa.set_audio_id
            WHERE sa.set_id = ? AND san.set_audio_id IS NULL
            ORDER BY sa.is_reference DESC, sa.downloaded_at DESC LIMIT 1
            """,
            (set_id,),
        ).fetchone()
    if row is None:
        return None
    return SetAudioAsset(
        set_audio_id=row["set_audio_id"], set_id=row["set_id"],
        platform=row["platform"], source_url=row["source_url"],
        path=row["path"], sha256=row["sha256"], duration_s=row["duration_s"],
        sample_rate=row["sample_rate"], codec=row["codec"],
        bitrate_kbps=row["bitrate_kbps"],
    )


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--set-id", required=True)
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--stems-dir", default=str(DEFAULT_STEMS_DIR))
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    db_path = Path(args.db)
    stems_dir = Path(args.stems_dir)
    stems_dir.mkdir(parents=True, exist_ok=True)

    asset = _load_set_asset(db_path, args.set_id)
    if asset is None:
        print(f"[done] set {args.set_id} is already analyzed (or no set_audio row)")
        return 0

    print(f"[plan] set={args.set_id} asset={asset.path} duration={asset.duration_s:.0f}s")
    t0 = time.time()
    print("[load] loading analyzers...")
    ar = ap.load_analyzers(device=args.device)
    match ar:
        case Err(e):
            print(f"[err] load_analyzers: {e}", file=sys.stderr)
            return 2
        case Ok(analyzers):
            pass
    print(f"[load] done in {time.time()-t0:.1f}s on device={analyzers.demucs.device}")

    t0 = time.time()
    result_r = analyze_set(analyzers, asset, stems_dir)
    match result_r:
        case Err(e):
            print(f"[err] analyze_set: {e}", file=sys.stderr)
            return 3
        case Ok(result):
            pass
    print(
        f"[ok] set_audio_id={result.set_audio_id} "
        f"beats={len(result.beats.beat_times)} "
        f"downbeats={len(result.beats.downbeat_times)} "
        f"bpm={result.beats.bpm:.1f} "
        f"stems={[s.stem_name for s in result.stems.stems]} "
        f"({time.time()-t0:.1f}s)"
    )

    persist = db_adapter.persist_set_analysis(db_path, result)
    match persist:
        case Err(e):
            print(f"[err] persist: {e}", file=sys.stderr)
            return 4
        case Ok(_):
            print("[persist] ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
