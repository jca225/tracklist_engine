"""CLI: run full per-track analysis (stems + beats + cues + loudness + MERT sections)
for every downloaded track belonging to one DJ set.

    python -m audio_pipeline.analysis_main --set-id 2nvzlh2k

Idempotent: tracks with an existing `track_analysis` row are skipped. Failures
on individual tracks are logged and the batch continues.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

from .adapters import db as db_adapter
from .analysis import pipeline as ap
from .models import AudioAsset
from .result import Err, Ok


DEFAULT_DB = Path("data/db/music_database.db")
DEFAULT_STEMS_DIR = Path.home() / "Desktop" / "tracklist_audio_drive" / "stems"


def _load_pending(db_path: Path, set_id: str) -> list[AudioAsset]:
    """Return track_audio rows for this set that have no track_analysis row yet."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT DISTINCT ta.track_audio_id, ta.track_id, ta.platform, ta.source_url,
                            ta.player_id, ta.path, ta.sha256, ta.duration_s,
                            ta.sample_rate, ta.codec, ta.bitrate_kbps
            FROM track_audio ta
            JOIN dj_set_track_media_links l ON l.track_id = ta.track_id
            LEFT JOIN track_analysis tan ON tan.track_audio_id = ta.track_audio_id
            WHERE l.set_id = ? AND tan.track_audio_id IS NULL
            ORDER BY ta.track_audio_id
            """,
            (set_id,),
        ).fetchall()
    return [
        AudioAsset(
            track_audio_id=r["track_audio_id"], track_id=r["track_id"],
            platform=r["platform"], source_url=r["source_url"], player_id=r["player_id"],
            path=r["path"], sha256=r["sha256"], duration_s=r["duration_s"],
            sample_rate=r["sample_rate"], codec=r["codec"], bitrate_kbps=r["bitrate_kbps"],
        )
        for r in rows
    ]


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--set-id", required=True)
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--stems-dir", default=str(DEFAULT_STEMS_DIR))
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = p.parse_args(argv)

    db_path = Path(args.db)
    stems_dir = Path(args.stems_dir)
    stems_dir.mkdir(parents=True, exist_ok=True)

    pending = _load_pending(db_path, args.set_id)
    if args.limit > 0:
        pending = pending[: args.limit]
    if not pending:
        print(f"[done] no pending tracks for set {args.set_id}")
        return 0

    print(f"[plan] set={args.set_id} tracks={len(pending)} stems_dir={stems_dir}")
    print(f"[plan] est time on MPS: ~{len(pending) * 135 // 60} min")

    t0_all = time.time()
    print("[load] loading analyzers (first-run model downloads can take minutes)...")
    ar = ap.load_analyzers(device=args.device)
    match ar:
        case Err(e):
            print(f"[err] load_analyzers: {e}", file=sys.stderr)
            return 2
        case Ok(analyzers):
            pass
    print(f"[load] done in {time.time()-t0_all:.1f}s on device={analyzers.demucs.device}")

    n_ok = n_err = 0
    for i, asset in enumerate(pending, start=1):
        t0 = time.time()
        try:
            result = ap.analyze_track(analyzers, asset, stems_dir)
        except Exception as e:
            print(f"[{i:3d}/{len(pending)}] id={asset.track_audio_id} {asset.track_id} "
                  f"CRASH: {type(e).__name__}: {e}", flush=True)
            n_err += 1
            continue

        match result:
            case Err(err):
                print(f"[{i:3d}/{len(pending)}] id={asset.track_audio_id} {asset.track_id} "
                      f"FAIL: {err}", flush=True)
                n_err += 1
                continue
            case Ok(r):
                pass

        persist = db_adapter.persist_analysis(db_path, r)
        match persist:
            case Err(err):
                print(f"[{i:3d}/{len(pending)}] id={asset.track_audio_id} {asset.track_id} "
                      f"PERSIST FAIL: {err}", flush=True)
                n_err += 1
                continue
            case Ok(_):
                pass

        dt = time.time() - t0
        print(f"[{i:3d}/{len(pending)}] id={asset.track_audio_id} {asset.track_id} "
              f"bpm={r.beats.bpm:.1f} cues={len(r.cues.cue_times)} sections={len(r.sections)} "
              f"lufs={r.loudness.integrated_lufs:.1f} ({dt:.1f}s)",
              flush=True)
        n_ok += 1

    elapsed = time.time() - t0_all
    print(f"[done] ok={n_ok} err={n_err} total={len(pending)} elapsed={elapsed/60:.1f}min")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
