"""Entry point — download audio for a single DJ set.

Usage:
    python -m audio_pipeline.main --set-id 1d9zwh49                      # per-track mode (default)
    python -m audio_pipeline.main --set-id 1d9zwh49 --mode set           # download the full mix + build timeline
    python -m audio_pipeline.main --set-id 1d9zwh49 --mode both          # both
    python -m audio_pipeline.main --set-id 1d9zwh49 --limit 5            # dry-run a few tracks
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .adapters.downloader import DownloadConfig
from .adapters.spotdl_adapter import SpotdlConfig
from .pipeline import process_set
from .result import Err, Ok


DEFAULT_DB = Path("data/db/music_database.db")
DEFAULT_OUT = Path("data/audio")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--set-id", required=True)
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--format", default="m4a")
    p.add_argument("--limit", type=int, default=0, help="0 = no limit (track mode)")
    p.add_argument("--mode", choices=("track", "set", "both"), default="track",
                   help="'track' downloads each canonical track (default), 'set' downloads the "
                        "full-mix audio posted for this DJ set plus a timeline sidecar, 'both' does both")
    args = p.parse_args(argv)

    db_path = Path(args.db)
    out_dir = Path(args.out) / args.set_id
    dl_cfg = DownloadConfig(out_dir=out_dir, audio_format=args.format)
    sd_cfg = SpotdlConfig(out_dir=out_dir, audio_format=args.format)

    rc = 0

    if args.mode in ("set", "both"):
        from .set_pipeline import process_set
        print(f"[set-mode] downloading full mix for {args.set_id}")
        out = process_set(db_path, args.set_id, dl_cfg)
        if out.audio is not None:
            print(f"[set-ok]  {args.set_id} via {out.attempted} -> {out.audio.path}")
        elif out.last_error is None:
            print(f"[set-skip] {args.set_id} (already downloaded)")
        else:
            print(f"[set-err] {args.set_id} via {out.attempted}: {out.last_error}", file=sys.stderr)
            rc = 1
        if out.timeline is not None:
            print(f"[timeline] {args.set_id}: {len(out.timeline.segments)} segments")

    if args.mode in ("track", "both"):
        from .adapters import db as db_adapter
        from .pipeline import process_track
        tracks_r = db_adapter.load_set_tracks(db_path, args.set_id)
        match tracks_r:
            case Err(e):
                print(f"[db-error] {e.kind}: {e.detail}", file=sys.stderr)
                return 2
            case Ok(tracks):
                if args.limit > 0:
                    tracks = tracks[: args.limit]
                print(f"[plan] set_id={args.set_id} tracks={len(tracks)} out={out_dir}")

        n_ok = n_skip = n_err = 0
        for t in tracks:
            outcome = process_track(t, db_path, dl_cfg, sd_cfg)
            if outcome.success is not None:
                n_ok += 1
                print(f"[ok]   {outcome.track_id} -> {outcome.success.path}")
            elif outcome.last_error is None:
                n_skip += 1
                print(f"[skip] {outcome.track_id} (already in DB)")
            else:
                n_err += 1
                print(f"[err]  {outcome.track_id} via {outcome.attempted}: {outcome.last_error}")
        print(f"[done-tracks] ok={n_ok} skip={n_skip} err={n_err}")
        if n_err > 0:
            rc = 1

    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
