from __future__ import annotations

import argparse
from pathlib import Path

from downloader.pipeline import download_from_music_db
from downloader.storage import resolve_output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Spotify/YouTube/SoundCloud tracks from the music database."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/db/music_database.db"),
        help="Path to SQLite music database.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Root directory for downloaded files. "
            "If omitted, uses HARD_DRIVE_OUTPUT_ROOT from downloader/constants.py."
        ),
    )
    parser.add_argument(
        "--spotify-downloader-dir",
        type=Path,
        default=Path("spotify-downloader"),
        help="Local spotDL repository directory.",
    )
    parser.add_argument(
        "--cue-detr-dir",
        type=Path,
        default=Path("cue-detr"),
        help="Local cue-detr repository directory.",
    )
    parser.add_argument(
        "--cue-detr-checkpoint",
        type=str,
        default="disco-eth/cue-detr",
        help="cue-detr checkpoint (HF repo id or local checkpoint path).",
    )
    parser.add_argument(
        "--cue-detr-sensitivity",
        type=float,
        default=0.9,
        help="cue-detr peak sensitivity threshold.",
    )
    parser.add_argument(
        "--cue-detr-radius",
        type=int,
        default=16,
        help="Minimum distance between cue points in bars.",
    )
    parser.add_argument(
        "--skip-demucs",
        action="store_true",
        help="Skip Demucs stem extraction.",
    )
    parser.add_argument(
        "--demucs-model",
        type=str,
        default="htdemucs",
        help="Demucs model name.",
    )
    parser.add_argument(
        "--demucs-device",
        type=str,
        default="mps",
        help="Demucs device (mps/cpu/cuda).",
    )
    parser.add_argument(
        "--demucs-segment",
        type=int,
        default=7,
        help="Demucs segment size.",
    )
    parser.add_argument(
        "--demucs-overlap",
        type=float,
        default=0.25,
        help="Demucs overlap value.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout for URL canonicalization.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of unique canonical URLs to process.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without running downloader commands.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = resolve_output_root(args.output_root)
    print(f"Using output root: {output_root}")

    stats = download_from_music_db(
        db_path=args.db_path,
        output_root=output_root,
        spotify_downloader_dir=args.spotify_downloader_dir,
        cue_detr_dir=args.cue_detr_dir,
        cue_detr_checkpoint=args.cue_detr_checkpoint,
        cue_detr_sensitivity=args.cue_detr_sensitivity,
        cue_detr_radius=args.cue_detr_radius,
        run_demucs=not args.skip_demucs,
        demucs_model=args.demucs_model,
        demucs_device=args.demucs_device,
        demucs_segment=args.demucs_segment,
        demucs_overlap=args.demucs_overlap,
        timeout=args.timeout,
        limit=args.limit,
        dry_run=args.dry_run,
    )

    print("\nSummary")
    print(f"loaded_rows={stats['loaded_rows']}")
    print(f"processed={stats['processed']}")
    print(f"downloaded={stats['downloaded']}")
    print(f"failed={stats['failed']}")
    print(f"skipped_duplicate={stats['skipped_duplicate']}")
    print(f"skipped_unknown={stats['skipped_unknown']}")
    print(f"analyzed={stats['analyzed']}")
    print(f"analysis_failed={stats['analysis_failed']}")
    print(f"stems_generated={stats['stems_generated']}")
    print(f"stems_skipped={stats['stems_skipped']}")
    print(f"stems_failed={stats['stems_failed']}")
    return 0 if stats["failed"] == 0 else 1
