#!/usr/bin/env python3
"""Acquire a vocal/instrumental variant of a track from a URL or local file.

Two modes:

  Staging mode (default — for the Ableton manual-labeling workflow):
    download -> lossless WAV into a staging folder with slot-prefixed naming
    (so the clip identity survives the drag-in) and append a provenance line
    to ``replacements.tsv``. Needs --name; no DB touched.

  Canonical-ingest mode (when --track-id / --track-audio-id is given):
    acquire the audio, place it under the canonical objects/ store, and INSERT
    a ``track_audio`` row carrying ``variant_tag`` (acappella | instrumental)
    alongside the existing 'original' row (this ADDS a variant, it does NOT
    replace). Reuses the canonical-write path from the sibling
    ``replace_track_audio.py`` rather than duplicating it.

    Downstream gating on variant_tag is partial: cue-detr already runs only on
    variant_tag='original' (analysis/canonical_cues.py), so variants get no
    canonical cues. Essentia BPM/key does NOT yet gate on variant_tag, so an
    acappella variant would currently receive meaningless features — gating
    Essentia on variant_tag is a TODO (see the no-essentia-on-acapellas rule).

Works with any yt-dlp-supported URL (YouTube, SoundCloud, ...). Spotify URLs
route through spotdl in canonical mode.

TODO (follow-up): chromaprint the acquired variant against the track's
'original' fingerprint as an identity sanity-check before insert — needs a
fingerprint adapter (none exists in-repo yet) and the original audio present.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
YT_DLP = REPO_ROOT / "venvs" / "audio" / "bin" / "yt-dlp"
DEFAULT_DEST = Path(
    "/Users/johnnycabrahams/Desktop/big bootie 12 labeling Project/sourced"
)

# input role -> (canonical track_audio.variant_tag, display suffix)
_ROLES = {
    "acappella": ("acappella", "Acapella"),
    "acapella": ("acappella", "Acapella"),
    "vocals": ("acappella", "Acapella"),
    "instrumental": ("instrumental", "Instrumental"),
    "instr": ("instrumental", "Instrumental"),
}


def norm_role(role: str) -> tuple[str, str]:
    hit = _ROLES.get(role.strip().lower())
    if hit is None:
        sys.exit(f"unknown role {role!r}; use: acappella | instrumental")
    return hit


def basename(slot: int | None, name: str, display: str) -> str:
    prefix = f"{slot:03d}__" if slot is not None else ""
    return f"{prefix}{name.strip().replace('/', '-')} ({display})"


def download(url: str, dest: Path, stem: str) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(YT_DLP), "--no-playlist",
        "-f", "bestaudio/best",
        "-x", "--audio-format", "wav", "--audio-quality", "0",
        "-o", str(dest / f"{stem}.%(ext)s"), url,
    ]
    subprocess.run(cmd, check=True)
    out = dest / f"{stem}.wav"
    if not out.exists():
        sys.exit(f"expected {out} after download, not found")
    return out


def log_provenance(
    dest: Path, slot: int | None, variant_tag: str, name: str, url: str, filename: str
) -> Path:
    log = dest / "replacements.tsv"
    fresh = not log.exists()
    with log.open("a") as fh:
        if fresh:
            fh.write("slot\tvariant_tag\tname\turl\tfilename\tacquired_at\n")
        fh.write(
            f"{'' if slot is None else slot}\t{variant_tag}\t{name}\t{url}"
            f"\t{filename}\t{dt.datetime.now().isoformat(timespec='seconds')}\n"
        )
    return log


def staging_ingest(args: argparse.Namespace) -> int:
    """v1: download a WAV into the staging folder for manual Ableton labeling."""
    if not args.url:
        sys.exit("staging mode needs a URL (positional)")
    if not args.name:
        sys.exit('staging mode needs --name "Artist - Title"')
    variant_tag, display = norm_role(args.role)
    stem = basename(args.slot, args.name, display)
    out = download(args.url, args.dest, stem)
    log = log_provenance(args.dest, args.slot, variant_tag, args.name, args.url, out.name)
    print(f"\nsaved: {out}")
    print(f"variant_tag={variant_tag}  |  logged to {log}")
    return 0


def canonical_ingest(args: argparse.Namespace) -> int:
    """v2: acquire + INSERT a variant track_audio row in the canonical DB.

    Adds the variant alongside the existing 'original' (track_audio_id=None
    passed to the reused helpers => no delete/cascade).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    sys.path.insert(0, str(REPO_ROOT))
    from scripts import replace_track_audio as rta  # reuse canonical-write path

    variant_tag, _ = norm_role(args.role)

    track_id = args.track_id
    if track_id is None and args.track_audio_id is not None:
        track_id = rta._resolve_track_id_from_taid(args.db, args.track_audio_id)
        if track_id is None:
            sys.exit(f"track_audio_id {args.track_audio_id} not found in {args.db}")
    if track_id is None:
        sys.exit("canonical ingest needs --track-id or --track-audio-id")

    if args.url and args.file:
        sys.exit("--url and --file are mutually exclusive")
    if args.url:
        return rta._replace_via_url(
            args.db, args.audio_root, track_id, args.url,
            track_audio_id=None, variant_tag=variant_tag,
        )
    if args.file:
        pid = args.player_id or args.file.stem
        return rta._replace_via_file(
            args.db, args.audio_root, track_id, args.file, pid,
            track_audio_id=None, variant_tag=variant_tag,
        )
    sys.exit("canonical ingest needs --url or --file")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Acquire an acapella/instrumental variant (staging WAV or canonical ingest).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("url", nargs="?", help="yt-dlp/Spotify URL (omit for --file)")
    ap.add_argument("--role", required=True, help="acappella | instrumental")

    # Staging mode
    ap.add_argument("--name", help='"Artist - Title" (staging mode)')
    ap.add_argument("--slot", type=int, default=None, help="set position, e.g. 9 -> 009__ (staging)")
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="staging folder")

    # Canonical-ingest mode (presence of --track-id / --track-audio-id selects it)
    ap.add_argument("--track-id", default=None, help="canonical track_id to attach the variant to")
    ap.add_argument("--track-audio-id", type=int, default=None,
                    help="resolve track_id from this taid (canonical mode)")
    ap.add_argument("--file", type=Path, default=None, help="local audio file (canonical mode)")
    ap.add_argument("--player-id", default=None,
                    help="player_id for --file (defaults to filename stem)")
    ap.add_argument("--db", type=Path,
                    default=Path(os.environ.get("TRACKLIST_DB",
                                                "/mnt/storage/data/db/music_database.db")))
    ap.add_argument("--audio-root", type=Path,
                    default=Path(os.environ.get("TRACKLIST_AUDIO_ROOT", "/mnt/storage")))
    args = ap.parse_args()

    if args.track_id is not None or args.track_audio_id is not None:
        return canonical_ingest(args)
    return staging_ingest(args)


if __name__ == "__main__":
    sys.exit(main())
