#!/usr/bin/env python3
"""Acquire a vocal/instrumental variant of a track from a URL.

v1 scope (this file): download -> lossless WAV into a staging folder with
slot-prefixed naming (so the clip identity survives the Ableton drag-in) and
append a provenance line to ``replacements.tsv``.

Next increment (v2, gated on pi-storage reachability so it can be tested):
canonical ingest -- write a ``track_audio`` row with ``variant_tag``, chromaprint
the variant for an identity sanity-check, and skip Essentia (vocals-only audio
has no intrinsic BPM/key). Reuse the canonical-write path from the sibling
``replace_track_audio.py`` rather than duplicating it.

Works with any yt-dlp-supported URL (YouTube, SoundCloud, ...).
"""
from __future__ import annotations

import argparse
import datetime as dt
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download an acapella/instrumental variant into the staging folder."
    )
    ap.add_argument("url")
    ap.add_argument("--role", required=True, help="acappella | instrumental")
    ap.add_argument("--name", required=True, help='"Artist - Title"')
    ap.add_argument("--slot", type=int, default=None, help="set position, e.g. 9 -> 009__")
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = ap.parse_args()

    variant_tag, display = norm_role(args.role)
    stem = basename(args.slot, args.name, display)
    out = download(args.url, args.dest, stem)
    log = log_provenance(args.dest, args.slot, variant_tag, args.name, args.url, out.name)

    print(f"\nsaved: {out}")
    print(f"variant_tag={variant_tag}  |  logged to {log}")


if __name__ == "__main__":
    main()
