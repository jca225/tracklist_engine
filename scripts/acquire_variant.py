#!/usr/bin/env python3
"""Acquire a vocal/instrumental variant of a track from a URL or local file.

Two modes:

  Staging mode (default — for the Ableton manual-labeling workflow):
    download -> lossless WAV into a staging folder with slot-prefixed naming
    (so the clip identity survives the drag-in) and append a provenance line
    to ``replacements.tsv``. Needs --name; no DB touched.

  Canonical-ingest mode (when --track-id / --track-audio-id is given):
    acquire the audio, place it under the canonical objects/ store, and INSERT
    a ``track_audio`` row carrying ``stem`` (acappella | instrumental)
    alongside the existing ``regular`` row (this ADDS a sibling, it does NOT
    replace). Reuses the canonical-write path from the sibling
    ``replace_track_audio.py`` rather than duplicating it.

    Downstream gating on ``stem`` is partial: cue-detr runs only on
    ``stem='regular'`` (analysis/canonical_cues.py). Essentia BPM/key gates on
    ``stem='regular'`` in analysis/pipeline.py.

Works with any yt-dlp-supported URL (YouTube, SoundCloud, ...). Spotify URLs
route through spotdl in canonical mode.

Canonical mode runs an advisory chromaprint check (_identity_check) after
insert when a regular reference row exists; it never hard-blocks.
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

# input role -> (canonical track_audio.stem, display suffix)
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
    dest: Path, slot: int | None, stem: str, name: str, url: str, filename: str
) -> Path:
    log = dest / "replacements.tsv"
    fresh = not log.exists()
    with log.open("a") as fh:
        if fresh:
            fh.write("slot\tstem\tname\turl\tfilename\tacquired_at\n")
        fh.write(
            f"{'' if slot is None else slot}\t{stem}\t{name}\t{url}"
            f"\t{filename}\t{dt.datetime.now().isoformat(timespec='seconds')}\n"
        )
    return log


def staging_ingest(args: argparse.Namespace) -> int:
    """v1: download a WAV into the staging folder for manual Ableton labeling."""
    if not args.url:
        sys.exit("staging mode needs a URL (positional)")
    if not args.name:
        sys.exit('staging mode needs --name "Artist - Title"')
    stem_axis, display = norm_role(args.role)
    fname = basename(args.slot, args.name, display)
    out = download(args.url, args.dest, fname)
    log = log_provenance(args.dest, args.slot, stem_axis, args.name, args.url, out.name)
    print(f"\nsaved: {out}")
    print(f"stem={stem_axis}  |  logged to {log}")
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

    stem_axis, _ = norm_role(args.role)

    track_id = args.track_id
    if track_id is None and args.track_audio_id is not None:
        track_id = rta._resolve_track_id_from_taid(args.db, args.track_audio_id)
        if track_id is None:
            sys.exit(f"track_audio_id {args.track_audio_id} not found in {args.db}")
    if track_id is None:
        sys.exit("canonical ingest needs --track-id or --track-audio-id")

    if args.url and args.file:
        sys.exit("--url and --file are mutually exclusive")
    promote = not args.no_promote_reference
    if args.url:
        rc = rta._replace_via_url(
            args.db, args.audio_root, track_id, args.url,
            track_audio_id=None, stem=stem_axis,
            promote_reference=promote,
        )
    elif args.file:
        pid = args.player_id or args.file.stem
        rc = rta._replace_via_file(
            args.db, args.audio_root, track_id, args.file, pid,
            track_audio_id=None, stem=stem_axis,
            promote_reference=promote,
        )
    else:
        sys.exit("canonical ingest needs --url or --file")

    if rc == 0:
        _identity_check(args.db, track_id, stem_axis)
        if not args.no_log:
            _log_to_ledger(args, track_id, stem_axis)
    return rc


def _log_to_ledger(args: argparse.Namespace, track_id: str, stem_axis: str) -> None:
    """Append an additive (stem-axis) correction row (non-fatal)."""
    from core.result import Err, Ok
    from ingest.corrections import Correction, latest_row, log_correction

    new = latest_row(args.db, track_id, stem_axis)
    position = None if args.slot is None else str(args.slot)
    c = Correction(
        track_id=track_id, axis="stem", action="add",
        set_id=args.set_id, position=position,
        new_track_audio_id=(new or {}).get("track_audio_id"),
        new_platform=(new or {}).get("platform"),
        new_player_id=(new or {}).get("player_id"),
        new_url=(new or {}).get("source_url"),
        stem_value=stem_axis,
        reason=args.reason, source="acquire_variant",
    )
    match log_correction(args.db, c):
        case Ok(cid):
            print(f"logged correction_id={cid} (stem/add)")
        case Err(e):
            print(f"correction log failed (non-fatal): {e.kind} — {e.detail}")


def _lookup_audio_path(db_path: Path, track_id: str, stem_axis: str) -> tuple[int, str] | None:
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT track_audio_id, path FROM track_audio "
            "WHERE recording_id = ? AND stem = ? "
            "ORDER BY is_reference DESC, downloaded_at DESC LIMIT 1",
            (track_id, stem_axis),
        ).fetchone()
    return (int(row[0]), row[1]) if row else None


def _identity_check(db_path: Path, track_id: str, stem_axis: str) -> None:
    """Advisory: compare the just-acquired variant against the track's
    'original' via chromaprint and print a verdict. Never blocks the insert —
    this is a manual annotator aid; calibrate thresholds before hard-gating.
    """
    from core.result import Ok, Err
    from ingest.adapters import fingerprint as fp

    var = _lookup_audio_path(db_path, track_id, stem_axis)
    orig = _lookup_audio_path(db_path, track_id, "regular")
    if var is None:
        print("identity-check: variant row not found post-insert — skipping")
        return
    if orig is None:
        print(f"identity-check: WARNING — no 'original' present for track {track_id}.")
        print("  · the variant has no Essentia-feature source (variants don't get their own BPM/key),")
        print("  · and the chromaprint identity check can't run.")
        print("  -> download the regular version first (normal ingest) so the variant can inherit its features.")
        return

    fa = fp.fingerprint_file(orig[1])
    fb = fp.fingerprint_file(var[1])
    match (fa, fb):
        case (Ok(a), Ok(b)):
            sim = fp.similarity(a.raw, b.raw)
            dur_ratio = (b.duration_s / a.duration_s) if a.duration_s else 0.0
            verdict, detail = fp.classify(stem_axis, sim, dur_ratio)
            print(f"identity-check [{verdict}]: {detail}")
            print(f"  similarity={sim:.3f}  variant={b.duration_s:.1f}s  original={a.duration_s:.1f}s  ratio={dur_ratio:.2f}")
        case (Err(e), _) | (_, Err(e)):
            print(f"identity-check: skipped (fingerprint failed: {e.kind} — {e.detail})")


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
    # Correction-ledger fields (canonical mode)
    ap.add_argument("--set-id", default=None, help="set where noticed (correction ledger)")
    ap.add_argument("--reason", default=None, help="free-text why (correction ledger)")
    ap.add_argument("--no-log", action="store_true", help="skip the correction-ledger row")
    ap.add_argument("--no-promote-reference", action="store_true",
                    help="Do not set is_reference=1 on the new row (default: promote).")
    args = ap.parse_args()

    if args.track_id is not None or args.track_audio_id is not None:
        return canonical_ingest(args)
    return staging_ingest(args)


if __name__ == "__main__":
    sys.exit(main())
