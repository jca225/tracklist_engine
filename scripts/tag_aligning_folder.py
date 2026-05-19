"""Inject BPM + Camelot key + feature comment into the iTunes-style tags
of every .m4a in an aligning folder, by reading the local manifest.json
and querying pi-storage for track_audio_features.

What gets written (for each track that has features):
  - tmpo            : BPM (integer, iTunes BPM atom — Ableton reads this)
  - ----:com.apple.iTunes:initialkey : Camelot key (e.g. "8A")
  - ©cmt (comment)  : "essentia bpm=97 key=4A energy=0.71 dnce=0.55 val=0.43 lufs=-9.1"
  - ©too / ©day     : left alone

Usage:
    ./venvs/audio/bin/python scripts/tag_aligning_folder.py \\
        ~/aligning/1fsnxchk__Two\\ Friends\\ -\\ Big\\ Bootie\\ Mix\\ Volume\\ 12
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from mutagen.mp4 import MP4

PI_HOST = "pi-storage"
PI_DB = "/mnt/storage/data/db/music_database.db"

# Camelot wheel: pitch class (0..11) + mode → Camelot string
KEY_PC_TO_CAMELOT_MAJOR = ["8B","3B","10B","5B","12B","7B","2B","9B","4B","11B","6B","1B"]
KEY_PC_TO_CAMELOT_MINOR = ["5A","12A","7A","2A","9A","4A","11A","6A","1A","8A","3A","10A"]


def to_camelot(pc: int | None, mode: str | None) -> str | None:
    if pc is None or pc < 0 or pc > 11:
        return None
    if mode == "major":
        return KEY_PC_TO_CAMELOT_MAJOR[pc]
    if mode == "minor":
        return KEY_PC_TO_CAMELOT_MINOR[pc]
    return None


def ssh_sqlite_json(query: str) -> list[dict]:
    script = f".mode json\n{query.strip()}\n"
    cmd = ["ssh", PI_HOST, f"sqlite3 {PI_DB}"]
    out = subprocess.run(cmd, input=script, capture_output=True, text=True, check=True)
    body = out.stdout.strip()
    return json.loads(body) if body else []


def fetch_features(audio_ids: list[int]) -> dict[int, dict]:
    """Return {track_audio_id: {bpm, key_camelot, energy, dance, valence, lufs}}.
    Prefer essentia_v2 for BPM/key/etc., fall back to audio_pipeline_v1 for LUFS.
    """
    if not audio_ids:
        return {}
    ids = ",".join(str(i) for i in audio_ids)
    rows = ssh_sqlite_json(f"""
        SELECT track_audio_id, source, bpm, key_pc, key_mode,
               energy, danceability, valence, lufs
        FROM track_audio_features
        WHERE track_audio_id IN ({ids});
    """)
    out: dict[int, dict] = {}
    for r in rows:
        taid = r["track_audio_id"]
        entry = out.setdefault(taid, {})
        src = r["source"]
        if src == "essentia_v2":
            entry.update({
                "bpm": r.get("bpm"),
                "key_camelot": to_camelot(r.get("key_pc"), r.get("key_mode")),
                "energy": r.get("energy"),
                "danceability": r.get("danceability"),
                "valence": r.get("valence"),
                "lufs": r.get("lufs"),  # may be None — fallback fills
            })
        else:  # audio_pipeline_v1
            # Only use as fallback for fields essentia didn't fill
            entry.setdefault("bpm", r.get("bpm"))
            entry.setdefault("lufs", r.get("lufs"))
    # Pull up audio_pipeline_v1 LUFS over essentia null
    for r in rows:
        if r["source"] == "audio_pipeline_v1":
            taid = r["track_audio_id"]
            if out[taid].get("lufs") is None and r.get("lufs") is not None:
                out[taid]["lufs"] = r["lufs"]
    return out


def fmt_comment(f: dict) -> str:
    parts = ["essentia"]
    if f.get("bpm") is not None:
        parts.append(f"bpm={f['bpm']:.0f}")
    if f.get("key_camelot"):
        parts.append(f"key={f['key_camelot']}")
    if f.get("energy") is not None:
        parts.append(f"energy={f['energy']:.2f}")
    if f.get("danceability") is not None:
        parts.append(f"dnce={f['danceability']:.2f}")
    if f.get("valence") is not None:
        parts.append(f"val={f['valence']:.2f}")
    if f.get("lufs") is not None:
        parts.append(f"lufs={f['lufs']:.1f}")
    return " ".join(parts)


def tag_one(local_path: Path, features: dict) -> str:
    """Write tags to one .m4a. Returns a one-line summary."""
    audio = MP4(local_path)
    bpm = features.get("bpm")
    key = features.get("key_camelot")
    if bpm is not None:
        audio["tmpo"] = [int(round(bpm))]
    if key:
        audio["----:com.apple.iTunes:initialkey"] = [key.encode("utf-8")]
    audio["\xa9cmt"] = [fmt_comment(features)]
    audio.save()
    return f"bpm={int(round(bpm)) if bpm is not None else '?'} key={key or '?'}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", help="Path to an aligning folder containing manifest.json")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be written")
    args = ap.parse_args()

    folder = Path(args.folder).expanduser()
    manifest_path = folder / "manifest.json"
    if not manifest_path.is_file():
        print(f"ERROR: no manifest.json in {folder}", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text())

    audio_ids = [t["track_audio_id"] for t in manifest["tracks"] if t.get("track_audio_id")]
    print(f"Querying features for {len(audio_ids)} tracks...")
    features_by_id = fetch_features(audio_ids)
    print(f"Found features for {len(features_by_id)} tracks.\n")

    tagged = 0
    skipped = 0
    missing = 0
    for t in manifest["tracks"]:
        local = Path(t.get("local_path", ""))
        taid = t.get("track_audio_id")
        if not local.is_file():
            print(f"  [skip] missing on disk: {local.name}")
            skipped += 1
            continue
        if local.suffix.lower() != ".m4a":
            print(f"  [skip] non-m4a: {local.name}")
            skipped += 1
            continue
        feats = features_by_id.get(taid)
        if not feats:
            print(f"  [no-features] {local.name}")
            missing += 1
            continue
        if args.dry_run:
            print(f"  [dry] {local.name} <- {fmt_comment(feats)}")
        else:
            summary = tag_one(local, feats)
            print(f"  [ok]  {local.name} <- {summary}")
        tagged += 1

    print(f"\nTagged: {tagged}, skipped: {skipped}, no-features: {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
