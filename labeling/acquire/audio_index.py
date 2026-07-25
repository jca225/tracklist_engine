"""Stable track_audio_id → local path index for pulled aligning folders."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import msgspec

from core.contracts import MANIFEST_FILENAME, load_manifest

INDEX_NAME = "audio_index.json"
INDEX_VERSION = 2

_SLOT_RE = re.compile(r"^(\d{3}(?:w\d+)?)")


def _slot_of(track: dict) -> str | None:
    for key in ("slot_label", "label"):
        raw = track.get(key)
        if raw:
            return str(raw)
    local = track.get("local_path") or ""
    m = _SLOT_RE.match(Path(local).stem)
    return m.group(1) if m else None


def _slot_variants(slot_label: str) -> list[str]:
    s = str(slot_label)
    out = [s]
    m = re.match(r"^0*(\d+)(.*)$", s)
    if m:
        for cand in (m.group(1) + m.group(2), m.group(1).zfill(3) + m.group(2)):
            if cand not in out:
                out.append(cand)
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _unique_track_file(set_dir: Path, slot_label: str) -> Path | None:
    tracks = set_dir / "tracks"
    if not tracks.is_dir():
        return None
    candidates = [
        hit
        for slot in _slot_variants(slot_label)
        for hit in sorted(tracks.glob(f"{slot}__*"))
        if hit.suffix.lower() != ".asd" and hit.is_file()
    ]
    # Deduplicate by resolved path (hardlinks / duplicate globs).
    by_res = {c.resolve(): c for c in candidates}
    if len(by_res) == 1:
        return next(iter(by_res.values()))
    return None


def _unique_stem_file(set_dir: Path, slot_label: str, stem_name: str) -> Path | None:
    stems_root = set_dir / "stems"
    if not stems_root.is_dir():
        return None
    candidates: dict[Path, Path] = {}
    for slot in _slot_variants(slot_label):
        for directory in sorted(stems_root.glob(f"{slot}__*")):
            candidate = directory / f"{stem_name}.flac"
            if candidate.is_file():
                candidates[candidate.resolve()] = candidate
    if len(candidates) == 1:
        return next(iter(candidates.values()))
    return None


def build_audio_index(set_dir: Path, tracks: list[dict]) -> dict:
    """Build an on-disk index keyed by track_audio_id.

    Prefers existing manifest paths; fills gaps only with unique slot matches.
    """
    set_dir = Path(set_dir)
    by_id: dict[str, dict] = {}
    for track in tracks:
        taid = track.get("track_audio_id")
        if taid is None:
            continue
        entry: dict = {
            "recording_id": track.get("recording_id") or track.get("track_id"),
            "stem": track.get("stem") or "regular",
            "variant": track.get("variant") or "regular",
            "canonical_path": track.get("pi_path"),
            "local_path": None,
            "local_sha256": None,
            "stems": {},
            "stem_sha256": {},
        }
        local = track.get("local_path")
        if local and Path(local).is_file():
            entry["local_path"] = str(Path(local))
        else:
            slot = _slot_of(track)
            if slot:
                hit = _unique_track_file(set_dir, slot)
                if hit is not None:
                    entry["local_path"] = str(hit)
        if entry["local_path"]:
            entry["local_sha256"] = _sha256(Path(entry["local_path"]))

        stems = track.get("stems") or {}
        for stem_name in ("vocals", "instrumental"):
            p = stems.get(stem_name)
            if p and Path(p).is_file():
                entry["stems"][stem_name] = str(Path(p))
                entry["stem_sha256"][stem_name] = _sha256(Path(p))
                continue
            slot = _slot_of(track)
            if not slot:
                continue
            hit = _unique_stem_file(set_dir, slot, stem_name)
            if hit is not None:
                entry["stems"][stem_name] = str(hit)
                entry["stem_sha256"][stem_name] = _sha256(hit)

        if entry["local_path"] or entry["stems"]:
            by_id[str(taid)] = entry
    return {"version": INDEX_VERSION, "by_track_audio_id": by_id}


def write_audio_index(set_dir: Path, index: dict) -> Path:
    path = Path(set_dir) / INDEX_NAME
    path.write_text(json.dumps(index, indent=2) + "\n")
    return path


def load_audio_index(set_dir: Path | None) -> dict:
    if not set_dir:
        return {"version": INDEX_VERSION, "by_track_audio_id": {}}
    path = Path(set_dir) / INDEX_NAME
    if not path.is_file():
        return {"version": INDEX_VERSION, "by_track_audio_id": {}}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"version": INDEX_VERSION, "by_track_audio_id": {}}
    if not isinstance(data, dict):
        return {"version": INDEX_VERSION, "by_track_audio_id": {}}
    by_id = data.get("by_track_audio_id")
    if not isinstance(by_id, dict):
        by_id = {}
    return {"version": int(data.get("version") or INDEX_VERSION), "by_track_audio_id": by_id}


def has_audio_index(set_dir: Path | None) -> bool:
    return bool(set_dir) and (Path(set_dir) / INDEX_NAME).is_file()


def lookup_ref(index: dict, track_audio_id: object) -> Path | None:
    if track_audio_id is None:
        return None
    entry = (index.get("by_track_audio_id") or {}).get(str(track_audio_id)) or {}
    p = entry.get("local_path")
    if p and Path(p).is_file():
        return Path(p)
    return None


def lookup_stem(index: dict, track_audio_id: object, stem_name: str) -> Path | None:
    if track_audio_id is None:
        return None
    entry = (index.get("by_track_audio_id") or {}).get(str(track_audio_id)) or {}
    p = (entry.get("stems") or {}).get(stem_name)
    if p and Path(p).is_file():
        return Path(p)
    return None


def refresh_audio_index(set_dir: Path) -> Path:
    """Rebuild audio_index.json from the folder's current manifest."""
    set_dir = Path(set_dir)
    manifest_path = set_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no {MANIFEST_FILENAME} in {set_dir}")
    manifest = load_manifest(manifest_path)
    tracks = [msgspec.to_builtins(row) for row in manifest.tracks]
    index = build_audio_index(set_dir, tracks)
    return write_audio_index(set_dir, index)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Rebuild audio_index.json for a pulled aligning folder"
    )
    ap.add_argument("set_dir", type=Path, help="Path containing the set manifest")
    args = ap.parse_args(argv)
    try:
        path = refresh_audio_index(args.set_dir.expanduser())
    except FileNotFoundError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    data = json.loads(path.read_text())
    n = len(data.get("by_track_audio_id") or {})
    print(f"wrote {path} ({n} track_audio_id entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
