"""Resolve source-track audio from a pulled ~/aligning tree.

Manifest ``track_id`` lookup is preferred, but BB12 (and similar) can have GT
rows whose recording_id is absent from the aligning manifest while the audio
still lives under ``tracks/{slot}__*`` / ``stems/{slot}__*``. Fall back through
slot-label + filesystem with name disambiguation — fail closed when multiple
slot folders exist and none match the span name.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from core.contracts import MANIFEST_FILENAME
from alignment.fp_placement_refine import find_aligning_dir

_STEM_FILE = {"acappella": "vocals", "instrumental": "instrumental"}
_AUDIO_EXT = (".flac", ".wav", ".m4a", ".mp3")


def resolve_aligning_audio(set_dir: Path, path_str: str | None) -> Path | None:
    """Map a manifest path onto an existing file under ``set_dir``."""
    if not path_str:
        return None
    raw = Path(path_str)
    if raw.is_file():
        return raw
    parts = raw.parts
    for marker in ("tracks", "stems"):
        if marker not in parts:
            continue
        i = parts.index(marker)
        candidate = set_dir.joinpath(*parts[i:])
        if candidate.is_file():
            return candidate
    by_name = set_dir / "tracks" / raw.name
    if by_name.is_file():
        return by_name
    return None


def _slot_aliases(slot: str) -> list[str]:
    """``37`` / ``037`` / ``10w2`` / ``010w2`` forms for globbing."""
    s = (slot or "").strip()
    if not s:
        return []
    out: list[str] = [s]
    m = re.match(r"^(\d+)(.*)$", s)
    if m:
        n, rest = m.group(1), m.group(2)
        for cand in (f"{int(n):03d}{rest}", f"{int(n)}{rest}"):
            if cand not in out:
                out.append(cand)
    return out


def _name_tokens(name: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]{3,}", (name or "").lower()) if t]


def _score_path(path: Path, *, name_hint: str, prefer_stem_file: str | None) -> int:
    blob = path.as_posix().lower()
    score = 0
    for tok in _name_tokens(name_hint):
        if tok in blob:
            score += 1
    if prefer_stem_file and path.name.startswith(prefer_stem_file):
        score += 10
    return score


def load_manifest_tracks(set_id: str) -> dict[str, dict]:
    """Indexes: track_id, recording_id, and slot_label → manifest track dict."""
    set_dir = find_aligning_dir(set_id)
    if set_dir is None:
        return {}
    manifest = set_dir / MANIFEST_FILENAME
    if not manifest.is_file():
        return {}
    doc = json.loads(manifest.read_text())
    by_id: dict[str, dict] = {}
    for t in doc.get("tracks") or []:
        if t.get("track_id"):
            by_id[str(t["track_id"])] = t
        if t.get("recording_id"):
            by_id.setdefault(str(t["recording_id"]), t)
        slot = str(t.get("slot_label") or "").strip()
        if slot:
            by_id.setdefault(f"slot:{slot}", t)
            for alias in _slot_aliases(slot):
                by_id.setdefault(f"slot:{alias}", t)
    return by_id


def _resolve_from_track(set_dir: Path, track: dict, gt_stem: str) -> Path | None:
    stem_key = _STEM_FILE.get(gt_stem or "regular")
    if stem_key:
        stem_path = (track.get("stems") or {}).get(stem_key)
        resolved = resolve_aligning_audio(set_dir, stem_path)
        if resolved is not None:
            return resolved
    return resolve_aligning_audio(set_dir, track.get("local_path"))


def _resolve_from_filesystem(
    set_dir: Path,
    slot: str,
    *,
    gt_stem: str,
    name_hint: str,
) -> Path | None:
    """Glob tracks/stems by slot; disambiguate with span name when needed."""
    prefer = _STEM_FILE.get(gt_stem or "regular")
    candidates: list[Path] = []
    for alias in _slot_aliases(slot):
        for folder in sorted(set_dir.glob(f"stems/{alias}__*")):
            if not folder.is_dir():
                continue
            if prefer:
                for ext in _AUDIO_EXT:
                    f = folder / f"{prefer}{ext}"
                    if f.is_file():
                        candidates.append(f)
        for f in sorted(set_dir.glob(f"tracks/{alias}__*")):
            if f.is_file() and f.suffix.lower() in _AUDIO_EXT:
                candidates.append(f)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    ranked = sorted(
        candidates,
        key=lambda p: _score_path(p, name_hint=name_hint, prefer_stem_file=prefer),
        reverse=True,
    )
    best = ranked[0]
    best_score = _score_path(best, name_hint=name_hint, prefer_stem_file=prefer)
    second = (
        _score_path(ranked[1], name_hint=name_hint, prefer_stem_file=prefer)
        if len(ranked) > 1
        else -1
    )
    if best_score == 0 and second == 0:
        return None
    if best_score == second:
        return None
    return best


def resolve_source_audio(
    set_id: str,
    recording_id: str,
    *,
    gt_stem: str,
    tracks: dict[str, dict] | None = None,
    slot: str | None = None,
    name: str | None = None,
) -> Path | None:
    """Prefer stem channel for acappella/instrumental; else full track.

    Resolution order:
    1. manifest by recording_id / track_id
    2. manifest by slot_label (zero-padded aliases)
    3. filesystem ``tracks|stems/{slot}__*`` with name disambiguation
    """
    set_dir = find_aligning_dir(set_id)
    if set_dir is None:
        return None
    tracks = tracks if tracks is not None else load_manifest_tracks(set_id)

    track_by_id = tracks.get(recording_id) if recording_id else None
    track = track_by_id
    if track is None and slot:
        for alias in _slot_aliases(slot):
            track = tracks.get(f"slot:{alias}")
            if track is not None:
                break
    if track is not None:
        resolved = _resolve_from_track(set_dir, track, gt_stem)
        if resolved is not None:
            return resolved
        # Manifest knows this recording_id but has no playable path (unresolved
        # inventory stub). Do NOT fall back to tracks/stems/{slot}__* — BB12
        # still has old annotator files under colliding slot numbers (e.g. 032
        # Lux Holm missing vs 032__AFROJACK… still on disk).
        if track_by_id is not None:
            return None

    if slot:
        return _resolve_from_filesystem(
            set_dir, slot, gt_stem=gt_stem, name_hint=name or ""
        )
    return None


def preflight_aligning_audio(set_id: str, gt_tracks: list[dict]) -> list[str]:
    """recording_ids with no resolvable audio (excluding unalignable/skip_training)."""
    tracks = load_manifest_tracks(set_id)
    missing: list[str] = []
    seen: set[str] = set()
    for row in gt_tracks:
        if row.get("unalignable") or row.get("skip_training"):
            continue
        if str(row.get("slot_label") or "") == "mix":
            continue
        rid = str(row.get("track_id") or "")
        if not rid or rid in seen:
            continue
        seen.add(rid)
        slot = str(row.get("slot_label") or "")
        gstem = row.get("claimed_stem") or "regular"
        name = str(row.get("name") or row.get("label") or "")
        if (
            resolve_source_audio(
                set_id,
                rid,
                gt_stem=gstem,
                tracks=tracks,
                slot=slot or None,
                name=name or None,
            )
            is None
        ):
            missing.append(rid)
    return missing


def run_audio_preflight(
    set_id: str, gt_tracks: list[dict], *, strict: bool = False
) -> int:
    """Print preflight summary; return 1 when *strict* and any ref audio is missing."""
    missing = preflight_aligning_audio(set_id, gt_tracks)
    print(f"[preflight] {len(missing)} missing ref audio (excluding unalignable)")
    for rid in missing[:20]:
        print(f"  missing: {rid}")
    if len(missing) > 20:
        print(f"  ... and {len(missing) - 20} more")
    return 1 if strict and missing else 0
