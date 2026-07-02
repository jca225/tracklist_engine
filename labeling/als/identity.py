"""Identity side of the `.als` codec: clip paths → slot / stem / manifest identity.

The `.als` is the canonical identity oracle: the file the human placed decides
the stem and display label; `track_id` is filled only on an exact manifest
path match (pull inventory), never from scrape slot or title guessing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from core.identity import normalize_stem

from labeling.als.models import ManifestIndex, ManifestSlot, ParsedClip
from labeling.als.tags import strip_user_tags

_SLOT_FROM_PATH = re.compile(r"(?:^|[/\\])(\d{3}(?:w\d+)?)__")


def slot_from_path(path: str) -> str | None:
    m = _SLOT_FROM_PATH.search(path)
    return m.group(1) if m else None


def _normalize_path(path: str) -> str:
    if not path:
        return ""
    return str(Path(path.replace("\\", "/")).expanduser())


def _stem_folder_name(path: str) -> str | None:
    """Return the ``tracks/`` or ``stems/`` child folder name, if any."""
    parts = Path(path.replace("\\", "/")).parts
    for idx, part in enumerate(parts):
        if part in ("tracks", "stems") and idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def _tagless_path(path: str) -> str:
    """Path with annotator bracket tags stripped from every component.

    The annotator renames files/subdirs with ``[NNNbpm KK]`` tags on the Mac
    only; ``manifest.json`` keeps canonical (un-tagged) names. Comparing paths
    tag-stripped lets a tagged clip path resolve to its un-tagged manifest row.
    """
    parts = Path(path.replace("\\", "/")).parts
    if not parts:
        return ""
    return str(Path(*(strip_user_tags(part) or part for part in parts)))


def build_manifest_index(manifest_path: Path) -> ManifestIndex:
    payload = json.loads(manifest_path.read_text())
    by_slot: dict[str, ManifestSlot] = {}
    by_path: dict[str, ManifestSlot] = {}
    rows: list[ManifestSlot] = []
    for row in payload.get("tracks") or []:
        local_path = str(row.get("local_path") or "")
        slot = str(row.get("label") or "").strip() or (slot_from_path(local_path) or "")
        if not slot and not local_path:
            continue
        artist = str(row.get("artist") or "").strip()
        title = str(row.get("title") or "").strip()
        version = row.get("version_tag")
        display = f"{artist} - {title}"
        if version:
            display = f"{display} ({version})"
        slot_row = ManifestSlot(
            slot_label=slot,
            track_id=str(row.get("track_id") or "").strip() or None,
            display=display,
            local_path=local_path,
        )
        rows.append(slot_row)
        if slot:
            by_slot[slot] = slot_row
        if local_path:
            by_path[_normalize_path(local_path)] = slot_row
    return ManifestIndex(by_slot=by_slot, by_path=by_path, rows=tuple(rows))


def match_manifest_for_path(path: str, manifest: ManifestIndex) -> ManifestSlot | None:
    """Exact manifest row for an ALS clip path (file or same stems folder only).

    No label guessing — the ALS path is canonical; manifest is a pull inventory
    used only when the clip points at the exact file (or stem tree) we synced.
    "Exact" is tag-insensitive: annotator ``[NNNbpm KK]`` renames are stripped
    from both sides before comparison (they never reach the manifest).
    """
    norm = _normalize_path(path)
    if norm in manifest.by_path:
        return manifest.by_path[norm]

    # Annotator [NNNbpm KK] renames live only on the Mac; the manifest keeps
    # canonical names. Every tier below compares tag-stripped so a tagged clip
    # path still resolves (BB11 GT export: 0/127 rows matched before this).
    tagless = _tagless_path(norm)
    for key, row in manifest.by_path.items():
        if _tagless_path(key) == tagless:
            return row

    folder = _stem_folder_name(path)
    if folder:
        folder = strip_user_tags(folder) or folder
        for row in manifest.rows:
            row_folder = _stem_folder_name(row.local_path) if row.local_path else None
            if row_folder and (strip_user_tags(row_folder) or row_folder) == folder:
                return row

    for row in manifest.rows:
        if not row.local_path:
            continue
        stem_root = (
            _tagless_path(_normalize_path(row.local_path))
            .replace("/tracks/", "/stems/")
            .rsplit(".", 1)[0]
        )
        if tagless.startswith(stem_root + "/"):
            return row

    return None


def _filename_stem_marker(fname: str) -> str | None:
    """Explicit stem qualifier in a master filename, e.g. ``... (Acappella).m4a``
    or ``... (Instrumental Mix).m4a``. A downloaded acappella/instrumental master
    lives in ``tracks/`` too — the qualifier, not the folder, names the stem."""
    if "acappella" in fname or "acapella" in fname:
        return "acappella"
    if "instrumental" in fname:
        return "instrumental"
    return None


def classify_path(path: str) -> tuple[str, str]:
    """Return (claimed_stem, ref_source) from the clip's referenced AUDIO FILE.

    The ``.als`` is the canonical stem oracle: the file the human placed decides
    the stem, in precedence order — Demucs stems and candidate downloads are
    unambiguous; a master is ``regular`` UNLESS its filename carries an explicit
    ``(Acappella)`` / ``(Instrumental)`` qualifier.

    The folder is NOT authoritative: the old code returned ``regular`` for
    everything under ``/tracks/`` *before* reading the filename, silently
    dropping the stem of every ``tracks/... (Acappella).m4a`` master (45 BB12 GT
    rows landed as untagged-regular, incl. the real ``Bad Day (Acappella)``).
    See ``test_classify_path_tracks_master_stem_marker``.
    """
    p = path.replace("\\", "/").lower()
    fname = p.rsplit("/", 1)[-1]

    # 1. Demucs separated stems — unambiguous, regardless of parent folder name.
    if p.endswith("/vocals.flac"):
        return "acappella", "demucs"
    if p.endswith("/instrumental.flac"):
        return "instrumental", "demucs"
    # 2. Downloaded candidate stems.
    if "/candidates/vocals/" in p:
        return "acappella", "online_candidate"
    if "/candidates/instrumental/" in p:
        return "instrumental", "online_candidate"
    if "/candidates/" in p:
        if "instrumental" in fname:
            return "instrumental", "online_candidate"
        return "acappella", "online_candidate"
    # 3. Phase-cancel extractions.
    if "/phase_cancel/" in p or "phase_cancel" in p:
        if "vocals" in p or "acap" in p:
            return "acappella", "phase_cancel"
        return "instrumental", "phase_cancel"
    # 4. Master file (tracks/ or anywhere else): the filename qualifier is the
    #    oracle; default regular. Version tags like (Remix)/(Rework) do NOT flip
    #    the stem.
    marker = _filename_stem_marker(fname)
    if marker:
        return marker, "reference"
    return "regular", "reference"


def display_from_path(path: str) -> str:
    """Human label inferred from an aligning-folder path (filename or parent dir)."""
    p = Path(path.replace("\\", "/"))
    name = p.name
    if name in ("vocals.flac", "instrumental.flac"):
        name = p.parent.name
        name = re.sub(r"^\d+(?:w\d+)?__", "", name)
        if "__" in name:
            name = name.rsplit("__", 1)[0]
        else:
            name = Path(name).stem
    elif name.startswith("cand") and "__" in name:
        name = name.split("__", 1)[1]
        if "__" in name:
            name = name.rsplit("__", 1)[0]
        else:
            name = Path(name).stem
    else:
        name = re.sub(r"^\d+(?:w\d+)?__", "", name)
        if "__" in name:
            name = name.rsplit("__", 1)[0]
        else:
            name = Path(name).stem
    return strip_user_tags(name)


def labels_overlap(left: str, right: str, *, min_tokens: int = 2) -> bool:
    """True when two display labels share enough distinctive tokens."""

    def _tokens(label: str) -> set[str]:
        cleaned = re.sub(r"[^\w\s]", " ", label.lower())
        return {w for w in cleaned.split() if len(w) > 2}

    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return False
    shared = a & b
    if len(shared) >= min_tokens:
        return True
    shorter = min(len(a), len(b))
    return shorter > 0 and len(shared) / shorter >= 0.4


def resolve_identity(
    clip: ParsedClip,
    manifest: ManifestIndex,
) -> tuple[str | None, str | None, str, str]:
    """Return (recording_id, slot_label, display_label, claimed_stem).

    Identity is ALS-canonical: display/stem/slot come from the clip path.
    ``track_id`` is filled only on an exact manifest path match (pull inventory),
    never from scrape slot or title guessing.
    """
    claimed_stem, _ = classify_path(clip.path)
    path_label = display_from_path(clip.path)
    path_slot = slot_from_path(clip.path) or ""

    matched = match_manifest_for_path(clip.path, manifest)
    track_id = matched.track_id if matched is not None else None

    return track_id, path_slot, path_label or clip.track_name, claimed_stem


def normalize_stem_value(raw: str) -> str:
    return normalize_stem(raw.strip() or None)
