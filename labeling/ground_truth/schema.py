"""Ground-truth yaml schema — dataclasses, parser, serializer, validator.

This is the one place that knows the shape of
`tests/fixtures/*_ground_truth.yaml`. Everything else (the eval harness,
the Streamlit editor) goes through `load` / `dump` / `save` here.

Style notes (per CLAUDE.md):
  - Frozen dataclasses, no mutation.
  - Domain errors returned as `Result[T, GroundTruthError]`, not raised.
  - PyYAML is the one library boundary; its exceptions are caught
    exactly where yaml.safe_load is invoked and mapped to typed errors.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from core.identity import normalize_stem
from core.result import Err, Ok, Result


@dataclass(frozen=True)
class GroundTruthError:
    kind: str  # 'yaml_parse' | 'io' | 'schema'
    detail: str
    path: Path | None = None


@dataclass(frozen=True)
class RefSegment:
    """One iteration of a loop or one slice of a cut-up."""
    ref_start_s: float
    ref_end_s: float
    mix_start_s: float


@dataclass(frozen=True)
class MediaLinks:
    youtube: str = ""
    spotify: str = ""
    soundcloud: str = ""
    other: str = ""

    def any(self) -> bool:
        return bool(self.youtube or self.spotify or self.soundcloud or self.other)

    def as_dict(self) -> dict[str, str]:
        return {k: v for k, v in (
            ("youtube", self.youtube),
            ("spotify", self.spotify),
            ("soundcloud", self.soundcloud),
            ("other", self.other),
        ) if v}


@dataclass(frozen=True)
class GroundTruthTrack:
    """One annotated play-span from a DJ set."""
    label: str                              # human-readable track name
    track_id: str | None                    # 1001tracklists data-trackid
    claimed_stem: str                       # regular | acappella | instrumental
    set_start_s: float
    set_end_s: float
    ref_start_s: float                      # MANDATORY — see schema doc
    ref_end_s: float | None = None
    is_loop: bool = False
    ref_segments: tuple[RefSegment, ...] = ()
    media_links: MediaLinks = field(default_factory=MediaLinks)


@dataclass(frozen=True)
class GroundTruthSet:
    set_id: str
    tracks: tuple[GroundTruthTrack, ...]
    source: str = "ableton_session"
    annotated_by: str = "user"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_track(idx: int, t: dict[str, Any], path: Path) -> Result[GroundTruthTrack, GroundTruthError]:
    label = str(t.get("track", "")).strip()
    if "ref_start_s" not in t or t["ref_start_s"] is None:
        return Err(GroundTruthError(
            kind="schema",
            detail=(
                f"track #{idx} ({label!r}) is missing mandatory `ref_start_s`. "
                "Add the seconds offset into the reference where the DJ first "
                "dropped in (top-level, not inside ref_segments)."
            ),
            path=path,
        ))
    is_loop = bool(t.get("is_loop", False))
    raw_segs = t.get("ref_segments") or ()
    segments: list[RefSegment] = []
    for s in raw_segs:
        if not isinstance(s, dict):
            continue
        if not all(k in s for k in ("ref_start_s", "ref_end_s", "mix_start_s")):
            continue
        segments.append(RefSegment(
            ref_start_s=float(s["ref_start_s"]),
            ref_end_s=float(s["ref_end_s"]),
            mix_start_s=float(s["mix_start_s"]),
        ))
    if is_loop and not segments:
        return Err(GroundTruthError(
            kind="schema",
            detail=(
                f"track #{idx} ({label!r}) has `is_loop: true` but no "
                "`ref_segments`. Add one segment per loop iteration."
            ),
            path=path,
        ))
    tid_raw = t.get("track_id")
    track_id = str(tid_raw).strip() if tid_raw not in (None, "") else None
    ml = t.get("media_links") or {}
    if not isinstance(ml, dict):
        ml = {}
    ref_end_raw = t.get("ref_end_s")
    ref_end = float(ref_end_raw) if isinstance(ref_end_raw, (int, float)) else None
    return Ok(GroundTruthTrack(
        label=label,
        track_id=track_id,
        claimed_stem=normalize_stem(
            str(t.get("claimed_stem") or t.get("version_tag") or "").strip() or None
        ),
        set_start_s=float(t["set_start_s"]),
        set_end_s=float(t["set_end_s"]),
        ref_start_s=float(t["ref_start_s"]),
        ref_end_s=ref_end,
        is_loop=is_loop,
        ref_segments=tuple(segments),
        media_links=MediaLinks(
            youtube=str(ml.get("youtube") or "").strip(),
            spotify=str(ml.get("spotify") or "").strip(),
            soundcloud=str(ml.get("soundcloud") or "").strip(),
            other=str(ml.get("other") or "").strip(),
        ),
    ))


def load(yaml_path: Path | str) -> Result[GroundTruthSet, GroundTruthError]:
    """Parse one ground-truth yaml into a typed `GroundTruthSet`."""
    path = Path(yaml_path)
    try:
        raw = path.read_text()
    except OSError as e:
        return Err(GroundTruthError(kind="io", detail=str(e), path=path))
    try:
        payload = yaml.safe_load(raw) or {}
    except yaml.YAMLError as e:
        return Err(GroundTruthError(kind="yaml_parse", detail=str(e), path=path))
    if not isinstance(payload, dict):
        return Err(GroundTruthError(
            kind="schema",
            detail="top-level yaml must be a mapping",
            path=path,
        ))
    set_id = str(payload.get("set_id") or "").strip()
    if not set_id:
        return Err(GroundTruthError(
            kind="schema",
            detail="missing required `set_id` field",
            path=path,
        ))

    tracks: list[GroundTruthTrack] = []
    raw_tracks = payload.get("tracks") or []
    if not isinstance(raw_tracks, list):
        return Err(GroundTruthError(
            kind="schema",
            detail="`tracks:` must be a list",
            path=path,
        ))
    for idx, t in enumerate(raw_tracks):
        if not isinstance(t, dict):
            continue
        r = _parse_track(idx, t, path)
        if not r.is_ok():
            return Err(r.error)
        tracks.append(r.value)

    return Ok(GroundTruthSet(
        set_id=set_id,
        tracks=tuple(tracks),
        source=str(payload.get("source") or "ableton_session"),
        annotated_by=str(payload.get("annotated_by") or "user"),
    ))


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _fmt_num(v: float) -> str:
    """Match the existing fixture style: `%g`, no trailing zeros."""
    return f"{float(v):g}"


def dump(gt: GroundTruthSet, *, title: str | None = None) -> str:
    """Serialize to yaml text with stable ordering and inline comments.

    We hand-render rather than `yaml.safe_dump` so the file stays
    diff-friendly and preserves the human-readable field order seen
    in the existing fixtures (track → track_id → claimed_stem → ...).
    """
    out: list[str] = []
    if title:
        out.append(f"# Hand-annotated ground-truth for {title}")
        out.append("#")
    out.append("# Schema: see labeling/ground_truth/ (DSL).")
    out.append(f"set_id: {gt.set_id}")
    out.append(f"source: {gt.source}")
    out.append(f"annotated_by: {gt.annotated_by}")
    out.append("tracks:")
    for t in gt.tracks:
        label = t.label.replace('"', r'\"')
        out.append(f'  - track:       "{label}"')
        if t.track_id:
            out.append(f"    track_id:    {t.track_id}")
        if t.claimed_stem and t.claimed_stem != "regular":
            out.append(f"    claimed_stem: {t.claimed_stem}")
        out.append(f"    set_start_s: {_fmt_num(t.set_start_s)}")
        out.append(f"    set_end_s:   {_fmt_num(t.set_end_s)}")
        out.append(f"    ref_start_s: {_fmt_num(t.ref_start_s)}")
        if t.ref_end_s is not None:
            out.append(f"    ref_end_s:   {_fmt_num(t.ref_end_s)}")
        if t.is_loop:
            out.append(f"    is_loop:     true")
        if t.ref_segments:
            out.append("    ref_segments:")
            for s in t.ref_segments:
                out.append(f"      - mix_start_s: {_fmt_num(s.mix_start_s)}")
                out.append(f"        ref_start_s: {_fmt_num(s.ref_start_s)}")
                out.append(f"        ref_end_s:   {_fmt_num(s.ref_end_s)}")
        if t.media_links.any():
            out.append("    media_links:")
            for k, v in t.media_links.as_dict().items():
                out.append(f"      {k}: {v}")
    return "\n".join(out) + "\n"


def save(
    gt: GroundTruthSet,
    path: Path | str,
    *,
    title: str | None = None,
    archive_dir: Path | str | None = None,
) -> Result[Path, GroundTruthError]:
    """Write yaml to disk, archiving any prior version first.

    If `archive_dir` is provided and the target already exists, the prior
    file is copied under `<archive_dir>/<stem>_<timestamp>.yaml`. Returns
    the final output path on success.
    """
    out_path = Path(path)
    try:
        if archive_dir and out_path.exists():
            archive_root = Path(archive_dir)
            archive_root.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            shutil.copy2(out_path, archive_root / f"{out_path.stem}_{stamp}.yaml")
        out_path.write_text(dump(gt, title=title))
    except OSError as e:
        return Err(GroundTruthError(kind="io", detail=str(e), path=out_path))
    return Ok(out_path)


# ---------------------------------------------------------------------------
# Small conveniences used by the editor UI
# ---------------------------------------------------------------------------

def upsert_track(
    gt: GroundTruthSet, new_track: GroundTruthTrack,
) -> GroundTruthSet:
    """Replace the track with the same track_id (if present) else append.

    Useful when the UI round-trips a single row without rebuilding the
    whole set from scratch.
    """
    if not new_track.track_id:
        return replace(gt, tracks=gt.tracks + (new_track,))
    updated: list[GroundTruthTrack] = []
    found = False
    for t in gt.tracks:
        if t.track_id and t.track_id == new_track.track_id:
            updated.append(new_track)
            found = True
        else:
            updated.append(t)
    if not found:
        updated.append(new_track)
    return replace(gt, tracks=tuple(updated))
