"""Load UnmixDB mixes + perfect placement labels for aligner pretraining.

Dataset: https://zenodo.org/records/1422385
Format docs: https://github.com/Ircam-RnD/unmixdb-creation
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..records import SlotCandidate, SpanTarget

_GOOD_MIX_LIST = "unmixdb-v1.1-goodmixes-with-tracks-with-silence-less-than-1s.txt"
_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg")


@dataclass(frozen=True)
class UnmixTrackSpan:
    track_idx: int
    filename: str
    set_start_s: float
    set_end_s: float
    ref_start_s: float
    ref_end_s: float
    tempo_ratio: float
    bpm: float | None


@dataclass(frozen=True)
class UnmixMix:
    mix_id: str
    mix_audio: Path
    labels_path: Path
    track_audio: dict[int, Path]
    spans: tuple[UnmixTrackSpan, ...]


def discover_root(path: Path | str) -> Path:
    """Accept an UnmixDB root or a nested v1.1 directory."""
    root = Path(path).expanduser().resolve()
    if (root / "mixes").is_dir() or list(root.glob("*.labels.txt")):
        return root
    nested = root / "unmixdb-v1.1"
    if nested.is_dir():
        return nested
    # real Zenodo layout: a parent of per-set dirs (mixotic-*/mixes/*.labels.txt)
    if list(root.glob("*/mixes")) or next(root.rglob("*.labels.txt"), None) is not None:
        return root
    raise FileNotFoundError(f"not an UnmixDB root (no mixes/ or *.labels.txt): {root}")


def good_mix_ids(root: Path) -> frozenset[str] | None:
    """Return v1.1 good-mix ids when the filter file is present."""
    for candidate in (
        root / _GOOD_MIX_LIST,
        root / "unmixdb-v1.1" / _GOOD_MIX_LIST,
    ):
        if candidate.is_file():
            ids = {
                line.strip()
                for line in candidate.read_text().splitlines()
                if line.strip() and not line.startswith("#")
            }
            return frozenset(ids)
    return None


def _recording_id(filename: str) -> str:
    name = Path(filename.strip('"')).name
    return Path(name).stem


def parse_labels(labels_path: Path) -> tuple[UnmixTrackSpan, ...]:
    """Parse one `.labels.txt` into per-track active spans in the mix."""
    rows: list[tuple[float, float, int, str, str]] = []
    with labels_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            start = float(parts[0])
            end = float(parts[1])
            track = int(parts[2])
            label = parts[3].strip()
            param = parts[4].strip().strip('"') if len(parts) > 4 else ""
            rows.append((start, end, track, label, param))

    by_track: dict[int, dict[str, list]] = {}
    for start, end, track, label, param in rows:
        by_track.setdefault(track, {}).setdefault(label, []).append((start, end, param))

    spans: list[UnmixTrackSpan] = []
    for track_idx in sorted(by_track):
        ev = by_track[track_idx]
        starts = ev.get("start", [])
        if not starts:
            continue
        filename = starts[0][2]
        speed_rows = ev.get("speed", [])
        tempo_ratio = float(speed_rows[0][2]) if speed_rows else 1.0
        bpm_rows = ev.get("bpm", [])
        bpm = float(bpm_rows[0][2]) if bpm_rows else None

        fadeins = ev.get("fadein", [])
        fadeouts = ev.get("fadeout", [])
        stops = ev.get("stop", [])

        set_start = fadeins[0][0] if fadeins else starts[0][0]
        if fadeouts:
            set_end = fadeouts[0][1]
        elif stops:
            set_end = stops[0][0]
        else:
            set_end = starts[0][0] + 20.0

        if set_end <= set_start:
            continue

        dur = set_end - set_start
        spans.append(
            UnmixTrackSpan(
                track_idx=track_idx,
                filename=filename,
                set_start_s=set_start,
                set_end_s=set_end,
                ref_start_s=0.0,
                ref_end_s=dur,
                tempo_ratio=tempo_ratio,
                bpm=bpm,
            )
        )
    return tuple(spans)


def _find_audio(stem: str, *dirs: Path) -> Path | None:
    for d in dirs:
        if not d.is_dir():
            continue
        for ext in _AUDIO_EXTS:
            hit = d / f"{stem}{ext}"
            if hit.is_file():
                return hit
        for hit in d.glob(f"{stem}.*"):
            if hit.suffix.lower() in _AUDIO_EXTS:
                return hit
    return None


def _resolve_track_paths(
    spans: tuple[UnmixTrackSpan, ...],
    *,
    mixes_dir: Path,
    tracks_dir: Path,
    mix_dir: Path,
) -> dict[int, Path]:
    out: dict[int, Path] = {}
    search_dirs = (tracks_dir, mix_dir, mixes_dir)
    for sp in spans:
        rid = _recording_id(sp.filename)
        path = _find_audio(rid, *search_dirs)
        if path is None:
            path = _find_audio(Path(sp.filename).stem, *search_dirs)
        if path is None:
            raise FileNotFoundError(
                f"track audio not found for {sp.filename!r} under {search_dirs}"
            )
        out[sp.track_idx] = path
    return out


def load_mix(labels_path: Path, *, root: Path) -> UnmixMix:
    mix_id = labels_path.name.replace(".labels.txt", "")
    mixes_dir = root / "mixes"
    # real Zenodo layout: tracks live in the set dir's refsongs/ (labels are in
    # <set>/mixes/, so the set dir is labels_path.parent.parent).
    tracks_dir = labels_path.parent.parent / "refsongs"
    if not tracks_dir.is_dir():
        tracks_dir = root / "tracks"
    mix_dir = labels_path.parent

    mix_audio = _find_audio(mix_id, mixes_dir, mix_dir)
    if mix_audio is None:
        raise FileNotFoundError(f"mix audio not found for {mix_id}")

    spans = parse_labels(labels_path)
    if not spans:
        raise ValueError(f"no spans parsed from {labels_path}")

    track_audio = _resolve_track_paths(
        spans,
        mixes_dir=mixes_dir,
        tracks_dir=tracks_dir,
        mix_dir=mix_dir,
    )
    return UnmixMix(
        mix_id=mix_id,
        mix_audio=mix_audio,
        labels_path=labels_path,
        track_audio=track_audio,
        spans=spans,
    )


def iter_mixes(
    root: Path | str,
    *,
    good_only: bool = True,
    max_mixes: int | None = None,
) -> tuple[UnmixMix, ...]:
    """Load mixes from an on-disk UnmixDB tree."""
    base = discover_root(root)
    good = good_mix_ids(base) if good_only else None

    label_files: list[Path] = []
    mixes_dir = base / "mixes"
    if mixes_dir.is_dir():
        label_files = sorted(mixes_dir.glob("*.labels.txt"))
    if not label_files:
        label_files = sorted(base.rglob("*.labels.txt"))

    out: list[UnmixMix] = []
    for labels_path in label_files:
        mix_id = labels_path.name.replace(".labels.txt", "")
        if good is not None and mix_id not in good:
            continue
        try:
            out.append(load_mix(labels_path, root=base))
        except (FileNotFoundError, ValueError):
            continue
        if max_mixes is not None and len(out) >= max_mixes:
            break
    return tuple(out)


def labels_to_targets(mix: UnmixMix) -> tuple[SpanTarget, ...]:
    """Map one UnmixDB mix to aligner SpanTarget rows."""
    rows: list[SpanTarget] = []
    for sp in mix.spans:
        rid = _recording_id(sp.filename)
        rows.append(
            SpanTarget(
                slot_label=f"{mix.mix_id}t{sp.track_idx}",
                recording_id=rid,
                claimed_stem="regular",
                set_start_s=sp.set_start_s,
                set_end_s=sp.set_end_s,
                ref_start_s=sp.ref_start_s,
                ref_end_s=sp.ref_end_s,
                tempo_ratio=sp.tempo_ratio,
                pitch_shift_semi=0,
                label=sp.filename,
            )
        )
    return tuple(rows)


def slot_pools_for_mix(mix: UnmixMix) -> dict[str, tuple[SlotCandidate, ...]]:
    """Three-track mixes: each slot's pool is all tracks in the mix."""
    pool = tuple(
        SlotCandidate(recording_id=_recording_id(sp.filename), claimed_stem="regular")
        for sp in mix.spans
    )
    return {f"{mix.mix_id}t{sp.track_idx}": pool for sp in mix.spans}


def summarize_mixes(mixes: tuple[UnmixMix, ...]) -> str:
    n_spans = sum(len(m.spans) for m in mixes)
    return f"mixes={len(mixes)} spans={n_spans}"
