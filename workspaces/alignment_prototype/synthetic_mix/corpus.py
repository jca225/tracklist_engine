"""Load synthetic mashup corpus for aligner pretrain (UnmixDB-compatible interface)."""

from __future__ import annotations

import json
from pathlib import Path

from core.result import Err, Ok
from labeling.ground_truth.schema import GroundTruthTrack, load
from workspaces.alignment_prototype.dataset import track_to_target
from workspaces.alignment_prototype.external.unmixdb import UnmixMix, UnmixTrackSpan
from workspaces.alignment_prototype.records import SlotCandidate, SpanTarget

_MANIFEST = "corpus_manifest.json"


def track_to_targets(t: GroundTruthTrack) -> tuple[SpanTarget, ...]:
    """One SpanTarget per ref_segment row (loops / jump-cut instrumentals)."""
    if not t.track_id:
        return ()
    if t.ref_segments:
        out: list[SpanTarget] = []
        segs = t.ref_segments
        for i, seg in enumerate(segs):
            if i + 1 < len(segs):
                mix_end = segs[i + 1].mix_start_s
            elif t.is_loop:
                tr = t.tempo_ratio if t.tempo_ratio else 1.0
                phrase_s = (seg.ref_end_s - seg.ref_start_s) / max(tr, 1e-6)
                mix_end = min(t.set_end_s, seg.mix_start_s + phrase_s)
            else:
                mix_end = t.set_end_s
            out.append(
                SpanTarget(
                    slot_label=t.slot_label or t.label,
                    recording_id=t.track_id,
                    claimed_stem=t.claimed_stem or "regular",
                    set_start_s=seg.mix_start_s,
                    set_end_s=mix_end,
                    ref_start_s=seg.ref_start_s,
                    ref_end_s=seg.ref_end_s,
                    tempo_ratio=t.tempo_ratio,
                    pitch_shift_semi=t.pitch_shift_semi,
                    label=t.label,
                )
            )
        return tuple(out)
    return (track_to_target(t),)


def _ref_path(refs_dir: Path, recording_id: str, claimed_stem: str) -> Path | None:
    if claimed_stem == "acappella":
        stem_file = "vocals"
    elif claimed_stem == "regular":
        stem_file = "regular"
    else:
        stem_file = "instrumental"
    p = refs_dir / f"{recording_id}_{stem_file}.flac"
    return p if p.is_file() else None


def _track_filename(t: GroundTruthTrack) -> str:
    return f"{t.track_id}.flac"


def load_mix(mix_dir: Path) -> UnmixMix | None:
    """Load one synthetic mix directory into UnmixMix."""
    gt_path = mix_dir / "ground_truth.yaml"
    mix_audio = mix_dir / "mix.flac"
    if not gt_path.is_file() or not mix_audio.is_file():
        return None

    match load(gt_path):
        case Err():
            return None
        case Ok(gt):
            pass

    refs_dir = mix_dir / "refs"
    track_audio: dict[int, Path] = {}
    spans: list[UnmixTrackSpan] = []

    for idx, t in enumerate(gt.tracks):
        if not t.track_id:
            continue
        ref = _ref_path(refs_dir, t.track_id, t.claimed_stem)
        if ref is None:
            return None
        track_audio[idx] = ref
        ref_end = t.ref_end_s or (t.ref_start_s + (t.set_end_s - t.set_start_s))
        tr = t.tempo_ratio if t.tempo_ratio is not None else 1.0
        spans.append(
            UnmixTrackSpan(
                track_idx=idx,
                filename=_track_filename(t),
                set_start_s=t.set_start_s,
                set_end_s=t.set_end_s,
                ref_start_s=t.ref_start_s,
                ref_end_s=ref_end,
                tempo_ratio=tr,
                bpm=None,
            )
        )

    if not spans:
        return None

    return UnmixMix(
        mix_id=gt.set_id,
        mix_audio=mix_audio,
        labels_path=gt_path,
        track_audio=track_audio,
        spans=tuple(spans),
    )


def _mix_dirs(root: Path) -> list[Path]:
    """Discover mix directories (manifest or synth_*/synthv2_* + ground_truth)."""
    manifest = root / _MANIFEST
    if manifest.is_file():
        dirs: list[Path] = []
        for row in json.loads(manifest.read_text()):
            d = root / row["dir"]
            if d.is_dir():
                dirs.append(d)
        return dirs
    return sorted(
        p
        for p in root.iterdir()
        if p.is_dir()
        and (p.name.startswith("synth_") or p.name.startswith("synthv2_"))
        and (p / "ground_truth.yaml").is_file()
    )


def iter_mixes(
    root: Path | str,
    *,
    max_mixes: int | None = None,
) -> tuple[UnmixMix, ...]:
    """Iterate synthetic mixes under root (reads corpus_manifest.json if present)."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        return ()

    dirs = _mix_dirs(root)
    if max_mixes is not None:
        dirs = dirs[:max_mixes]

    mixes: list[UnmixMix] = []
    for d in dirs:
        m = load_mix(d)
        if m is not None:
            mixes.append(m)
    return tuple(mixes)


def targets_for_mix(mix: UnmixMix) -> tuple[SpanTarget, ...]:
    """Span targets from ground_truth.yaml (expands ref_segments)."""
    match load(mix.labels_path):
        case Err():
            return ()
        case Ok(gt):
            rows: list[SpanTarget] = []
            for t in gt.tracks:
                rows.extend(track_to_targets(t))
            return tuple(rows)


def slot_pools_for_mix(mix: UnmixMix) -> dict[str, tuple[SlotCandidate, ...]]:
    """Candidate pool per GT slot_label."""
    targets = targets_for_mix(mix)
    by_slot: dict[str, list[SlotCandidate]] = {}
    for t in targets:
        if not t.recording_id:
            continue
        by_slot.setdefault(t.slot_label, [])
        cand = SlotCandidate(recording_id=t.recording_id, claimed_stem=t.claimed_stem)
        if cand not in by_slot[t.slot_label]:
            by_slot[t.slot_label].append(cand)
    return {k: tuple(v) for k, v in by_slot.items()}


def summarize_mixes(mixes: tuple[UnmixMix, ...]) -> str:
    if not mixes:
        return "synthetic corpus: 0 mixes"
    span_rows = sum(len(m.spans) for m in mixes)
    train_rows = 0
    for m in mixes:
        train_rows += len(targets_for_mix(m))
    extra = f", {train_rows} train rows" if train_rows != span_rows else ""
    return f"synthetic corpus: {len(mixes)} mixes, {span_rows} labeled spans{extra}"
