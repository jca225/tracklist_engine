"""Export BB12 (or any pulled set) Ableton `.als` → ground-truth YAML.

Reads the live `.als`, maps clips through 1-mix warp markers into mix seconds,
reads identity from each clip's file path, optionally attaches ``track_id`` when
the path exactly matches a row in `~/aligning/<set>/manifest.json` (pull
inventory only — not used for labels or timing), applies clip hygiene, and
writes a `*_ground_truth.yaml` consumable by `labeling.write_back_ground_truth`.

Usage (Mac, from repo root):

    venvs/audio/bin/python -m labeling.export_als_to_gt \\
        --als "$HOME/Desktop/big bootie 12 labeling Project/big bootie 12 labeling_fast.als" \\
        --set-dir "$HOME/aligning/1fsnxchk__Two Friends - Big Bootie Mix Volume 12" \\
        --out labeling/fixtures/bb12_ground_truth.yaml

Review table only:

    venvs/audio/bin/python -m labeling.export_als_to_gt ... --review
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from labeling.als_io import (
    ArrangementMapper,
    MUTE_THR,
    ParsedClip,
    audible_span,
    build_manifest_index,
    classify_path,
    load_als_xml,
    parse_layer_clips,
    resolve_identity,
    split_clip_at_mix_span_edges,
    tempo_ratio,
    track_display_name,
)
from labeling.ground_truth.schema import GroundTruthSet, GroundTruthTrack, RefSegment, save
from core.result import Err, Ok

DEFAULT_ALS = Path.home() / "Desktop/big bootie 12 labeling Project/big bootie 12 labeling_fast.als"
DEFAULT_SET_DIR = Path.home() / "aligning/1fsnxchk__Two Friends - Big Bootie Mix Volume 12"

PARKING_TOL_S = 5.0
SLIVER_MAX_S = 3.0
MICRO_SLIVER_DROP_S = 0.01   # clip-boundary specks (e.g. slot 107)
LOOP_OVERLAP_MIN = 0.35
BLINK_182_SLOTS = frozenset({"024", "024w1", "029"})
AUDIBLE_EPS_S = 0.05


def _min_frac(*values: float | None) -> float | None:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    m = min(nums)
    return None if m >= 1.0 else m


def _min_optional(*values: float | None, default: float) -> float | None:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    m = min(nums)
    return None if abs(m - default) < AUDIBLE_EPS_S else m


def _max_optional(*values: float | None, default: float) -> float | None:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    m = max(nums)
    return None if abs(m - default) < AUDIBLE_EPS_S else m


@dataclass(frozen=True)
class ReviewRow:
    action: str  # kept | dropped | merged
    reason: str
    group: str
    slot: str
    track: str
    set_start_s: float | None = None
    set_end_s: float | None = None
    recording_id: str | None = None


@dataclass(frozen=True)
class ClipRow:
    clip: ParsedClip
    set_start_s: float
    set_end_s: float
    ref_start_s: float
    ref_end_s: float
    recording_id: str | None
    slot_label: str
    display: str
    claimed_stem: str
    ref_source: str
    tempo_ratio: float | None
    pitch_shift_semi: int
    is_loop: bool = False
    ref_segments: tuple[RefSegment, ...] = ()
    audible_frac: float | None = None
    audible_start_s: float | None = None
    audible_end_s: float | None = None
    skip_training: bool = False


def _mix_track(root) -> object | None:
    for track_el in root.xpath(".//LiveSet/Tracks/*"):
        if track_el.tag != "AudioTrack":
            continue
        if track_display_name(track_el) == "1-mix":
            return track_el
    return None


def _clip_row(clip: ParsedClip, mapper: ArrangementMapper, manifest) -> ClipRow | None:
    set_start = mapper.arr_to_set_sec(clip.arr_start)
    set_end = mapper.arr_to_set_sec(clip.arr_end)
    if set_start is None or set_end is None:
        return None
    recording_id, slot_label, display, claimed_stem = resolve_identity(clip, manifest)
    _, ref_source = classify_path(clip.path)
    ref_start = clip.ref_start_s()
    ref_end = clip.ref_end_s()
    set_span = set_end - set_start
    ref_span = ref_end - ref_start
    aud = audible_span(clip.vol_points, clip.arr_start, clip.arr_end)
    audible_start = mapper.arr_to_set_sec(aud.arr_start) if aud.fraction > 0 else None
    audible_end = mapper.arr_to_set_sec(aud.arr_end) if aud.fraction > 0 else None
    skip = aud.fraction < MUTE_THR
    aud_start_out = audible_start
    aud_end_out = audible_end
    if aud_start_out is not None and abs(aud_start_out - set_start) < AUDIBLE_EPS_S:
        aud_start_out = None
    if aud_end_out is not None and abs(aud_end_out - set_end) < AUDIBLE_EPS_S:
        aud_end_out = None
    aud_frac_out = None if aud.fraction >= 1.0 else round(aud.fraction, 3)
    return ClipRow(
        clip=clip,
        set_start_s=set_start,
        set_end_s=set_end,
        ref_start_s=ref_start,
        ref_end_s=ref_end,
        recording_id=recording_id,
        slot_label=slot_label,
        display=display,
        claimed_stem=claimed_stem,
        ref_source=ref_source,
        tempo_ratio=tempo_ratio(set_span, ref_span),
        pitch_shift_semi=clip.pitch_coarse,
        audible_frac=aud_frac_out,
        audible_start_s=aud_start_out,
        audible_end_s=aud_end_out,
        skip_training=skip,
    )


def _drop_parking(rows: list[ClipRow], mix_end_s: float) -> tuple[list[ClipRow], list[ReviewRow]]:
    kept: list[ClipRow] = []
    review: list[ReviewRow] = []
    cutoff = mix_end_s + PARKING_TOL_S
    for row in rows:
        if row.set_start_s > cutoff:
            review.append(ReviewRow(
                action="dropped",
                reason=f"parking-lot (set_start_s>{cutoff:.1f})",
                group=row.clip.group_name,
                slot=row.slot_label,
                track=row.display,
                set_start_s=row.set_start_s,
                set_end_s=row.set_end_s,
                recording_id=row.recording_id,
            ))
        else:
            kept.append(row)
    return kept, review


def _merge_slivers(rows: list[ClipRow]) -> tuple[list[ClipRow], list[ReviewRow]]:
    if not rows:
        return rows, []
    rows = sorted(rows, key=lambda r: (r.clip.path, r.set_start_s))
    kept: list[ClipRow] = []
    review: list[ReviewRow] = []
    i = 0
    while i < len(rows):
        row = rows[i]
        span = row.set_end_s - row.set_start_s
        if 0 < span <= SLIVER_MAX_S and kept and kept[-1].clip.path == row.clip.path:
            prev = kept[-1]
            merged = replace(
                prev,
                set_end_s=max(prev.set_end_s, row.set_end_s),
                ref_end_s=max(prev.ref_end_s, row.ref_end_s),
                tempo_ratio=tempo_ratio(
                    max(prev.set_end_s, row.set_end_s) - prev.set_start_s,
                    max(prev.ref_end_s, row.ref_end_s) - prev.ref_start_s,
                ),
                audible_frac=_min_frac(prev.audible_frac, row.audible_frac),
                audible_start_s=_min_optional(
                    prev.audible_start_s, row.audible_start_s, default=prev.set_start_s,
                ),
                audible_end_s=_max_optional(
                    prev.audible_end_s, row.audible_end_s, default=row.set_end_s,
                ),
                skip_training=prev.skip_training or row.skip_training,
            )
            kept[-1] = merged
            review.append(ReviewRow(
                action="merged",
                reason=f"sliver ({span:.2f}s) into neighbor",
                group=row.clip.group_name,
                slot=row.slot_label,
                track=row.display,
                set_start_s=row.set_start_s,
                set_end_s=row.set_end_s,
                recording_id=row.recording_id,
            ))
            i += 1
            continue
        kept.append(row)
        i += 1
    return kept, review


def _drop_micro_slivers(rows: list[ClipRow]) -> tuple[list[ClipRow], list[ReviewRow]]:
    kept: list[ClipRow] = []
    review: list[ReviewRow] = []
    for row in rows:
        span = row.set_end_s - row.set_start_s
        if span < MICRO_SLIVER_DROP_S:
            review.append(ReviewRow(
                action="dropped",
                reason=f"micro-sliver ({span:.4f}s < {MICRO_SLIVER_DROP_S}s)",
                group=row.clip.group_name,
                slot=row.slot_label,
                track=row.display,
                set_start_s=row.set_start_s,
                set_end_s=row.set_end_s,
                recording_id=row.recording_id,
            ))
        else:
            kept.append(row)
    return kept, review


def _detect_loops(rows: list[ClipRow]) -> list[ClipRow]:
    """Group same-path clips into loop rows when they overlap in the mix."""
    if not rows:
        return rows
    by_key: dict[tuple[str, str, str], list[ClipRow]] = {}
    for row in rows:
        key = (row.clip.path, row.slot_label, row.claimed_stem)
        by_key.setdefault(key, []).append(row)
    out: list[ClipRow] = []
    for key_rows in by_key.values():
        key_rows.sort(key=lambda r: (r.clip.arr_start, r.set_start_s))
        if len(key_rows) == 1:
            out.append(key_rows[0])
            continue
        slot_base = key_rows[0].slot_label.split("w", 1)[0]
        if slot_base in BLINK_182_SLOTS:
            out.extend(key_rows)
            continue
        segments: list[RefSegment] = []
        for row in key_rows:
            segments.append(RefSegment(
                ref_start_s=row.ref_start_s,
                ref_end_s=row.ref_end_s,
                mix_start_s=row.set_start_s,
            ))
        first = key_rows[0]
        last = key_rows[-1]
        set_span = sum(
            max(0.0, row.set_end_s - row.set_start_s) for row in key_rows
        )
        # tempo_ratio is the PLAYBACK SPEED: sum of the segment ref-durations
        # actually played, NOT the outer ref envelope. The envelope counts the
        # ref region the DJ JUMPED OVER between non-contiguous segments — that
        # inflated Emily's instrumental (slot 003) to 2.69x when its 3 segments
        # each played at 1.0x (65.3s song played over 65.3s of mix).
        ref_span = sum(
            max(0.0, row.ref_end_s - row.ref_start_s) for row in key_rows
        )
        out.append(replace(
            first,
            set_start_s=min(row.set_start_s for row in key_rows),
            set_end_s=max(row.set_end_s for row in key_rows),
            ref_end_s=last.ref_end_s,
            tempo_ratio=tempo_ratio(set_span, ref_span),
            is_loop=len(segments) > 1,
            ref_segments=tuple(segments),
            audible_frac=_min_frac(*(row.audible_frac for row in key_rows)),
            audible_start_s=_min_optional(
                *(row.audible_start_s for row in key_rows),
                default=min(row.set_start_s for row in key_rows),
            ),
            audible_end_s=_max_optional(
                *(row.audible_end_s for row in key_rows),
                default=max(row.set_end_s for row in key_rows),
            ),
            skip_training=any(row.skip_training for row in key_rows),
        ))
    return sorted(out, key=lambda r: (r.set_start_s, r.slot_label, r.claimed_stem))


import re as _re

# A clip whose audio IS the set's own mix / mix-instrumental (or an imported
# placeholder) is the human's UNALIGNABLE marker: too hard to align OR the
# source doesn't exist anywhere (e.g. Lux Omega). Not a song placement — a
# positive abstain LABEL for the aligner to learn to predict.
_PLACEHOLDER_RE = _re.compile(r"(^mix\.(m4a|flac|wav)$|^mix_instrumental\.|^instrumental-\d+\.)")


def _placeholder_note(path: str, group: str) -> str | None:
    fname = Path(path).name.lower()
    if not _PLACEHOLDER_RE.match(fname):
        return None
    base = Path(path).name
    if fname.startswith("mix_instrumental"):
        return f"mix_instrumental substituted as host — original unavailable ({group})"
    if fname.startswith("mix."):
        return f"mix self-reference — too difficult to align ({group})"
    return f"imported placeholder {base} ({group})"


def _to_gt_track(row: ClipRow) -> GroundTruthTrack:
    note = _placeholder_note(row.clip.path, row.clip.group_name)
    return GroundTruthTrack(
        label=row.display,
        track_id=row.recording_id,
        claimed_stem=row.claimed_stem,
        set_start_s=row.set_start_s,
        set_end_s=row.set_end_s,
        ref_start_s=row.ref_start_s,
        ref_end_s=row.ref_end_s,
        slot_label=row.slot_label,
        ref_source=row.ref_source,
        tempo_ratio=row.tempo_ratio,
        pitch_shift_semi=row.pitch_shift_semi,
        is_loop=row.is_loop,
        ref_segments=row.ref_segments,
        audible_frac=row.audible_frac,
        audible_start_s=row.audible_start_s,
        audible_end_s=row.audible_end_s,
        skip_training=row.skip_training,
        unalignable=note is not None,
        source_note=note,
    )


def collect_kept_clip_rows(
    als_path: Path,
    set_dir: Path,
    *,
    include_all: bool = False,
) -> tuple[str, list[ClipRow], list[ReviewRow]]:
    """Run the ALS export pipeline and return hygiene-passed clip rows."""
    manifest_path = set_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = build_manifest_index(manifest_path)
    manifest_json = __import__("json").loads(manifest_path.read_text())
    set_id = str(manifest_json.get("set_id") or "").strip()
    if not set_id:
        raise ValueError("manifest.json missing set_id")
    mix_duration_s = float(manifest_json.get("mix_duration_s") or 0.0)

    root = load_als_xml(als_path)
    mix_track = _mix_track(root)
    if mix_track is None:
        raise ValueError("no 1-mix track in .als")
    mapper = ArrangementMapper.from_mix_track(mix_track, mix_duration_s=mix_duration_s)

    raw_rows: list[ClipRow] = []
    review: list[ReviewRow] = []
    for clip in parse_layer_clips(root):
        for part in split_clip_at_mix_span_edges(clip, mapper):
            row = _clip_row(part, mapper, manifest)
            if row is None:
                review.append(ReviewRow(
                    action="dropped",
                    reason="outside mix warp span",
                    group=part.group_name,
                    slot=slot_from_path(part.path) or "",
                    track=part.track_name,
                ))
                continue
            if row.set_end_s <= row.set_start_s:
                review.append(ReviewRow(
                    action="dropped",
                    reason="non-positive set span after mix-span split",
                    group=part.group_name,
                    slot=row.slot_label or "",
                    track=row.display,
                    set_start_s=row.set_start_s,
                    set_end_s=row.set_end_s,
                    recording_id=row.recording_id,
                ))
                continue
            raw_rows.append(row)

    if include_all:
        for row in raw_rows:
            review.append(ReviewRow(
                action="kept",
                reason="include-all",
                group=row.clip.group_name,
                slot=row.slot_label,
                track=row.display,
                set_start_s=row.set_start_s,
                set_end_s=row.set_end_s,
                recording_id=row.recording_id,
            ))
        return set_id, sorted(raw_rows, key=lambda r: r.set_start_s), review

    rows, rev = _drop_parking(raw_rows, mix_duration_s)
    review.extend(rev)
    rows, rev = _merge_slivers(rows)
    review.extend(rev)
    rows, rev = _drop_micro_slivers(rows)
    review.extend(rev)
    rows = _detect_loops(rows)

    for row in rows:
        review.append(ReviewRow(
            action="kept",
            reason="hygiene pass",
            group=row.clip.group_name,
            slot=row.slot_label,
            track=row.display,
            set_start_s=row.set_start_s,
            set_end_s=row.set_end_s,
            recording_id=row.recording_id,
        ))

    return set_id, rows, review


def export_gt(
    als_path: Path,
    set_dir: Path,
    *,
    include_all: bool = False,
) -> tuple[GroundTruthSet, list[ReviewRow]]:
    set_id, rows, review = collect_kept_clip_rows(
        als_path, set_dir, include_all=include_all,
    )
    return GroundTruthSet(
        set_id=set_id,
        tracks=tuple(_to_gt_track(r) for r in rows),
    ), review


def slot_from_path(path: str) -> str | None:
    from labeling.als_io import slot_from_path as _slot
    return _slot(path)


def print_review(review: list[ReviewRow]) -> None:
    kept = sum(1 for r in review if r.action == "kept")
    dropped = sum(1 for r in review if r.action == "dropped")
    merged = sum(1 for r in review if r.action == "merged")
    unresolved = sorted({r.slot for r in review if r.action == "kept" and not r.recording_id and r.slot})
    print(f"review: kept={kept} dropped={dropped} merged={merged} unresolved_slots={len(unresolved)}")
    if unresolved:
        print("  unresolved:", ", ".join(unresolved[:20]), ("..." if len(unresolved) > 20 else ""))
    print(f"{'action':8} {'slot':8} {'set_span':22} {'recording':12} reason")
    for row in review[:40]:
        span = ""
        if row.set_start_s is not None and row.set_end_s is not None:
            span = f"{row.set_start_s:.1f}-{row.set_end_s:.1f}s"
        print(
            f"{row.action:8} {row.slot or '-':8} {span:22} "
            f"{row.recording_id or 'NULL':12} {row.reason}"
        )
    if len(review) > 40:
        print(f"  ... +{len(review) - 40} more")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--als", type=Path, default=DEFAULT_ALS)
    p.add_argument("--set-dir", type=Path, default=DEFAULT_SET_DIR)
    p.add_argument("--out", type=Path, default=_REPO / "labeling/fixtures/bb12_ground_truth.yaml")
    p.add_argument("--review", action="store_true", help="print review table only")
    p.add_argument("--include-all-clips", action="store_true")
    args = p.parse_args(argv)

    if not args.als.is_file():
        print(f"not found: {args.als}", file=sys.stderr)
        return 2
    if not args.set_dir.is_dir():
        print(f"not found: {args.set_dir}", file=sys.stderr)
        return 2

    try:
        gt, review = export_gt(args.als, args.set_dir, include_all=args.include_all_clips)
    except (OSError, ValueError) as e:
        print(f"export failed: {e}", file=sys.stderr)
        return 1

    print_review(review)
    if args.review:
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    title = args.set_dir.name
    match save(gt, args.out, title=title):
        case Err(e):
            print(f"write failed: {e.detail}", file=sys.stderr)
            return 1
        case Ok(path):
            print(f"wrote {len(gt.tracks)} tracks -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
