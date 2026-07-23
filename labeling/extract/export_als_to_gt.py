"""Export BB12 (or any pulled set) Ableton `.als` → ground-truth YAML.

Reads the live `.als`, maps clips through 1-mix warp markers into mix seconds,
reads identity from each clip's file path, optionally attaches ``track_id`` when
the path exactly matches a row in `~/aligning/<set>/manifest.json` (pull
inventory only — not used for labels or timing), applies clip hygiene, and
writes a `*_ground_truth.yaml` consumable by `labeling.write_back_ground_truth`.

Usage (Mac, from repo root):

    venvs/audio/bin/python -m labeling.extract.export_als_to_gt \\
        --als "$HOME/aligning/_labeling/1fsnxchk/BB12 align Project/bb12_align.als" \\
        --set-dir "$HOME/aligning/1fsnxchk__Two Friends - Big Bootie Mix Volume 12" \\
        --out labeling/fixtures/bb12_ground_truth.yaml

Review table only:

    venvs/audio/bin/python -m labeling.extract.export_als_to_gt ... --review
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from labeling.als import load_als_xml, parse_layer_clips, track_display_name
from labeling.als.validate import has_errors, validate_session
from labeling.extract._shared import (
    DEFAULT_ALS,
    DEFAULT_SET_DIR,
    ClipRow,
    ReviewRow,
    _placeholder_note,
    collect_kept_clip_rows,
)
from labeling.schema import (
    GroundTruthSet,
    GroundTruthTrack,
    save,
)
from core.result import Err, Ok


def _to_gt_track(row: ClipRow) -> GroundTruthTrack:
    note = _placeholder_note(row.clip.path, row.clip.group_name)
    return GroundTruthTrack(
        label=row.display,
        track_id=row.recording_id,
        claimed_stem=row.claimed_stem,
        claimed_variant=row.claimed_variant,
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
        gain_curve=row.gain_curve,
        skip_training=row.skip_training,
        unalignable=note is not None,
        source_note=note,
        id_source=row.id_source,
    )


def export_gt(
    als_path: Path,
    set_dir: Path,
    *,
    include_all: bool = False,
) -> tuple[GroundTruthSet, list[ReviewRow]]:
    set_id, rows, review = collect_kept_clip_rows(
        als_path,
        set_dir,
        include_all=include_all,
    )
    return GroundTruthSet(
        set_id=set_id,
        tracks=tuple(_to_gt_track(r) for r in rows),
    ), review


def print_review(review: list[ReviewRow]) -> None:
    kept = sum(1 for r in review if r.action == "kept")
    dropped = sum(1 for r in review if r.action == "dropped")
    merged = sum(1 for r in review if r.action == "merged")
    unresolved = sorted(
        {r.slot for r in review if r.action == "kept" and not r.recording_id and r.slot}
    )
    print(
        f"review: kept={kept} dropped={dropped} merged={merged} unresolved_slots={len(unresolved)}"
    )
    if unresolved:
        print(
            "  unresolved:",
            ", ".join(unresolved[:20]),
            ("..." if len(unresolved) > 20 else ""),
        )
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


def _reader_drop_count(root) -> int:
    """Non-mix AudioClips in the tree minus those ``parse_layer_clips`` returned.

    Mirrors the reader's track filter (skips ``1-mix``/``2-mix``); any positive
    delta is clips the *total* reader dropped silently — no resolvable path,
    missing CurrentStart/CurrentEnd/Loop, or malformed numerics. Each dropped
    clip is a GT label that would vanish without a trace.
    """
    raw = 0
    for track_el in root.xpath(".//LiveSet/Tracks/*"):
        if track_el.tag != "AudioTrack":
            continue
        nm = track_display_name(track_el)
        if nm.startswith("1-mix") or nm.startswith("2-mix"):
            continue
        raw += len(track_el.xpath(".//AudioClip"))
    return raw - len(parse_layer_clips(root))


# Minimum fraction of exported GT tracks that must resolve a manifest
# recording_id. resolve_identity fills track_id ONLY on an exact manifest match;
# a stale .als (clip file-refs not relinked after a slot/tag rename) resolves ~0
# ids and produces a fixture that joins to nothing downstream (BB12 re-export
# 2026-07-12 resolved 1/163 and was written silently). A healthy export resolves
# ~99%; a broken one ~0% — 0.5 cleanly separates them.
ID_COVERAGE_MIN = 0.5


def id_coverage(tracks) -> tuple[int, int, float]:
    """(content-bound, total, fraction) of GT tracks whose id came from content."""
    total = len(tracks)
    resolved = sum(1 for t in tracks if getattr(t, "id_source", "") == "content")
    return resolved, total, (resolved / total if total else 1.0)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--als", type=Path, default=DEFAULT_ALS)
    p.add_argument("--set-dir", type=Path, default=DEFAULT_SET_DIR)
    p.add_argument(
        "--out", type=Path, default=_REPO / "labeling/fixtures/bb12_ground_truth.yaml"
    )
    p.add_argument("--review", action="store_true", help="print review table only")
    p.add_argument("--include-all-clips", action="store_true")
    p.add_argument(
        "--allow-invalid",
        action="store_true",
        help="export even when the .als has error-severity diagnostics "
        "(clips the reader silently drops / malformed timing). Default: refuse.",
    )
    args = p.parse_args(argv)

    if not args.als.is_file():
        print(f"not found: {args.als}", file=sys.stderr)
        return 2
    if not args.set_dir.is_dir():
        print(f"not found: {args.set_dir}", file=sys.stderr)
        return 2

    # Gate: the reader (parse_layer_clips) is *total* — it silently drops clips
    # with no path / missing CurrentStart|CurrentEnd|Loop / malformed numerics,
    # any one of which is a silently-missing GT label. Two blocking signals:
    #   (1) validate error-severity diagnostics, and
    #   (2) a direct drop count — raw non-mix AudioClips vs what the reader
    #       returned. This catches drops `validate` grades below "error"
    #       (clip-incomplete is a warning) or doesn't model (missing Path).
    # Both fail-fast at the edge; --allow-invalid overrides. Warnings print but
    # do not block.
    root = load_als_xml(args.als)
    diags = validate_session(root)
    if diags:
        errs = [d for d in diags if d.severity == "error"]
        warns = [d for d in diags if d.severity != "error"]
        print(
            f"validate: {len(errs)} error(s), {len(warns)} warning(s) in {args.als.name}",
            file=sys.stderr,
        )
        for d in diags:
            print(f"  {d}", file=sys.stderr)

    dropped = _reader_drop_count(root)
    if dropped:
        print(
            f"reader silently dropped {dropped} non-mix clip(s) "
            "(no path / missing timing / malformed numerics) — each is a lost GT label",
            file=sys.stderr,
        )

    if (has_errors(diags) or dropped) and not args.allow_invalid:
        print(
            "REFUSING to export: the .als has clips the reader drops silently "
            "(bad/missing labels). Fix the .als, or pass --allow-invalid to override.",
            file=sys.stderr,
        )
        return 1

    try:
        gt, review = export_gt(
            args.als, args.set_dir, include_all=args.include_all_clips
        )
    except (OSError, ValueError) as e:
        print(f"export failed: {e}", file=sys.stderr)
        return 1

    print_review(review)
    if args.review:
        return 0

    # Gate: identity coverage. A stale .als whose clip file-refs no longer match
    # the manifest (e.g. not relinked after a slot/tag rename) resolves almost no
    # recording_ids; the resulting fixture joins to nothing in the scorer and
    # would silently corrupt canonical set_ground_truth on write-back. Refuse.
    resolved, total, coverage = id_coverage(gt.tracks)
    if total and coverage < ID_COVERAGE_MIN and not args.allow_invalid:
        abstained = [
            t.slot_label or t.label
            for t in gt.tracks
            if getattr(t, "id_source", "") != "content"
        ]
        print(
            f"REFUSING to export: only {resolved}/{total} tracks ({coverage:.0%}) "
            f"content-bound (min {ID_COVERAGE_MIN:.0%}). Abstained: "
            f"{', '.join(abstained[:30])}. Rebuild content_catalog.json (re-pull), "
            "or pass --allow-invalid to override.",
            file=sys.stderr,
        )
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    title = args.set_dir.name
    match save(gt, args.out, title=title):
        case Err(e):
            print(f"write failed: {e.detail}", file=sys.stderr)
            return 1
        case Ok(path):
            print(f"wrote {len(gt.tracks)} tracks -> {path}")
            _write_inventory_bundle(gt, path)
    return 0


def _write_inventory_bundle(gt, yaml_path: Path) -> None:
    """Sidecar JSON for alignment training (ref_source + slot labels)."""
    from collections import Counter

    bundle_path = yaml_path.with_suffix(".inventory.json")
    rows = []
    for t in gt.tracks:
        rows.append(
            {
                "label": t.slot_label or t.label,
                "track_id": t.track_id,
                "claimed_stem": t.claimed_stem,
                "ref_source": t.ref_source,
            }
        )
    payload = {
        "set_id": gt.set_id,
        "tracks": rows,
        "ref_source_counts": dict(Counter(t.ref_source for t in gt.tracks)),
    }
    bundle_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote inventory bundle -> {bundle_path}")


if __name__ == "__main__":
    sys.exit(main())
