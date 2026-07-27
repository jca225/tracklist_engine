"""CLI: select hard placement cases, compute ridge contrast, write heatmaps + table.

    venvs/audio/bin/python -m eda.alignment.ridge_diagnostic.run \\
        --n 12 --min-place-err-s 15 --bin-s 0.5 --contrast-threshold 2.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core.contracts import MANIFEST_FILENAME, load_manifest
from eda.alignment.ridge_diagnostic.cases import (
    SET_META,
    CaseRecord,
    _enrich_case,
    _rank_candidates,
    _score_set,
    dump_cases,
    resolve_timeline,
)
from eda.alignment.ridge_diagnostic.features import (
    Channel,
    _audio_duration_s,
    _clamp_crop_bounds,
    _mix_full_path,
    _ref_crop_bounds,
    _resolve_ref_path,
    compute_panel,
)
from eda.alignment.ridge_diagnostic.plot import (
    DEFAULT_CONTRAST_THRESHOLD,
    save_heatmap,
    write_contrast_table,
)
from eda.alignment.ridge_diagnostic.ridge import gt_diagonal_mask, ridge_contrast
from alignment.refine_ref_offsets import find_aligning_dir

_PKG = Path(__file__).resolve().parent
_OUT = _PKG / "out"
_HEATMAPS = _OUT / "heatmaps"
_CHANNELS: tuple[Channel, ...] = ("hubert", "chroma", "fp_hit", "instr_stem")


def _parse_timeline_overrides(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"expected SET=PATH, got {item!r}")
        set_id, path_str = item.split("=", 1)
        out[set_id.strip()] = Path(path_str.strip())
    return out


def _load_cases(path: Path) -> list[CaseRecord]:
    raw = json.loads(path.read_text())
    out: list[CaseRecord] = []
    for row in raw:
        rec = dict(row)
        rec["ref_segments"] = tuple(tuple(seg) for seg in rec["ref_segments"])
        rec["taxonomy_tags"] = tuple(rec.get("taxonomy_tags") or ())
        out.append(CaseRecord(**rec))
    return out


def _select_cases(
    *,
    n: int,
    min_place_err_s: float,
    sets: list[str],
    timeline_by_set: dict[str, Path],
) -> list[CaseRecord]:
    candidates: list[dict] = []
    for set_id in sets:
        if set_id not in SET_META:
            raise ValueError(f"unknown set_id {set_id!r}; known: {sorted(SET_META)}")
        set_label, gt_path = SET_META[set_id]
        timeline_path = timeline_by_set.get(set_id) or resolve_timeline(set_id)
        candidates.extend(_score_set(set_id, set_label, gt_path, timeline_path))
    ranked = _rank_candidates(candidates, min_place_err_s=min_place_err_s)[:n]
    return [_enrich_case(row) for row in ranked]


def _offsets_s(case: CaseRecord) -> tuple[float, ...]:
    if case.ref_segments:
        return tuple(
            mix_start - ref_start for mix_start, ref_start, _ in case.ref_segments
        )
    return (case.gt_audible_onset_s - case.gt_ref_start_s,)


def _crop_origins(case: CaseRecord, *, pad_s: float) -> tuple[float, float]:
    """Mix/ref crop lows matching ``compute_panel`` (pad-based, clamped)."""
    set_dir = find_aligning_dir(case.set_id)
    mix_full = _mix_full_path(set_dir)
    manifest = load_manifest(set_dir / MANIFEST_FILENAME)
    ref_path = _resolve_ref_path(set_dir, manifest.by_track_id(), case)
    if ref_path is None:
        raise FileNotFoundError(
            f"no ref audio for recording_id={case.recording_id} slot={case.slot}"
        )

    mix_lo = max(0.0, case.gt_audible_onset_s - pad_s)
    mix_hi = case.gt_set_end_s + pad_s
    mix_lo, _mix_hi = _clamp_crop_bounds(mix_lo, mix_hi, _audio_duration_s(mix_full))
    ref_lo, ref_hi = _ref_crop_bounds(case, pad_s)
    ref_lo, _ref_hi = _clamp_crop_bounds(ref_lo, ref_hi, _audio_duration_s(ref_path))
    return mix_lo, ref_lo


def _process_case(
    case: CaseRecord,
    *,
    bin_s: float,
    pad_s: float,
    contrast_threshold: float,
) -> tuple[list[dict], dict[str, str], list[str]]:
    """Return contrast rows, channel verdicts, and skip notes."""
    try:
        panel = compute_panel(case, bin_s=bin_s, pad_s=pad_s)
    except FileNotFoundError as exc:
        return [], {}, [str(exc)]

    mix_origin_s, ref_origin_s = _crop_origins(case, pad_s=pad_s)
    offsets = _offsets_s(case)
    rows: list[dict] = []
    channel_verdicts: dict[str, str] = {}
    skipped: list[str] = []

    for channel in _CHANNELS:
        sim = panel.get(channel)
        if sim is None:
            skipped.append(channel)
            continue

        m = sim.M
        mask = gt_diagonal_mask(
            m.shape,
            offsets_s=offsets,
            bin_s=sim.mix_bin_s,
            mix_origin_s=mix_origin_s,
            ref_origin_s=ref_origin_s,
        )
        contrast = ridge_contrast(m, mask)
        verdict = "ridge_present" if contrast >= contrast_threshold else "ridge_absent"
        channel_verdicts[channel] = verdict

        title = (
            f"{case.case_id} · {channel} · contrast={contrast:.2f} · "
            f"place_err={case.place_err_s:.1f}s"
        )
        png = _HEATMAPS / f"{case.case_id}__{channel}.png"
        save_heatmap(m, mask, png, title=title, contrast=contrast)

        rows.append(
            {
                "case_id": case.case_id,
                "set_id": case.set_id,
                "slot": case.slot,
                "claimed_stem": case.claimed_stem,
                "span_class": case.span_class,
                "place_err_s": case.place_err_s,
                "channel": channel,
                "ridge_contrast": contrast,
            }
        )

    return rows, channel_verdicts, skipped


def _case_verdict(channel_verdicts: dict[str, str]) -> str:
    if any(v == "ridge_present" for v in channel_verdicts.values()):
        return "decoder_wall"
    return "representation_wall"


def _print_summary(
    case_summaries: list[tuple[CaseRecord, dict[str, str], list[str], str]],
) -> tuple[int, int]:
    decoder = 0
    representation = 0
    print("\n=== ridge diagnostic summary ===\n")
    for case, channel_verdicts, skipped, verdict in case_summaries:
        present = [ch for ch, v in channel_verdicts.items() if v == "ridge_present"]
        absent = [ch for ch, v in channel_verdicts.items() if v == "ridge_absent"]
        skip_note = f" skipped={','.join(skipped)}" if skipped else ""
        if present:
            ridge_str = "ridge_present: " + ", ".join(present)
        else:
            ridge_str = "ridge_present: (none)"
        if absent:
            ridge_str += f" | absent: {', '.join(absent)}"
        print(
            f"{case.case_id}  place_err={case.place_err_s:.1f}s  "
            f"{verdict}  {ridge_str}{skip_note}"
        )
        if verdict == "decoder_wall":
            decoder += 1
        else:
            representation += 1

    n = len(case_summaries)
    print(
        f"\naggregate: decoder_wall={decoder}/{n}  representation_wall={representation}/{n}"
    )
    return decoder, representation


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Placement ridge diagnostic — hard cases, heatmaps, contrast table."
    )
    ap.add_argument("--n", type=int, default=12, help="max hard cases to select")
    ap.add_argument(
        "--min-place-err-s",
        type=float,
        default=15.0,
        help="minimum placement error (seconds) for case selection",
    )
    ap.add_argument(
        "--bin-s", type=float, default=0.5, help="similarity matrix bin width"
    )
    ap.add_argument(
        "--pad-s",
        type=float,
        default=15.0,
        help="audio crop padding around GT span (must match compute_panel)",
    )
    ap.add_argument(
        "--contrast-threshold",
        type=float,
        default=DEFAULT_CONTRAST_THRESHOLD,
        help="ridge_contrast ≥ threshold → ridge_present (looking aid only)",
    )
    ap.add_argument(
        "--cases-json",
        type=Path,
        default=None,
        help="reload prior out/cases.json instead of selecting",
    )
    ap.add_argument(
        "--sets",
        default=",".join(sorted(SET_META)),
        help="comma-separated set ids for selection (default: BB11+BB12)",
    )
    ap.add_argument(
        "--timeline",
        action="append",
        default=[],
        metavar="SET=PATH",
        help="override timeline JSON for a set (repeatable)",
    )
    args = ap.parse_args(argv)

    timeline_by_set = _parse_timeline_overrides(args.timeline)
    sets = [s.strip() for s in args.sets.split(",") if s.strip()]

    if args.cases_json is not None:
        cases = _load_cases(args.cases_json)
    else:
        cases = _select_cases(
            n=args.n,
            min_place_err_s=args.min_place_err_s,
            sets=sets,
            timeline_by_set=timeline_by_set,
        )

    if not cases:
        print(
            "no cases selected — loosen --min-place-err-s or check timelines",
            file=sys.stderr,
        )
        return 1

    dump_cases(cases, _OUT / "cases.json")

    all_rows: list[dict] = []
    case_summaries: list[tuple[CaseRecord, dict[str, str], list[str], str]] = []

    for case in cases:
        rows, channel_verdicts, skipped = _process_case(
            case,
            bin_s=args.bin_s,
            pad_s=args.pad_s,
            contrast_threshold=args.contrast_threshold,
        )
        all_rows.extend(rows)
        verdict = (
            _case_verdict(channel_verdicts)
            if channel_verdicts
            else "representation_wall"
        )
        case_summaries.append((case, channel_verdicts, skipped, verdict))

    write_contrast_table(
        all_rows,
        _OUT / "contrast_table.tsv",
        contrast_threshold=args.contrast_threshold,
    )

    decoder, representation = _print_summary(case_summaries)
    print(f"\nwrote {_OUT / 'cases.json'}")
    print(f"wrote {_OUT / 'contrast_table.tsv'}")
    print(
        f"wrote heatmaps under {_HEATMAPS}/ ({len(list(_HEATMAPS.glob('*.png')))} PNGs)"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
