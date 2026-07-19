from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from workspaces.alignment_prototype.path_decode import gt_placement_onset
from workspaces.alignment_prototype.score_timeline_vs_gt import (
    norm_slot,
    score_spans,
)

_REPO = Path(__file__).resolve().parents[3]
_OUT = Path(__file__).resolve().parent / "out"
_ALN_OUT = _REPO / "workspaces" / "alignment_prototype" / "out"

SET_META = {
    "1fsnxchk": ("BB12", _REPO / "labeling/fixtures/bb12_ground_truth.yaml"),
    "2nvzlh2k": ("BB11", _REPO / "labeling/fixtures/bb11_ground_truth.yaml"),
}


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    set_id: str
    set_label: str
    slot: str
    recording_id: str
    claimed_stem: str
    span_class: str
    place_err_s: float
    pred_set_start_s: float
    gt_set_start_s: float
    gt_audible_onset_s: float
    gt_ref_start_s: float
    gt_set_end_s: float
    gt_ref_end_s: float | None
    tempo_ratio: float
    pitch_shift_semi: int
    ref_segments: tuple[tuple[float, float, float], ...]
    source_timeline: str
    taxonomy_tags: tuple[str, ...]


def _rank_candidates(rows: list[dict], *, min_place_err_s: float) -> list[dict]:
    kept = [
        r
        for r in rows
        if r.get("id_correct") is True
        and r.get("place_err_s") is not None
        and float(r["place_err_s"]) >= min_place_err_s
    ]
    return sorted(kept, key=lambda r: float(r["place_err_s"]), reverse=True)


def _gt_ref_segments(row: dict) -> tuple[tuple[float, float, float], ...]:
    segs = row.get("ref_segments")
    if segs:
        return tuple(
            (float(s["mix_start_s"]), float(s["ref_start_s"]), float(s["ref_end_s"]))
            for s in segs
        )
    ref_end = row.get("ref_end_s")
    return (
        (
            float(row["set_start_s"]),
            float(row["ref_start_s"]),
            float(ref_end) if ref_end is not None else float(row["ref_start_s"]),
        ),
    )


def _gt_by_recording_id(gt_path: Path) -> dict[str, list[dict]]:
    gt_doc = yaml.safe_load(gt_path.read_text())
    gt_rows = [r for r in gt_doc["tracks"] if str(r.get("slot_label")) != "mix"]
    by_tid: dict[str, list[dict]] = {}
    for row in gt_rows:
        if row.get("track_id"):
            by_tid.setdefault(str(row["track_id"]), []).append(row)
    return by_tid


def _timeline_spans_by_key(timeline_path: Path) -> dict[tuple[str, str], dict]:
    timeline = json.loads(timeline_path.read_text())
    return {
        (norm_slot(s["slot_label"]), s["recording_id"]): s for s in timeline["spans"]
    }


def _score_set(
    set_id: str,
    set_label: str,
    gt_path: Path,
    timeline_path: Path,
) -> list[dict]:
    scores = score_spans(set_id, timeline_path, gt_path=gt_path)
    spans_by_key = _timeline_spans_by_key(timeline_path)
    gt_by_tid = _gt_by_recording_id(gt_path)
    source = str(timeline_path)
    return [
        {
            "set_id": set_id,
            "set_label": set_label,
            "slot": sc.slot,
            "recording_id": sc.recording_id,
            "claimed_stem": sc.stem or "regular",
            "span_class": sc.span_class or "linear",
            "id_correct": sc.id_correct,
            "place_err_s": sc.place_err_s,
            "source_timeline": source,
            "_span": spans_by_key.get((sc.slot, sc.recording_id)),
            "_gt_by_tid": gt_by_tid,
        }
        for sc in scores
    ]


def _enrich_case(row: dict) -> CaseRecord:
    span = row["_span"]
    if span is None:
        raise ValueError(
            f"no timeline span for slot={row['slot']} recording_id={row['recording_id']}"
        )
    gt_rows = row["_gt_by_tid"].get(row["recording_id"], [])
    if not gt_rows:
        raise ValueError(f"no GT row for recording_id={row['recording_id']}")
    pred_start = float(span["set_start_s"])
    gt_row = min(gt_rows, key=lambda r: abs(float(r["set_start_s"]) - pred_start))
    return CaseRecord(
        case_id=f"{row['set_label'].lower()}_{row['slot']}",
        set_id=row["set_id"],
        set_label=row["set_label"],
        slot=row["slot"],
        recording_id=row["recording_id"],
        claimed_stem=row["claimed_stem"],
        span_class=row["span_class"],
        place_err_s=float(row["place_err_s"]),
        pred_set_start_s=pred_start,
        gt_set_start_s=float(gt_row["set_start_s"]),
        gt_audible_onset_s=gt_placement_onset(gt_row),
        gt_ref_start_s=float(gt_row["ref_start_s"]),
        gt_set_end_s=float(gt_row["set_end_s"]),
        gt_ref_end_s=(
            float(gt_row["ref_end_s"]) if gt_row.get("ref_end_s") is not None else None
        ),
        tempo_ratio=float(gt_row.get("tempo_ratio") or 1.0),
        pitch_shift_semi=int(gt_row.get("pitch_shift_semi") or 0),
        ref_segments=_gt_ref_segments(gt_row),
        source_timeline=row["source_timeline"],
        taxonomy_tags=(),
    )


def resolve_timeline(set_id: str, override: Path | None = None) -> Path:
    if override is not None:
        return override
    preferred = [
        _ALN_OUT / f"{set_id}_agentic_timeline.json",
        _ALN_OUT / f"{set_id}_predicted_timeline_lt.json",
    ]
    for path in preferred:
        if path.is_file():
            return path
    hits = sorted(_ALN_OUT.glob(f"{set_id}_predicted_timeline*.json"))
    if not hits:
        raise FileNotFoundError(
            f"no timeline for {set_id} under {_ALN_OUT} — run "
            f"`make race SETS={set_id} DRIVERS=agentic` or `make align SET={set_id}`"
        )
    return hits[0]


def select_hard_cases(
    *,
    n: int = 12,
    min_place_err_s: float = 15.0,
    timeline_by_set: dict[str, Path] | None = None,
) -> list[CaseRecord]:
    overrides = timeline_by_set or {}
    candidates: list[dict] = []
    for set_id, (set_label, gt_path) in SET_META.items():
        timeline_path = overrides.get(set_id) or resolve_timeline(set_id)
        candidates.extend(_score_set(set_id, set_label, gt_path, timeline_path))
    ranked = _rank_candidates(candidates, min_place_err_s=min_place_err_s)[:n]
    return [_enrich_case(row) for row in ranked]


def dump_cases(cases: list[CaseRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(c) for c in cases], indent=2) + "\n")
