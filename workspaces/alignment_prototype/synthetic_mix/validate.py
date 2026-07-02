"""Validate synthetic window stats against BB12 targets."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from core.result import Err, Ok
from labeling.ground_truth.schema import GroundTruthSet, GroundTruthTrack, load

_REPO = Path(__file__).resolve().parents[3]
BB12_REFERENCE = _REPO / "labeling/fixtures/bb12_ground_truth.yaml"

# Per 5-min window targets (from docs/synthetic_mix_plan_v2_bb12.md).
WINDOW_TARGETS = {
    "bb12-lite": {
        "window_min": 2.5,
        "spans_min": 6,
        "spans_max": 14,
        "acap_ratio_min": 2.0,
        "seg_frac_min": 0.25,
        "loop_frac_min": 0.04,
        "overlap_min": 2,
    },
    "bb12-med": {
        "window_min": 4.0,
        "spans_min": 10,
        "spans_max": 22,
        "acap_ratio_min": 3.0,
        "seg_frac_min": 0.35,
        "loop_frac_min": 0.05,
        "overlap_min": 5,
    },
    "bb12-full": {
        "window_min": 6.0,
        "spans_min": 16,
        "spans_max": 35,
        "acap_ratio_min": 3.5,
        "seg_frac_min": 0.40,
        "loop_frac_min": 0.05,
        "overlap_min": 10,
    },
}


@dataclass(frozen=True)
class WindowStats:
    mix_id: str
    duration_s: float
    n_spans: int
    n_acap: int
    n_instr: int
    n_loops: int
    n_with_segments: int
    overlap_pairs: int

    @property
    def spans_per_min(self) -> float:
        return self.n_spans / max(self.duration_s / 60.0, 0.01)

    @property
    def seg_frac(self) -> float:
        return self.n_with_segments / max(self.n_spans, 1)

    @property
    def loop_frac(self) -> float:
        return self.n_loops / max(self.n_spans, 1)


def _overlap_pairs(tracks: tuple[GroundTruthTrack, ...]) -> int:
    n = 0
    for i, a in enumerate(tracks):
        for b in tracks[i + 1 :]:
            if a.set_start_s < b.set_end_s and b.set_start_s < a.set_end_s:
                n += 1
    return n


def stats_from_gt(gt: GroundTruthSet) -> WindowStats:
    tracks = gt.tracks
    dur = max((t.set_end_s for t in tracks), default=0.0)
    stems = Counter(t.claimed_stem for t in tracks)
    loops = sum(1 for t in tracks if t.is_loop)
    segs = sum(1 for t in tracks if t.ref_segments)
    return WindowStats(
        mix_id=gt.set_id,
        duration_s=dur,
        n_spans=len(tracks),
        n_acap=stems.get("acappella", 0),
        n_instr=stems.get("instrumental", 0),
        n_loops=loops,
        n_with_segments=segs,
        overlap_pairs=_overlap_pairs(tracks),
    )


def validate_window(
    gt: GroundTruthSet,
    *,
    curriculum: str = "bb12-lite",
) -> tuple[bool, list[str]]:
    """Return (ok, issues)."""
    t = WINDOW_TARGETS.get(curriculum, WINDOW_TARGETS["bb12-lite"])
    s = stats_from_gt(gt)
    issues: list[str] = []
    window_min = s.duration_s / 60.0
    if window_min < t["window_min"] * 0.85:
        issues.append(f"duration {window_min:.1f}min < target {t['window_min']}min")
    if s.n_spans < t["spans_min"]:
        issues.append(f"spans {s.n_spans} < min {t['spans_min']}")
    if s.n_spans > t["spans_max"]:
        issues.append(f"spans {s.n_spans} > max {t['spans_max']}")
    acap_ratio = s.n_acap / max(s.n_instr, 1)
    if acap_ratio < t["acap_ratio_min"]:
        issues.append(f"acap:instr {acap_ratio:.1f} < {t['acap_ratio_min']}")
    if s.seg_frac < t["seg_frac_min"]:
        issues.append(f"segment fraction {s.seg_frac:.2f} < {t['seg_frac_min']}")
    if s.loop_frac < t["loop_frac_min"]:
        issues.append(f"loop fraction {s.loop_frac:.2f} < {t['loop_frac_min']}")
    if s.overlap_pairs < t["overlap_min"]:
        issues.append(f"overlaps {s.overlap_pairs} < min {t['overlap_min']}")
    # Schema: loops must have ref_segments
    for tr in gt.tracks:
        if tr.is_loop and not tr.ref_segments:
            issues.append(f"{tr.slot_label}: is_loop without ref_segments")
    return (len(issues) == 0, issues)


def bb12_reference_stats() -> WindowStats | None:
    if not BB12_REFERENCE.is_file():
        return None
    match load(BB12_REFERENCE):
        case Err():
            return None
        case Ok(gt):
            return stats_from_gt(gt)


def format_stats(s: WindowStats) -> str:
    return (
        f"{s.mix_id}: {s.duration_s:.0f}s spans={s.n_spans} "
        f"(acap={s.n_acap} instr={s.n_instr} loops={s.n_loops} "
        f"seg={s.n_with_segments} overlap={s.overlap_pairs}) "
        f"{s.spans_per_min:.1f}/min"
    )
