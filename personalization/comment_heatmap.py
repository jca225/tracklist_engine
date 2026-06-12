"""Comment density heatmap vs mix structure / GT section boundaries."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sqlite3

from core.result import Err, Ok, Result
from eda.alignment.boundaries import gt_section_starts_s, score_boundaries, seconds_to_bar_indices
from labeling.ground_truth.schema import GroundTruthSet, load


@dataclass(frozen=True)
class HeatmapResult:
    mix_id: str
    n_comments: int
    n_with_position: int
    mix_duration_s: float
    bin_width_s: float
    bin_counts: tuple[int, ...]
    bin_centers_s: tuple[float, ...]
    peak_bin_centers_s: tuple[float, ...]
    gt_section_starts_s: tuple[float, ...] = ()
    gt_alignment: dict[str, float | int] | None = None


def _load_comments(conn: sqlite3.Connection, mix_id: str) -> list[tuple[int, str]]:
    rows = conn.execute(
        """
        SELECT mix_position_ms, body FROM sc_mix_comments
        WHERE mix_id = ? AND mix_position_ms IS NOT NULL
        ORDER BY mix_position_ms
        """,
        (mix_id,),
    ).fetchall()
    return [(int(r["mix_position_ms"]), str(r["body"])) for r in rows]


def build_heatmap(
    conn: sqlite3.Connection,
    mix_id: str,
    *,
    bin_width_s: float = 30.0,
    mix_duration_s: float | None = None,
    gt: GroundTruthSet | None = None,
    bar_start_s: np.ndarray | None = None,
    peak_percentile: float = 90.0,
) -> HeatmapResult:
    comments = _load_comments(conn, mix_id)
    positions_s = [ms / 1000.0 for ms, _ in comments]
    if mix_duration_s is None:
        mix_duration_s = max(positions_s) + bin_width_s if positions_s else 3600.0

    n_bins = max(int(np.ceil(mix_duration_s / bin_width_s)), 1)
    counts = np.zeros(n_bins, dtype=np.int64)
    for t in positions_s:
        idx = min(int(t / bin_width_s), n_bins - 1)
        counts[idx] += 1

    centers = tuple((i + 0.5) * bin_width_s for i in range(n_bins))
    thresh = float(np.percentile(counts, peak_percentile)) if counts.sum() > 0 else 0.0
    peaks = tuple(centers[i] for i in range(n_bins) if counts[i] >= thresh and counts[i] > 0)

    gt_starts: tuple[float, ...] = ()
    alignment: dict[str, float | int] | None = None
    if gt is not None:
        gt_starts = gt_section_starts_s(gt)
        if bar_start_s is not None and len(bar_start_s) > 0:
            gt_bars = seconds_to_bar_indices(gt_starts, bar_start_s)
            peak_bars = seconds_to_bar_indices(peaks, bar_start_s)
            tol = max(int(round(60.0 / (mix_duration_s / len(bar_start_s)))), 2)
            scored = score_boundaries(peak_bars, gt_bars, tolerance_bars=tol)
            alignment = {
                "tolerance_bars": scored.tolerance_bars,
                "n_gt": scored.n_gt,
                "n_pred_peaks": scored.n_pred,
                "tp": scored.tp,
                "precision": round(scored.precision, 4),
                "recall": round(scored.recall, 4),
                "f1": round(scored.f1, 4),
            }

    return HeatmapResult(
        mix_id=mix_id,
        n_comments=len(comments),
        n_with_position=len(positions_s),
        mix_duration_s=mix_duration_s,
        bin_width_s=bin_width_s,
        bin_counts=tuple(int(c) for c in counts),
        bin_centers_s=centers,
        peak_bin_centers_s=peaks,
        gt_section_starts_s=gt_starts,
        gt_alignment=alignment,
    )


def heatmap_to_dict(result: HeatmapResult) -> dict:
    top_bins = sorted(
        zip(result.bin_centers_s, result.bin_counts),
        key=lambda x: x[1],
        reverse=True,
    )[:15]
    return {
        "mix_id": result.mix_id,
        "n_comments": result.n_comments,
        "n_with_position": result.n_with_position,
        "mix_duration_s": result.mix_duration_s,
        "bin_width_s": result.bin_width_s,
        "top_comment_bins": [
            {"center_s": c, "center_mmss": _mmss(c), "count": n} for c, n in top_bins if n > 0
        ],
        "peak_bin_centers_s": result.peak_bin_centers_s,
        "gt_section_starts_s": result.gt_section_starts_s,
        "gt_alignment": result.gt_alignment,
        "bin_counts": list(result.bin_counts),
        "bin_centers_s": list(result.bin_centers_s),
    }


def _mmss(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


def run_heatmap_analysis(
    db_path: Path,
    mix_id: str,
    *,
    gt_yaml: Path | None = None,
    measure_times_json: Path | None = None,
    bin_width_s: float = 30.0,
    out_path: Path | None = None,
) -> Result[dict, str]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    gt: GroundTruthSet | None = None
    if gt_yaml is not None:
        match load(gt_yaml):
            case Ok(g):
                gt = g
            case Err(e):
                conn.close()
                return Err(f"gt load failed: {e}")

    bar_start_s: np.ndarray | None = None
    mix_duration_s: float | None = None
    if measure_times_json is not None and measure_times_json.is_file():
        data = json.loads(measure_times_json.read_text())
        if isinstance(data, list):
            bar_start_s = np.asarray(data, dtype=np.float64)
        else:
            bar_start_s = np.asarray(data["bar_start_s"], dtype=np.float64)
        mix_duration_s = float(bar_start_s[-1]) + 30.0

    result = build_heatmap(
        conn,
        mix_id,
        bin_width_s=bin_width_s,
        mix_duration_s=mix_duration_s,
        gt=gt,
        bar_start_s=bar_start_s,
    )
    conn.close()
    payload = heatmap_to_dict(result)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
    return Ok(payload)
