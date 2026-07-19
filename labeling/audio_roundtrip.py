"""Denotational audio round-trip — the GT release ruler.

Compiler analogy: hand ``.als`` (source) → export GT (IR) → offline-render both
→ assert the labeled layers sound the same. XML ``parse∘print`` cannot catch
distant-clip merges; this can.

Usage::

    venvs/audio/bin/python -m labeling.audio_roundtrip \\
        --als path/to/hand.als --set-dir ~/aligning/<set>__... \\
        --yaml labeling/fixtures/bb12_ground_truth.yaml

    # Or export GT in-memory from the .als (no yaml):
    venvs/audio/bin/python -m labeling.audio_roundtrip --als ... --set-dir ...
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from labeling.als.render_offline import (
    DEFAULT_SR,
    RenderClip,
    resolve_clip_audio_path,
    sum_render_clips,
)
from labeling.export_als_to_gt import ClipRow, collect_kept_clip_rows, export_gt
from labeling.ground_truth.schema import (
    GroundTruthSet,
    GroundTruthTrack,
    load as load_gt,
)
from labeling.als.identity import build_manifest_index
from core.contracts import MANIFEST_FILENAME
from core.result import Err, Ok

# Tuned so a 12s silence gap filled by a bad merge fails; normal export passes.
DEFAULT_MIN_CORR = 0.985
DEFAULT_MAX_RMS_DIFF = 0.08
DEFAULT_LAG_MS = 50.0


@dataclass(frozen=True)
class AudioCompareResult:
    ok: bool
    corr: float
    rms_diff: float
    lag_samples: int
    detail: str

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "corr": self.corr,
            "rms_diff": self.rms_diff,
            "lag_samples": self.lag_samples,
            "detail": self.detail,
        }


def assert_audio_equivalent(
    a: np.ndarray,
    b: np.ndarray,
    *,
    min_corr: float = DEFAULT_MIN_CORR,
    max_rms_diff: float = DEFAULT_MAX_RMS_DIFF,
    lag_ms: float = DEFAULT_LAG_MS,
    sr: int = DEFAULT_SR,
) -> AudioCompareResult:
    """Loudness-normalize, search small lag, compare corr + residual RMS."""
    if a.size == 0 and b.size == 0:
        return AudioCompareResult(True, 1.0, 0.0, 0, "both empty")
    if a.size == 0 or b.size == 0:
        return AudioCompareResult(
            False,
            0.0,
            1.0,
            0,
            f"length mismatch empty vs non-empty ({a.size}/{b.size})",
        )

    def _norm(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        if peak < 1e-9:
            return x
        return x / peak

    aa, bb = _norm(a), _norm(b)
    n = max(aa.size, bb.size)
    if aa.size < n:
        aa = np.pad(aa, (0, n - aa.size))
    if bb.size < n:
        bb = np.pad(bb, (0, n - bb.size))

    max_lag = max(0, int(round(lag_ms * 1e-3 * sr)))
    best_corr = -1.0
    best_lag = 0
    best_rms = 1.0
    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            seg_a, seg_b = aa, bb
        elif lag > 0:
            seg_a, seg_b = aa[lag:], bb[: n - lag]
        else:
            seg_a, seg_b = aa[: n + lag], bb[-lag:]
        if seg_a.size < sr // 4:  # need ≥250 ms overlap
            continue
        denom = float(np.linalg.norm(seg_a) * np.linalg.norm(seg_b))
        if denom < 1e-12:
            corr = 0.0
        else:
            corr = float(np.dot(seg_a, seg_b) / denom)
        diff = seg_a - seg_b
        rms = float(np.sqrt(np.mean(diff * diff)))
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
            best_rms = rms

    ok = best_corr >= min_corr and best_rms <= max_rms_diff
    detail = (
        f"corr={best_corr:.4f} (min {min_corr}) rms_diff={best_rms:.4f} "
        f"(max {max_rms_diff}) lag={best_lag} samples"
    )
    return AudioCompareResult(ok, best_corr, best_rms, best_lag, detail)


def _ratio_for_row(row: ClipRow) -> float:
    if row.tempo_ratio and row.tempo_ratio > 0:
        return float(row.tempo_ratio)
    set_span = row.set_end_s - row.set_start_s
    ref_span = row.ref_end_s - row.ref_start_s
    if set_span > 0 and ref_span > 0:
        return ref_span / set_span
    return 1.0


def render_clips_from_arrangement(als: Path, set_dir: Path) -> list[RenderClip]:
    """Pre-merge arrangement clips — Live denotation for the round-trip LHS."""
    _, rows, _ = collect_kept_clip_rows(als, set_dir, arrangement_denotation=True)
    out: list[RenderClip] = []
    for row in rows:
        if row.skip_training:
            continue
        path = resolve_clip_audio_path(row.clip.path, set_dir)
        if path is None:
            continue
        out.append(
            RenderClip(
                audio_path=path,
                set_start_s=row.set_start_s,
                set_end_s=row.set_end_s,
                ref_start_s=row.ref_start_s,
                tempo_ratio=_ratio_for_row(row),
                gain_curve=row.gain_curve,
                pitch_shift_semi=row.pitch_shift_semi,
            )
        )
    return out


def _resolve_gt_path(track: GroundTruthTrack, set_dir: Path) -> Path | None:
    """Best-effort path for a GT row via manifest slot / stem dirs."""
    manifest_path = set_dir / MANIFEST_FILENAME
    if manifest_path.is_file():
        idx = build_manifest_index(manifest_path)
        slot = (track.slot_label or "").strip()
        if slot and slot in idx.by_slot:
            local = idx.by_slot[slot].local_path
            if local:
                p = resolve_clip_audio_path(local, set_dir)
                if p is not None:
                    return p
        # Path keyed by basename from any slot with matching track_id.
        if track.track_id:
            for ms in idx.rows:
                if ms.track_id == track.track_id and ms.local_path:
                    p = resolve_clip_audio_path(ms.local_path, set_dir)
                    if p is not None:
                        return p
    return None


def render_clips_from_gt(
    gt: GroundTruthSet,
    set_dir: Path,
    *,
    path_by_slot: dict[str, Path] | None = None,
) -> list[RenderClip]:
    """GT IR → RenderClips. ``path_by_slot`` supplies paths when YAML has none."""
    path_by_slot = path_by_slot or {}
    out: list[RenderClip] = []
    for t in gt.tracks:
        if t.skip_training or t.unalignable:
            continue
        slot = (t.slot_label or "").strip()
        path = path_by_slot.get(slot) or _resolve_gt_path(t, set_dir)
        if path is None:
            continue
        if t.ref_segments:
            for seg in t.ref_segments:
                set_span = seg.ref_end_s - seg.ref_start_s
                # Segment duration on the mix is encoded by tempo_ratio of parent
                # when available; else assume 1:1 ref/set for the segment span.
                ratio = t.tempo_ratio if t.tempo_ratio and t.tempo_ratio > 0 else 1.0
                set_dur = set_span / ratio if ratio > 0 else set_span
                out.append(
                    RenderClip(
                        audio_path=path,
                        set_start_s=seg.mix_start_s,
                        set_end_s=seg.mix_start_s + set_dur,
                        ref_start_s=seg.ref_start_s,
                        tempo_ratio=ratio,
                        gain_curve=(),
                        pitch_shift_semi=t.pitch_shift_semi,
                    )
                )
            continue
        set_span = t.set_end_s - t.set_start_s
        if set_span <= 0:
            continue
        if t.tempo_ratio and t.tempo_ratio > 0:
            ratio = float(t.tempo_ratio)
        elif t.ref_end_s is not None:
            ratio = (t.ref_end_s - t.ref_start_s) / set_span
        else:
            ratio = 1.0
        out.append(
            RenderClip(
                audio_path=path,
                set_start_s=t.set_start_s,
                set_end_s=t.set_end_s,
                ref_start_s=t.ref_start_s,
                tempo_ratio=ratio,
                gain_curve=t.gain_curve,
                pitch_shift_semi=t.pitch_shift_semi,
            )
        )
    return out


def path_map_from_arrangement(als: Path, set_dir: Path) -> dict[str, Path]:
    """slot_label → audio path from arrangement rows (for GT path fill-in)."""
    _, rows, _ = collect_kept_clip_rows(als, set_dir, arrangement_denotation=True)
    out: dict[str, Path] = {}
    for row in rows:
        slot = (row.slot_label or "").strip()
        if not slot:
            continue
        path = resolve_clip_audio_path(row.clip.path, set_dir)
        if path is not None:
            out[slot] = path
    return out


def compare_als_to_gt(
    als: Path,
    set_dir: Path,
    gt: GroundTruthSet,
    *,
    min_corr: float = DEFAULT_MIN_CORR,
    max_rms_diff: float = DEFAULT_MAX_RMS_DIFF,
    sr: int = DEFAULT_SR,
) -> AudioCompareResult:
    arr_clips = render_clips_from_arrangement(als, set_dir)
    paths = path_map_from_arrangement(als, set_dir)
    gt_clips = render_clips_from_gt(gt, set_dir, path_by_slot=paths)
    if not arr_clips:
        return AudioCompareResult(False, 0.0, 1.0, 0, "no arrangement clips resolved")
    if not gt_clips:
        return AudioCompareResult(False, 0.0, 1.0, 0, "no GT clips resolved")
    cache: dict[Path, np.ndarray] = {}
    a = sum_render_clips(arr_clips, sr=sr, audio_cache=cache)
    b = sum_render_clips(gt_clips, sr=sr, audio_cache=cache)
    return assert_audio_equivalent(
        a, b, min_corr=min_corr, max_rms_diff=max_rms_diff, sr=sr
    )


def run_roundtrip(
    als: Path,
    set_dir: Path,
    *,
    yaml_path: Path | None = None,
    min_corr: float = DEFAULT_MIN_CORR,
    max_rms_diff: float = DEFAULT_MAX_RMS_DIFF,
) -> AudioCompareResult:
    if yaml_path is not None:
        match load_gt(yaml_path):
            case Err(e):
                return AudioCompareResult(False, 0.0, 1.0, 0, f"yaml: {e.detail}")
            case Ok(gt):
                pass
    else:
        gt, _ = export_gt(als, set_dir)
    return compare_als_to_gt(
        als, set_dir, gt, min_corr=min_corr, max_rms_diff=max_rms_diff
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--als", type=Path, required=True)
    p.add_argument("--set-dir", type=Path, required=True)
    p.add_argument(
        "--yaml",
        type=Path,
        default=None,
        help="GT YAML to render (default: export from --als in-memory)",
    )
    p.add_argument("--min-corr", type=float, default=DEFAULT_MIN_CORR)
    p.add_argument("--max-rms-diff", type=float, default=DEFAULT_MAX_RMS_DIFF)
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="write AudioCompareResult JSON",
    )
    args = p.parse_args(argv)
    if not args.als.is_file():
        print(f"als not found: {args.als}", file=sys.stderr)
        return 2
    if not args.set_dir.is_dir():
        print(f"set-dir not found: {args.set_dir}", file=sys.stderr)
        return 2
    result = run_roundtrip(
        args.als,
        args.set_dir,
        yaml_path=args.yaml,
        min_corr=args.min_corr,
        max_rms_diff=args.max_rms_diff,
    )
    print(("PASS: " if result.ok else "FAIL: ") + result.detail)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result.as_dict(), indent=2) + "\n")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
