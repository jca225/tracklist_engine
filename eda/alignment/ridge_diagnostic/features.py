"""Four-channel mix↔ref similarity matrices for the ridge diagnostic."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from core.contracts import MANIFEST_FILENAME, load_manifest
from eda.alignment.ridge_diagnostic.cases import CaseRecord
from alignment.landmark_fp import (
    SR as FP_SR,
    fingerprint_from_audio,
)
from alignment.path_decode import _ensure_feat
from alignment.refine_ref_offsets import (
    HOP,
    SR,
    find_aligning_dir,
    ref_audio_for,
)
from alignment.trajectory.features import pool_bins

Channel = Literal["hubert", "chroma", "fp_hit", "instr_stem"]

_CACHE = Path(__file__).resolve().parent / "out" / "panel_cache"
_HUBERT_LAYER = 9
_FPS = SR / HOP


@dataclass(frozen=True)
class SimMatrix:
    channel: Channel
    M: object  # np.ndarray (Tm, Tr), float32 in [-1, 1] or [0, 1]
    mix_bin_s: float
    ref_bin_s: float
    mix_audio: str
    ref_audio: str
    notes: str


def cosine_sim_matrix(mix_bins: np.ndarray, ref_bins: np.ndarray) -> np.ndarray:
    """(Tm,D)+(Tr,D) L2-normalized → (Tm,Tr) cosine."""
    return (mix_bins @ ref_bins.T).astype(np.float32)


def _align_bin_grid(m: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    """Crop or zero-pad a similarity matrix to ``target`` (Tm, Tr).

    Bin-grid contract: hubert/chroma/instr_stem use ``pool_bins`` (floor over
    feature frames). fp_hit counts bins from audio duration with the same floor
    rule, then this helper forces all channels to share one (Tm, Tr) grid.
    """
    tm, tr = target
    out = np.zeros((tm, tr), dtype=np.float32)
    cm, cr = m.shape
    out[: min(tm, cm), : min(tr, cr)] = m[: min(tm, cm), : min(tr, cr)]
    return out


def _audio_duration_s(path: Path) -> float:
    import soundfile as sf

    try:
        return float(sf.info(str(path)).duration)
    except (RuntimeError, OSError):
        import librosa

        return float(librosa.get_duration(path=str(path)))


def _clamp_crop_bounds(lo: float, hi: float, duration_s: float) -> tuple[float, float]:
    duration_s = max(duration_s, 0.0)
    lo = max(0.0, min(lo, duration_s))
    hi = max(lo, min(hi, duration_s))
    return lo, hi


def fp_hit_matrix(
    mix_y: np.ndarray,
    ref_y: np.ndarray,
    sr: int,
    *,
    bin_s: float,
) -> np.ndarray:
    """Sparse landmark co-occurrence density, normalized to [0, 1]."""
    mix_fp = fingerprint_from_audio(mix_y, sr=sr)
    ref_fp = fingerprint_from_audio(ref_y, sr=sr)
    mix_dur = float(len(mix_y) / sr)
    ref_dur = float(len(ref_y) / sr)
    # Floor matches pool_bins row count from duration; compute_panel may still
    # crop/pad via _align_bin_grid when feature-frame counts differ slightly.
    tm = max(1, int(mix_dur / bin_s))
    tr = max(1, int(ref_dur / bin_s))
    m = np.zeros((tm, tr), dtype=np.float32)
    for key, mix_frames in mix_fp.hashes.items():
        ref_frames = ref_fp.hashes.get(key)
        if not ref_frames:
            continue
        for mt in mix_frames:
            mi = min(int((mt / mix_fp.fps) / bin_s), tm - 1)
            for rt in ref_frames:
                ri = min(int((rt / ref_fp.fps) / bin_s), tr - 1)
                m[mi, ri] += 1.0
    peak = float(m.max())
    if peak > 0:
        m /= peak
    return m


def _mix_full_path(set_dir: Path) -> Path:
    hits = [p for p in sorted(set_dir.glob("mix.*")) if p.is_file()]
    if not hits:
        raise FileNotFoundError(f"no mix.* under {set_dir}")
    return hits[0]


def _ref_crop_bounds(case: CaseRecord, pad_s: float) -> tuple[float, float]:
    ref_starts = [seg[1] for seg in case.ref_segments]
    ref_ends = [seg[2] for seg in case.ref_segments]
    lo = max(0.0, min(ref_starts) - pad_s)
    hi = max(ref_ends) + pad_s
    if case.gt_ref_end_s is not None:
        hi = max(hi, case.gt_ref_end_s + pad_s)
    return lo, hi


def _load_crop(path: Path, lo_s: float, hi_s: float, *, sr: int) -> np.ndarray:
    import librosa

    y, _ = librosa.load(
        str(path), sr=sr, mono=True, offset=lo_s, duration=max(hi_s - lo_s, 0.0)
    )
    return y.astype(np.float32)


def _tempo_stretch(y: np.ndarray, tempo_ratio: float) -> np.ndarray:
    if abs(tempo_ratio - 1.0) <= 1e-3:
        return y
    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return librosa.effects.time_stretch(y, rate=tempo_ratio).astype(np.float32)


def _write_crop(path: Path, y: np.ndarray, sr: int) -> Path:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, format="FLAC", subtype="PCM_16")
    return path


def _crop_cache_path(
    case_id: str,
    role: str,
    lo_s: float,
    hi_s: float,
    *,
    tempo_ratio: float,
) -> Path:
    tag = f"{case_id}__{role}__{lo_s:.3f}_{hi_s:.3f}__tr{tempo_ratio:.4f}.flac"
    return _CACHE / tag


def _prepare_crop_audio(
    case: CaseRecord,
    audio_path: Path,
    lo_s: float,
    hi_s: float,
    *,
    role: str,
    tempo_ratio: float = 1.0,
    sr: int = SR,
) -> tuple[Path, str]:
    out = _crop_cache_path(case.case_id, role, lo_s, hi_s, tempo_ratio=tempo_ratio)
    note = ""
    if out.is_file():
        return out, note
    y = _load_crop(audio_path, lo_s, hi_s, sr=sr)
    if tempo_ratio != 1.0:
        y = _tempo_stretch(y, tempo_ratio)
        note = f"tempo-stretched ref by {tempo_ratio:.4f}"
    _write_crop(out, y, sr)
    return out, note


def _pooled_feature_matrix(
    mix_path: Path,
    ref_path: Path,
    *,
    feature: str,
    bin_s: float,
    layer: int = _HUBERT_LAYER,
) -> np.ndarray:
    mix_feat = np.load(_ensure_feat(mix_path, mix_path, feature, layer), mmap_mode="r")
    ref_feat = np.load(_ensure_feat(ref_path, ref_path, feature, layer), mmap_mode="r")
    mix_bins = pool_bins(mix_feat, _FPS, bin_s)
    ref_bins = pool_bins(ref_feat, _FPS, bin_s)
    if mix_bins.size == 0 or ref_bins.size == 0:
        return np.zeros(
            (max(mix_bins.shape[0], 0), max(ref_bins.shape[0], 0)), dtype=np.float32
        )
    return cosine_sim_matrix(mix_bins, ref_bins)


def _resolve_ref_path(
    set_dir: Path,
    manifest_tracks: dict,
    case: CaseRecord,
    *,
    stem: str | None = None,
) -> Path | None:
    track = manifest_tracks.get(case.recording_id)
    if track is None:
        return None
    span = {
        "claimed_stem": stem or case.claimed_stem,
        "slot_label": case.slot,
    }
    track_dict = {
        "local_path": track.local_path,
        "stems": dict(track.stems),
    }
    hit = ref_audio_for(span, track_dict, set_dir)
    return hit


def compute_panel(
    case: CaseRecord,
    *,
    bin_s: float = 0.5,
    pad_s: float = 15.0,
) -> dict[Channel, SimMatrix]:
    """Build hubert/chroma/fp_hit/instr_stem similarity matrices for one hard case."""
    set_dir = find_aligning_dir(case.set_id)
    manifest = load_manifest(set_dir / MANIFEST_FILENAME)
    by_tid = manifest.by_track_id()

    mix_full = _mix_full_path(set_dir)
    ref_path = _resolve_ref_path(set_dir, by_tid, case)
    if ref_path is None:
        raise FileNotFoundError(
            f"no ref audio for recording_id={case.recording_id} slot={case.slot}"
        )

    mix_lo = max(0.0, case.gt_audible_onset_s - pad_s)
    mix_hi = case.gt_set_end_s + pad_s
    mix_lo, mix_hi = _clamp_crop_bounds(mix_lo, mix_hi, _audio_duration_s(mix_full))
    ref_lo, ref_hi = _ref_crop_bounds(case, pad_s)
    ref_lo, ref_hi = _clamp_crop_bounds(ref_lo, ref_hi, _audio_duration_s(ref_path))

    mix_crop, _ = _prepare_crop_audio(
        case, mix_full, mix_lo, mix_hi, role="mix", tempo_ratio=1.0
    )
    ref_crop, stretch_note = _prepare_crop_audio(
        case,
        ref_path,
        ref_lo,
        ref_hi,
        role="ref",
        tempo_ratio=case.tempo_ratio,
    )

    out: dict[Channel, SimMatrix] = {}

    for channel, feature in (("hubert", "hubert"), ("chroma", "chroma")):
        m = _pooled_feature_matrix(mix_crop, ref_crop, feature=feature, bin_s=bin_s)
        out[channel] = SimMatrix(
            channel=channel,
            M=m,
            mix_bin_s=bin_s,
            ref_bin_s=bin_s,
            mix_audio=str(mix_crop),
            ref_audio=str(ref_crop),
            notes=stretch_note,
        )

    mix_y = _load_crop(mix_full, mix_lo, mix_hi, sr=FP_SR)
    ref_y_raw = _load_crop(ref_path, ref_lo, ref_hi, sr=FP_SR)
    ref_y = _tempo_stretch(ref_y_raw, case.tempo_ratio)
    hubert_shape = out["hubert"].M.shape
    fp_m = _align_bin_grid(
        fp_hit_matrix(mix_y, ref_y, FP_SR, bin_s=bin_s),
        hubert_shape,
    )
    out["fp_hit"] = SimMatrix(
        channel="fp_hit",
        M=fp_m,
        mix_bin_s=bin_s,
        ref_bin_s=bin_s,
        mix_audio=str(mix_full),
        ref_audio=str(ref_path),
        notes=stretch_note,
    )

    mix_instr = set_dir / "mix_instrumental.flac"
    ref_instr = _resolve_ref_path(set_dir, by_tid, case, stem="instrumental")
    if mix_instr.is_file() and ref_instr is not None and ref_instr.is_file():
        mi_crop, _ = _prepare_crop_audio(
            case, mix_instr, mix_lo, mix_hi, role="mix_instr", tempo_ratio=1.0
        )
        ri_crop, ri_note = _prepare_crop_audio(
            case,
            ref_instr,
            ref_lo,
            ref_hi,
            role="ref_instr",
            tempo_ratio=case.tempo_ratio,
        )
        instr_m = _pooled_feature_matrix(
            mi_crop, ri_crop, feature="hubert", bin_s=bin_s
        )
        out["instr_stem"] = SimMatrix(
            channel="instr_stem",
            M=instr_m,
            mix_bin_s=bin_s,
            ref_bin_s=bin_s,
            mix_audio=str(mi_crop),
            ref_audio=str(ri_crop),
            notes=ri_note,
        )
    else:
        missing = []
        if not mix_instr.is_file():
            missing.append("mix_instrumental.flac")
        if ref_instr is None or not ref_instr.is_file():
            missing.append("ref instrumental.flac")
        warnings.warn(
            f"instr_stem skipped for {case.case_id}: missing {', '.join(missing)}",
            stacklevel=2,
        )

    return out
