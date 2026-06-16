#!/usr/bin/env python3
"""Warp-consistency ref-offset decode: stack multiple probe windows per span
onto a single clip-start axis, so a *repeated* section can't win unless the
whole span structure repeats coherently.

The per-span argmax (`refine_ref_offsets`) slides ONE mix window over the ref.
When a chorus repeats in the song, that window scores equally at chorus 1 and
chorus 2 and the argmax picks one arbitrarily — the dominant error source
(BB12: 42% exact <2 s vs 67% repeat-equivalent; the 25-pt gap *is* this).

A straight clip is a line in (mix-time, ref-time): K probe windows taken at mix
offsets dt_0..dt_{K-1} into the span all sit at `r0 + dt_k * stretch` for one
shared clip-start r0. So we:

  1. matched-filter EACH probe over the whole ref -> a full score curve
     (`_scores_at_stretch`, reused from eval_ref_detection),
  2. shift curve k left by round(dt_k * stretch) frames and SUM — a 1-D
     shift-and-add (Hough over the line's intercept). The true line reinforces
     across all K probes; a spurious repeat spikes for one probe and washes out,
  3. argmax of the stacked curve = jointly-consistent r0; search the
     grid-derived stretch band and keep the best joint peak.

`--eval` scores baseline (single mid-span probe, = the strongest variant from
eval_ref_detection) vs the stack on the SAME spans against corrected BB12 GT,
with the same repeat-aware equivalence metric, so the hypothesis ("stacking
converts repeat-equivalent hits into exact hits") is testable directly.

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.continuity_refine \\
        --eval [--probes 5] [--window-s 12] [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.alignment_prototype.refine_ref_offsets import (  # noqa: E402
    HOP,
    SR,
    STRETCHES,
    _MIX_SOURCE,
    _STEM_FILE,
    chroma,
    detect_offset,
    find_aligning_dir,
)

_EQ_DELTA = 0.04  # GT scores within this of the peak => content-identical repeat

# --- HuBERT-frame feature option ------------------------------------------
# Chroma can't separate verse 1 from verse 2 (same melody) — the dominant
# "which section" failure on acappellas. HuBERT frame embeddings match on
# *what is sung* (lyrics are the most position-specific signal a vocal has),
# so they sharpen the diagonal where chroma is flat. Drop-in for chroma: same
# (D, T) matched-filter contract on the SR/HOP grid. Expensive (torch/MPS, not
# fork-safe), so features are precomputed serially in the parent and cached to
# disk; workers only np.load the cached array.
_FEAT_CACHE = _REPO / "workspaces/alignment_prototype/.feat_cache"


def _hubert_cache_path(path, layer: int) -> Path:
    import hashlib

    key = hashlib.md5(str(path).encode()).hexdigest()[:16]
    return _FEAT_CACHE / f"{key}_hubertL{layer}.npy"


def _compute_hubert(path, layer: int) -> np.ndarray:
    """(768, T) HuBERT frames on the SR/HOP grid, cached to disk by path."""
    cf = _hubert_cache_path(path, layer)
    if cf.is_file():
        return np.load(cf)
    import librosa

    from workspaces.section_hsmm.similarity_probe import _hubert

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(str(path), sr=SR, mono=True)
    f = _hubert(y, layer)
    _FEAT_CACHE.mkdir(parents=True, exist_ok=True)
    np.save(cf, f)
    return f


def _scores_at_stretch(win_f: np.ndarray, ref_f: np.ndarray, st: float) -> np.ndarray:
    """Full normalized matched-filter curve of one window over the whole ref."""
    from scipy.signal import fftconvolve

    n = win_f.shape[1]
    m = int(round(n * st))
    idx = np.clip((np.arange(m) / st).astype(int), 0, n - 1)
    w = win_f[:, idx]
    w = w / (np.linalg.norm(w) + 1e-9)
    if ref_f.shape[1] <= m:
        return np.zeros(0, np.float32)
    num = fftconvolve(ref_f, w[:, ::-1], mode="valid", axes=1).sum(axis=0)
    e = np.concatenate([[0.0], np.cumsum((ref_f**2).sum(axis=0))])
    den = np.sqrt(np.maximum(e[m:] - e[:-m], 1e-9))
    return (num / den).astype(np.float32)


TRIM = 1  # drop this many lowest-scoring probes per r0 (robust to 1 bad probe)


def stack_curves(
    curves: list[np.ndarray],
    shifts: list[int],
    trim: int = TRIM,
    band_frames: int = 0,
) -> tuple[np.ndarray, int]:
    """Shift-and-add probe curves onto the shared clip-start (r0) axis.

    curves[k] is the matched-filter score of probe k over the ref; probe k
    peaks at r0 + shifts[k]. Returns (J, max_shift) where J[r0] is a *trimmed*
    mean of the per-probe scores of the line with intercept r0 (length = min
    curve len - max shift). Trimming the lowest `trim` probes per r0 stops a
    single blended/wrong probe from dragging the consensus off the true line
    (the Martin Garrix regression); kept >= 2 probes so a real line still needs
    agreement. Mean keeps J comparable to a single-probe peak.

    band_frames > 0 relaxes the constant-slope assumption: each probe takes its
    best score within +/-band of the predicted position (a 1-D max-filter before
    stacking) so a sub-track-WARPED clip — acappellas are warped phrase-by-phrase
    to lock to the beat — still reinforces. Use for acappella spans only; a band
    on rigid (regular) spans only buys spurious agreement.
    """
    max_shift = max(shifts) if shifts else 0
    aligned = [(c, s) for c, s in zip(curves, shifts) if c.size and c.shape[0] - s > 0]
    if not aligned:
        return np.zeros(0, np.float32), max_shift
    if band_frames > 0:
        from scipy.ndimage import maximum_filter1d

        aligned = [(maximum_filter1d(c, 2 * band_frames + 1), s) for c, s in aligned]
    L = min(c.shape[0] - s for c, s in aligned)
    if L <= 0:
        return np.zeros(0, np.float32), max_shift
    m = np.stack([c[s : s + L] for c, s in aligned]).astype(np.float64)  # (k, L)
    k = m.shape[0]
    t = min(trim, max(0, k - 2)) if k >= 3 else 0
    if t > 0:
        m = np.sort(m, axis=0)[t:]  # drop t lowest scorers per r0
    return m.mean(axis=0).astype(np.float32), max_shift


def _stack_offset(
    windows: list[tuple[int, np.ndarray]],
    ref_f: np.ndarray,
    stretches: tuple[float, ...],
    band_frames: int = 0,
) -> tuple[float, float, float]:
    """(clip_ref_start_s, joint_peak, stretch) over the stretch band."""
    best = (0.0, -2.0, 1.0)
    for st in stretches:
        curves, shifts = [], []
        for dt_frames, win in windows:
            c = _scores_at_stretch(win, ref_f, st)
            if c.size == 0:
                continue
            curves.append(c)
            shifts.append(int(round(dt_frames * st)))
        if not curves:
            continue
        j, _ = stack_curves(curves, shifts, band_frames=band_frames)
        if j.size == 0:
            continue
        k = int(j.argmax())
        if j[k] > best[1]:
            best = (k * HOP / SR, float(j[k]), st)
    return best


def _stack_score_at(
    windows: list[tuple[int, np.ndarray]],
    ref_f: np.ndarray,
    stretch: float,
    r0_s: float,
    band_frames: int = 0,
) -> float:
    """Joint (mean) score of the line with intercept r0_s at a fixed stretch."""
    curves, shifts = [], []
    for dt_frames, win in windows:
        c = _scores_at_stretch(win, ref_f, stretch)
        if c.size == 0:
            continue
        curves.append(c)
        shifts.append(int(round(dt_frames * stretch)))
    if not curves:
        return float("nan")
    j, _ = stack_curves(curves, shifts, band_frames=band_frames)
    r0 = int(round(r0_s * SR / HOP))
    return float(j[r0]) if 0 <= r0 < j.size else float("nan")


def _job(args: tuple) -> dict:
    """Worker: baseline (mid probe argmax) + stacked decode for one span."""
    idx, ref_path, win_list, mid_k, stretches, gt_ref_start, band_frames, feature = args
    if feature == "hubert":
        ref_f = np.load(ref_path)  # ref_path is the cached feature .npy
    else:
        import librosa

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref_y, _ = librosa.load(ref_path, sr=SR, mono=True)
        ref_f = chroma(ref_y)
    windows = [(int(dt), np.asarray(w, dtype=np.float32)) for dt, w in win_list]

    # baseline: single mid-span window, exactly as the production refiner /
    # eval_ref_detection does. Express its hit as a CLIP start (subtract the
    # probe's own mix offset) so it shares the stack's r0 axis.
    mid_dt, mid_win = windows[mid_k]
    win_ref_start, peak, b_stretch = detect_offset(mid_win, ref_f, tuple(stretches))
    base_r0 = win_ref_start - (mid_dt * HOP / SR) * b_stretch

    # stacked decode (band_frames > 0 only for warped acappella spans)
    r0, jpeak, s_stretch = _stack_offset(windows, ref_f, tuple(stretches), band_frames)
    base_gt = _scores_at_stretch(mid_win, ref_f, b_stretch)
    kg = int(round((gt_ref_start + (mid_dt * HOP / SR) * b_stretch) * SR / HOP))
    base_score_gt = float(base_gt[kg]) if 0 <= kg < base_gt.size else float("nan")
    stack_score_gt = _stack_score_at(
        windows, ref_f, s_stretch, gt_ref_start, band_frames
    )

    return {
        "idx": idx,
        "n_probes": len(windows),
        "base_r0": round(base_r0, 3),
        "base_peak": round(peak, 3),
        "base_score_gt": base_score_gt,
        "base_stretch": b_stretch,
        "stack_r0": round(r0, 3),
        "stack_peak": round(jpeak, 3),
        "stack_score_gt": stack_score_gt,
        "stack_stretch": s_stretch,
    }


def _stack_best(windows, ref_f, stretches, band_frames):
    """(err-free) best stack over stretches:
    (r0_s, peak, prominence, stretch, zscore).

    prominence = peak - median(curve) (raw, biased across channels). zscore =
    (peak - mean(curve)) / std(curve) — the peak in units of the channel's OWN
    score noise, a debiased confidence: a sparse vocal stem and a dense
    instrumental stem become comparable, fixing the prominence bias that made
    raw cross-channel arbitration unreliable (8/27)."""
    best = None
    for st in stretches:
        curves, shifts = [], []
        for dt_frames, win in windows:
            c = _scores_at_stretch(win, ref_f, st)
            if c.size:
                curves.append(c)
                shifts.append(int(round(dt_frames * st)))
        if not curves:
            continue
        j, _ = stack_curves(curves, shifts, band_frames=band_frames)
        if j.size == 0:
            continue
        k = int(j.argmax())
        pk = float(j[k])
        if best is None or pk > best[1]:
            z = (pk - float(j.mean())) / (float(j.std()) + 1e-9)
            best = (k * HOP / SR, pk, pk - float(np.median(j)), st, z)
    return best


def _xchan_job(args: tuple) -> dict:
    """Stack every available channel for one span; return per-channel results."""
    idx, gt_ref_start, channels = args
    import librosa

    out = []
    for name, ref_path, win_list, stretches, band in channels:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref_y, _ = librosa.load(ref_path, sr=SR, mono=True)
        ref_f = chroma(ref_y)
        windows = [(int(dt), np.asarray(w, dtype=np.float32)) for dt, w in win_list]
        b = _stack_best(windows, ref_f, tuple(stretches), band)
        if b is None:
            out.append(
                {
                    "ch": name,
                    "err": float("nan"),
                    "peak": float("nan"),
                    "prom": float("nan"),
                    "z": float("nan"),
                    "r0": float("nan"),
                }
            )
        else:
            r0, pk, prom, _st, z = b
            out.append(
                {
                    "ch": name,
                    "err": abs(r0 - gt_ref_start),
                    "peak": round(pk, 3),
                    "prom": round(prom, 3),
                    "z": round(z, 3),
                    "r0": round(r0, 3),
                }
            )
    return {"idx": idx, "channels": out}


def _probe_offsets(span_len: float, window_s: float, k_max: int) -> list[float]:
    """Up to k_max mix offsets (s into the span) for probe windows.

    Probes are inset by a guard band from both span edges: set_start is a
    transition blend (incoming track still buried under the outgoing) and the
    span tail is the next transition — both poison a probe. Spacing allows up
    to 50% window overlap so the guard doesn't starve the probe count.
    """
    usable = span_len - window_s
    if usable <= 0.5:
        return [0.0]
    pad = min(4.0, usable * 0.2)
    lo, hi = pad, usable - pad
    if hi <= lo:
        return [usable / 2.0]
    n_fit = int((hi - lo) // (window_s * 0.5)) + 1
    k = max(1, min(k_max, n_fit))
    if k == 1:
        return [(lo + hi) / 2.0]
    return list(np.linspace(lo, hi, k))


def _grid_stretches(t, mix_series, ref_series) -> tuple[float, ...]:
    """Grid-derived octave-folded stretch band (same recipe as the refiner)."""
    if t.recording_id not in ref_series:
        return STRETCHES
    j = int(np.searchsorted(mix_series.start_s, t.set_start_s))
    lo, hi = max(0, j - 2), min(mix_series.n_measures, j + 3)
    mix_bar = float(np.median(mix_series.end_s[lo:hi] - mix_series.start_s[lo:hi]))
    rser = ref_series[t.recording_id]
    ref_bar = float(np.median(rser.end_s - rser.start_s))
    if mix_bar <= 0 or ref_bar <= 0:
        return STRETCHES
    e = ref_bar / mix_bar
    while e > 1.45:
        e *= 0.5
    while e < 0.7:
        e *= 2.0
    return tuple(e * f for f in (0.96, 0.98, 1.0, 1.02, 1.04))


def _windows_from(mc: np.ndarray, set_start_s: float, dts: list[float], n: int):
    """Probe windows sliced from a mix-channel chroma at the given offsets."""
    wl = []
    for dt in dts:
        a = int((set_start_s + dt) * SR / HOP)
        a = min(a, max(0, mc.shape[1] - n))
        w = mc[:, a : a + n]
        if w.shape[1] >= n // 2:
            wl.append((int(round(dt * SR / HOP)), w.tolist()))
    return wl


# mix-channel chroma key + ref-side audio resolver per channel
_CHANNELS = (
    ("regular", "regular", None),  # full mix vs full ref track
    ("acappella", "acappella", "vocals"),  # mix vocals vs ref vocals stem
    ("instrumental", "instrumental", "instrumental"),
)


def _run_cross_channel(
    args,
    targets,
    src_by_key,
    linear_by_key,
    by_tid,
    mix_chroma,
    mix_series,
    ref_series,
    n,
) -> int:
    """Heuristic #2/#6: per span, stack every channel whose mix + ref audio both
    exist; flag where the best-localizing (or most-prominent) channel disagrees
    with claimed_stem. The audio, not the label, names the stem."""
    acap_band = int(round(args.acap_band_s * SR / HOP))
    jobs, meta, skipped = [], [], 0
    for i, t in enumerate(targets):
        if t.slot_label == "mix":
            continue
        if not linear_by_key.get((t.slot_label, round(t.set_start_s, 2)), True):
            continue
        track = by_tid.get(t.recording_id)
        if track is None:
            skipped += 1
            continue
        span_len = max(0.0, t.set_end_s - t.set_start_s)
        dts = _probe_offsets(span_len, args.window_s, args.probes)
        stretches = _grid_stretches(t, mix_series, ref_series)
        chans = []
        for name, mix_key, ref_stem in _CHANNELS:
            mc = mix_chroma.get(mix_key)
            if mc is None:
                continue
            if ref_stem is None:
                ref_path = track.get("local_path")
            else:
                ref_path = (track.get("stems") or {}).get(ref_stem)
            if not ref_path or not Path(ref_path).is_file():
                continue
            win_list = _windows_from(mc, t.set_start_s, dts, n)
            if not win_list:
                continue
            band = acap_band if name == "acappella" else 0
            chans.append((name, str(ref_path), win_list, stretches, band))
        if len(chans) < 2:  # nothing to arbitrate
            skipped += 1
            continue
        jobs.append((i, t.ref_start_s, chans))
        meta.append(t)

    print(f"cross-channel on {len(jobs)} spans (>=2 channels each; {skipped} skipped)…")
    res = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, r in enumerate(ex.map(_xchan_job, jobs, chunksize=2)):
            res[r["idx"]] = r
            if (k + 1) % 20 == 0:
                print(f"  {k + 1}/{len(jobs)}")

    rows = []
    for (i, _gt, _ch), t in zip(jobs, meta):
        chans = {c["ch"]: c for c in res[i]["channels"]}
        claimed = t.claimed_stem or "regular"
        if claimed not in chans:
            continue
        # best channel by GT error (oracle) and by prominence (label-free)
        valid = {k: c for k, c in chans.items() if not np.isnan(c["err"])}
        if not valid:
            continue
        best_err = min(valid, key=lambda k: valid[k]["err"])
        best_prom = max(valid, key=lambda k: valid[k]["prom"])
        rows.append((t, claimed, chans, best_err, best_prom))

    print("\n=== cross-channel mislabel scan (vs corrected GT) ===")
    agree_err = sum(1 for r in rows if r[3] == r[1])
    agree_prom = sum(1 for r in rows if r[4] == r[1])
    prom_matches_err = sum(1 for r in rows if r[3] == r[4])
    print(
        f"n={len(rows)}   claimed==best-by-error: {agree_err}/{len(rows)}"
        f"   claimed==best-by-prominence: {agree_prom}/{len(rows)}"
        f"   prominence agrees with oracle: {prom_matches_err}/{len(rows)}"
    )

    flagged = [r for r in rows if r[4] != r[1]]
    print(f"\nFLAGGED (best-by-prominence != claimed) — {len(flagged)}:")
    print(
        f"{'slot':6} {'claimed':11} {'best_prom':11} {'best_err':11} "
        f"{'claim_err':>9} {'alt_err':>8}  label"
    )
    for t, claimed, chans, best_err, best_prom in flagged:
        ce = chans[claimed]["err"]
        ae = chans[best_prom]["err"]
        print(
            f"{t.slot_label:6} {claimed:11} {best_prom:11} {best_err:11} "
            f"{ce:9.1f} {ae:8.1f}  {(t.label or '')[:34]}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval", action="store_true", help="score vs BB12 GT")
    p.add_argument(
        "--gt", type=Path, default=_REPO / "labeling/fixtures/bb12_ground_truth.yaml"
    )
    p.add_argument("--window-s", type=float, default=12.0)
    p.add_argument("--probes", type=int, default=5, help="max probe windows/span")
    p.add_argument(
        "--acap-band-s",
        type=float,
        default=0.0,
        help="warp tolerance (s) applied to ACAPPELLA spans only — each "
        "probe may slip +/- this off the line (heuristic #7: acaps "
        "are sub-track warped). 0 = rigid stack everywhere.",
    )
    p.add_argument(
        "--cross-channel",
        action="store_true",
        help="heuristic #2/#6: stack every available channel (full / "
        "vocal / instrumental) per span; flag where the audio's best "
        "channel disagrees with claimed_stem (mislabel detector).",
    )
    p.add_argument(
        "--feature",
        choices=["chroma", "hubert"],
        default="chroma",
        help="matched-filter feature. hubert = phonetic frames (lyrics are "
        "position-specific) — the fix for acappella 'which section'.",
    )
    p.add_argument(
        "--hubert-layer", type=int, default=9, help="HuBERT layer (mid = phonetic)"
    )
    p.add_argument(
        "--stems",
        default="regular,acappella,instrumental",
        help="comma list of claimed_stem values to evaluate (limits the "
        "expensive HuBERT precompute to the channel under test)",
    )
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)

    if not args.eval:
        p.error("only --eval is wired (the stack core is importable)")
    if args.cross_channel and args.feature == "hubert":
        p.error("--feature hubert is not wired into --cross-channel (chroma only)")
    want_stems = {s.strip() for s in args.stems.split(",") if s.strip()}

    import librosa
    import yaml as _yaml
    from core.result import Err, Ok
    from workspaces.alignment_prototype.dataset import load_set
    from workspaces.alignment_prototype.mert_store import load_bb12_mert

    match load_set(args.gt):
        case Err(msg):
            sys.exit(f"GT load failed: {msg}")
        case Ok((gt, targets)):
            pass
    match load_bb12_mert(gt.set_id):
        case Err(msg):
            sys.exit(f"MERT bundle (grids) load failed: {msg}")
        case Ok((_sid, mix_series, ref_series)):
            pass
    print(f"set={gt.set_id} GT spans={len(targets)} probes<= {args.probes}")

    raw = _yaml.safe_load(args.gt.read_text())
    src_by_key, linear_by_key = {}, {}
    for row in raw.get("tracks", []):
        key = (str(row.get("slot_label")), round(float(row.get("set_start_s", -1)), 2))
        src_by_key[key] = row.get("ref_source") or "reference"
        ratio = float(row.get("tempo_ratio") or 1.0)
        linear_by_key[key] = (
            not row.get("is_loop")
            and not row.get("ref_segments")
            and 0.9 <= ratio <= 1.15
        )

    set_dir = find_aligning_dir(gt.set_id)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    for t in manifest["tracks"]:
        if t.get("recording_id"):
            by_tid.setdefault(t["recording_id"], t)

    mix_chroma: dict[str, np.ndarray] = {}
    for stem, (fname, _) in _MIX_SOURCE.items():
        if args.feature == "hubert" and stem not in want_stems:
            continue  # don't compute hour-long HuBERT for an unused channel
        f = set_dir / fname
        if not f.is_file():
            continue
        if args.feature == "hubert":
            print(f"hubert L{args.hubert_layer}({fname}) …")
            mix_chroma[stem] = _compute_hubert(f, args.hubert_layer)
        else:
            print(f"chroma({fname}) …")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y, _sr = librosa.load(str(f), sr=SR, mono=True)
            mix_chroma[stem] = chroma(y)

    n = int(args.window_s * SR / HOP)

    if args.cross_channel:
        return _run_cross_channel(
            args,
            targets,
            src_by_key,
            linear_by_key,
            by_tid,
            mix_chroma,
            mix_series,
            ref_series,
            n,
        )

    jobs, meta, skipped, nonlinear = [], [], 0, 0
    for i, t in enumerate(targets):
        if t.slot_label == "mix":
            continue
        if not linear_by_key.get((t.slot_label, round(t.set_start_s, 2)), True):
            nonlinear += 1
            continue
        track = by_tid.get(t.recording_id)
        if track is None:
            skipped += 1
            continue
        stem = t.claimed_stem or "regular"
        if stem not in want_stems:
            continue
        ref_path = None
        stem_key = _STEM_FILE.get(stem)
        if stem_key:
            sp = (track.get("stems") or {}).get(stem_key)
            if sp and Path(sp).is_file():
                ref_path = sp
        if ref_path is None:
            ref_path = track["local_path"]
        if not Path(ref_path).is_file():
            skipped += 1
            continue
        mc = mix_chroma.get(stem)
        if mc is None:
            mc = mix_chroma.get("regular")
        if mc is None:
            skipped += 1
            continue
        span_len = max(0.0, t.set_end_s - t.set_start_s)
        dts = _probe_offsets(span_len, args.window_s, args.probes)
        win_list = []
        for dt in dts:
            a = int((t.set_start_s + dt) * SR / HOP)
            a = min(a, max(0, mc.shape[1] - n))
            w = mc[:, a : a + n]
            if w.shape[1] < n // 2:
                continue
            win_list.append((int(round(dt * SR / HOP)), w.tolist()))
        if not win_list:
            skipped += 1
            continue
        # baseline probe = the one nearest mid-span (strongest single-probe)
        mid_k = int(np.argmin([abs(dt - span_len / 2) for dt in dts][: len(win_list)]))
        stretches = _grid_stretches(t, mix_series, ref_series)
        band_frames = (
            int(round(args.acap_band_s * SR / HOP)) if stem == "acappella" else 0
        )
        if args.feature == "hubert":
            _compute_hubert(ref_path, args.hubert_layer)  # cache serially (MPS)
            ref_arg = str(_hubert_cache_path(ref_path, args.hubert_layer))
        else:
            ref_arg = str(ref_path)
        jobs.append(
            (
                i,
                ref_arg,
                win_list,
                mid_k,
                stretches,
                t.ref_start_s,
                band_frames,
                args.feature,
            )
        )
        meta.append(t)

    print(
        f"evaluating {len(jobs)} linear spans "
        f"(excluded: {nonlinear} loop/segment/odd-ratio, {skipped} no-audio)…"
    )
    res: dict[int, dict] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, r in enumerate(ex.map(_job, jobs, chunksize=2)):
            res[r["idx"]] = r
            if (k + 1) % 25 == 0:
                print(f"  {k + 1}/{len(jobs)}")

    rows = []
    for (i, *_), t in zip(jobs, meta):
        r = res[i]
        src = src_by_key.get((t.slot_label, round(t.set_start_s, 2)), "reference")
        b_err = abs(r["base_r0"] - t.ref_start_s)
        s_err = abs(r["stack_r0"] - t.ref_start_s)
        b_eq = (b_err < 2.0) or (
            not np.isnan(r["base_score_gt"])
            and r["base_score_gt"] >= r["base_peak"] - _EQ_DELTA
        )
        s_eq = (s_err < 2.0) or (
            not np.isnan(r["stack_score_gt"])
            and r["stack_score_gt"] >= r["stack_peak"] - _EQ_DELTA
        )
        rows.append(
            (
                t.slot_label,
                t.claimed_stem,
                src,
                r["n_probes"],
                b_err,
                b_eq,
                s_err,
                s_eq,
                t.label or "",
            )
        )

    def report(name: str, sel: list) -> None:
        if not sel:
            return
        be = np.array([r[4] for r in sel])
        beq = np.array([r[5] for r in sel])
        se = np.array([r[6] for r in sel])
        seq = np.array([r[7] for r in sel])
        print(
            f"  {name:22} n={len(sel):3}  "
            f"baseline exact<2s {100 * (be < 2).mean():3.0f}% / equiv {100 * beq.mean():3.0f}%"
            f"   ||  stack exact<2s {100 * (se < 2).mean():3.0f}% / equiv {100 * seq.mean():3.0f}%"
        )

    comparable = [r for r in rows if r[2] != "online_candidate"]
    print("\n=== baseline (mid probe) vs continuity stack — vs corrected GT ===")
    report("ALL comparable", comparable)
    for stem in ("regular", "acappella", "instrumental"):
        report(stem, [r for r in comparable if r[1] == stem])

    fixed = [r for r in comparable if r[4] >= 2.0 and r[6] < 2.0]
    broke = [r for r in comparable if r[4] < 2.0 and r[6] >= 2.0]
    print(f"\nstack FIXED (baseline>2s -> stack<2s): {len(fixed)}")
    for slot, stem, src, npr, be, _bq, se, _sq, label in sorted(
        fixed, key=lambda r: -r[4]
    )[:12]:
        print(
            f"  {slot:6} {stem:11} probes={npr} base_err={be:6.1f} -> {se:4.1f}  {label[:36]}"
        )
    print(f"stack BROKE (baseline<2s -> stack>2s): {len(broke)}")
    for slot, stem, src, npr, be, _bq, se, _sq, label in sorted(
        broke, key=lambda r: -r[6]
    )[:12]:
        print(
            f"  {slot:6} {stem:11} probes={npr} base_err={be:4.1f} -> {se:6.1f}  {label[:36]}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
