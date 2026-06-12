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
    HOP, SR, STRETCHES, _MIX_SOURCE, _STEM_FILE,
    chroma, detect_offset, find_aligning_dir,
)

_EQ_DELTA = 0.04  # GT scores within this of the peak => content-identical repeat


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
    e = np.concatenate([[0.0], np.cumsum((ref_f ** 2).sum(axis=0))])
    den = np.sqrt(np.maximum(e[m:] - e[:-m], 1e-9))
    return (num / den).astype(np.float32)


TRIM = 1  # drop this many lowest-scoring probes per r0 (robust to 1 bad probe)


def stack_curves(
    curves: list[np.ndarray], shifts: list[int], trim: int = TRIM
) -> tuple[np.ndarray, int]:
    """Shift-and-add probe curves onto the shared clip-start (r0) axis.

    curves[k] is the matched-filter score of probe k over the ref; probe k
    peaks at r0 + shifts[k]. Returns (J, max_shift) where J[r0] is a *trimmed*
    mean of the per-probe scores of the line with intercept r0 (length = min
    curve len - max shift). Trimming the lowest `trim` probes per r0 stops a
    single blended/wrong probe from dragging the consensus off the true line
    (the Martin Garrix regression); kept >= 2 probes so a real line still needs
    agreement. Mean keeps J comparable to a single-probe peak.
    """
    max_shift = max(shifts) if shifts else 0
    aligned = [(c, s) for c, s in zip(curves, shifts) if c.size and c.shape[0] - s > 0]
    if not aligned:
        return np.zeros(0, np.float32), max_shift
    L = min(c.shape[0] - s for c, s in aligned)
    if L <= 0:
        return np.zeros(0, np.float32), max_shift
    m = np.stack([c[s:s + L] for c, s in aligned]).astype(np.float64)  # (k, L)
    k = m.shape[0]
    t = min(trim, max(0, k - 2)) if k >= 3 else 0
    if t > 0:
        m = np.sort(m, axis=0)[t:]  # drop t lowest scorers per r0
    return m.mean(axis=0).astype(np.float32), max_shift


def _stack_offset(
    windows: list[tuple[int, np.ndarray]], ref_f: np.ndarray,
    stretches: tuple[float, ...],
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
        j, _ = stack_curves(curves, shifts)
        if j.size == 0:
            continue
        k = int(j.argmax())
        if j[k] > best[1]:
            best = (k * HOP / SR, float(j[k]), st)
    return best


def _stack_score_at(
    windows: list[tuple[int, np.ndarray]], ref_f: np.ndarray,
    stretch: float, r0_s: float,
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
    j, _ = stack_curves(curves, shifts)
    r0 = int(round(r0_s * SR / HOP))
    return float(j[r0]) if 0 <= r0 < j.size else float("nan")


def _job(args: tuple) -> dict:
    """Worker: baseline (mid probe argmax) + stacked decode for one span."""
    idx, ref_path, win_list, mid_k, stretches, gt_ref_start = args
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

    # stacked decode
    r0, jpeak, s_stretch = _stack_offset(windows, ref_f, tuple(stretches))
    base_gt = _scores_at_stretch(mid_win, ref_f, b_stretch)
    kg = int(round((gt_ref_start + (mid_dt * HOP / SR) * b_stretch) * SR / HOP))
    base_score_gt = float(base_gt[kg]) if 0 <= kg < base_gt.size else float("nan")
    stack_score_gt = _stack_score_at(windows, ref_f, s_stretch, gt_ref_start)

    return {
        "idx": idx, "n_probes": len(windows),
        "base_r0": round(base_r0, 3), "base_peak": round(peak, 3),
        "base_score_gt": base_score_gt, "base_stretch": b_stretch,
        "stack_r0": round(r0, 3), "stack_peak": round(jpeak, 3),
        "stack_score_gt": stack_score_gt, "stack_stretch": s_stretch,
    }


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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval", action="store_true", help="score vs BB12 GT")
    p.add_argument("--gt", type=Path,
                   default=_REPO / "labeling/fixtures/bb12_ground_truth.yaml")
    p.add_argument("--window-s", type=float, default=12.0)
    p.add_argument("--probes", type=int, default=5, help="max probe windows/span")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)

    if not args.eval:
        p.error("only --eval is wired (the stack core is importable)")

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
        linear_by_key[key] = (not row.get("is_loop")
                              and not row.get("ref_segments")
                              and 0.9 <= ratio <= 1.15)

    set_dir = find_aligning_dir(gt.set_id)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    for t in manifest["tracks"]:
        if t.get("recording_id"):
            by_tid.setdefault(t["recording_id"], t)

    mix_chroma: dict[str, np.ndarray] = {}
    for stem, (fname, _) in _MIX_SOURCE.items():
        f = set_dir / fname
        if f.is_file():
            print(f"chroma({fname}) …")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y, _sr = librosa.load(str(f), sr=SR, mono=True)
            mix_chroma[stem] = chroma(y)

    n = int(args.window_s * SR / HOP)
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
        mc = mix_chroma.get(stem, mix_chroma["regular"])
        span_len = max(0.0, t.set_end_s - t.set_start_s)
        dts = _probe_offsets(span_len, args.window_s, args.probes)
        win_list = []
        for dt in dts:
            a = int((t.set_start_s + dt) * SR / HOP)
            a = min(a, max(0, mc.shape[1] - n))
            w = mc[:, a:a + n]
            if w.shape[1] < n // 2:
                continue
            win_list.append((int(round(dt * SR / HOP)), w.tolist()))
        if not win_list:
            skipped += 1
            continue
        # baseline probe = the one nearest mid-span (strongest single-probe)
        mid_k = int(np.argmin([abs(dt - span_len / 2) for dt in dts][:len(win_list)]))
        stretches = _grid_stretches(t, mix_series, ref_series)
        jobs.append((i, str(ref_path), win_list, mid_k, stretches, t.ref_start_s))
        meta.append(t)

    print(f"evaluating {len(jobs)} linear spans "
          f"(excluded: {nonlinear} loop/segment/odd-ratio, {skipped} no-audio)…")
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
        b_eq = (b_err < 2.0) or (not np.isnan(r["base_score_gt"])
                                 and r["base_score_gt"] >= r["base_peak"] - _EQ_DELTA)
        s_eq = (s_err < 2.0) or (not np.isnan(r["stack_score_gt"])
                                 and r["stack_score_gt"] >= r["stack_peak"] - _EQ_DELTA)
        rows.append((t.slot_label, t.claimed_stem, src, r["n_probes"],
                     b_err, b_eq, s_err, s_eq, t.label or ""))

    def report(name: str, sel: list) -> None:
        if not sel:
            return
        be = np.array([r[4] for r in sel]); beq = np.array([r[5] for r in sel])
        se = np.array([r[6] for r in sel]); seq = np.array([r[7] for r in sel])
        print(f"  {name:22} n={len(sel):3}  "
              f"baseline exact<2s {100*(be<2).mean():3.0f}% / equiv {100*beq.mean():3.0f}%"
              f"   ||  stack exact<2s {100*(se<2).mean():3.0f}% / equiv {100*seq.mean():3.0f}%")

    comparable = [r for r in rows if r[2] != "online_candidate"]
    print("\n=== baseline (mid probe) vs continuity stack — vs corrected GT ===")
    report("ALL comparable", comparable)
    for stem in ("regular", "acappella", "instrumental"):
        report(stem, [r for r in comparable if r[1] == stem])

    fixed = [r for r in comparable if r[4] >= 2.0 and r[6] < 2.0]
    broke = [r for r in comparable if r[4] < 2.0 and r[6] >= 2.0]
    print(f"\nstack FIXED (baseline>2s -> stack<2s): {len(fixed)}")
    for slot, stem, src, npr, be, _bq, se, _sq, label in sorted(fixed, key=lambda r: -r[4])[:12]:
        print(f"  {slot:6} {stem:11} probes={npr} base_err={be:6.1f} -> {se:4.1f}  {label[:36]}")
    print(f"stack BROKE (baseline<2s -> stack>2s): {len(broke)}")
    for slot, stem, src, npr, be, _bq, se, _sq, label in sorted(broke, key=lambda r: -r[6])[:12]:
        print(f"  {slot:6} {stem:11} probes={npr} base_err={be:4.1f} -> {se:6.1f}  {label[:36]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
