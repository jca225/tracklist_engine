#!/usr/bin/env python3
"""Evaluate ref-offset detection against BB12 ground truth.

For every GT span: take the mix window at the GT set position (stem-routed,
exactly as refine_ref_offsets does on unlabeled sets), detect the ref offset
by matched filter with grid-derived stretch, and score against the
hand-labeled ref_start. Isolates the detector from the decode — measures
"given the right place in the mix, do we find the right place in the song?"

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.eval_ref_detection \\
        [--gt labeling/fixtures/bb12_ground_truth.yaml] [--workers 8]
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

from workspaces.alignment_prototype.refine_ref_offsets import (
    HOP, SR, STRETCHES, _MIX_SOURCE, _STEM_FILE,
    chroma, detect_offset, find_aligning_dir,
)


_EQ_DELTA = 0.04  # score@GT within this of the peak => content-identical repeat


def _scores_at_stretch(win_f: np.ndarray, ref_f: np.ndarray, st: float) -> np.ndarray:
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


def _job(args: tuple) -> dict:
    idx, ref_path, win, stretches, gt_ref_start = args
    import librosa
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref_y, _ = librosa.load(ref_path, sr=SR, mono=True)
    ref_f = chroma(ref_y)
    win_f = np.asarray(win, dtype=np.float32)
    ref_start, peak, stretch = detect_offset(win_f, ref_f, tuple(stretches))
    scores = _scores_at_stretch(win_f, ref_f, stretch)
    score_gt, earliest = float("nan"), ref_start
    if scores.size:
        kg = int(round(gt_ref_start * SR / HOP))
        if 0 <= kg < scores.size:
            score_gt = float(scores[kg])
        good = np.flatnonzero(scores >= peak - _EQ_DELTA)
        if good.size:
            earliest = float(good[0] * HOP / SR)
    return {"idx": idx, "det": ref_start, "peak": peak, "stretch": stretch,
            "score_gt": score_gt, "earliest": earliest}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt", type=Path, default=_REPO / "labeling/fixtures/bb12_ground_truth.yaml")
    p.add_argument("--window-s", type=float, default=12.0)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)

    import librosa
    from core.result import Err, Ok
    from workspaces.alignment_prototype.dataset import load_set
    from workspaces.alignment_prototype.mert_store import load_bb12_mert

    match load_set(args.gt):
        case Err(msg):
            sys.exit(f"GT load failed: {msg}")
        case Ok((gt, targets)):
            pass
    print(f"set={gt.set_id} GT spans={len(targets)}")

    match load_bb12_mert(gt.set_id):
        case Err(msg):
            sys.exit(f"MERT bundle (grids) load failed: {msg}")
        case Ok((_sid, mix_series, ref_series)):
            pass

    # ref_source per span from the raw yaml (load_set doesn't carry it):
    # online_candidate spans were labeled against a DIFFERENT file (the
    # downloaded acappella), so their offsets aren't comparable with
    # detection in Demucs stems — scored separately.
    import yaml as _yaml
    raw = _yaml.safe_load(args.gt.read_text())
    src_by_key: dict[tuple, str] = {}
    linear_by_key: dict[tuple, bool] = {}
    for row in raw.get("tracks", []):
        key = (str(row.get("slot_label")), round(float(row.get("set_start_s", -1)), 2))
        src_by_key[key] = row.get("ref_source") or "reference"
        # loops / split clips break the linear set->ref map this eval
        # assumes (measured: loop rows have tempo_ratio 0.11-0.54 and
        # detection correctly finds the SAME looped phrase at every probe
        # point while linear extrapolation walks away). Only spans the
        # human placed as one straight clip are scorable here.
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

    jobs, meta, skipped, nonlinear = [], [], 0, 0
    for i, t in enumerate(targets):
        if t.slot_label == "mix":
            continue  # artifact row in the GT export
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
        # probe MID-span: at set_start the incoming track is still buried
        # under the outgoing one (transition blend) — probing there matched
        # loud later sections instead of the quiet intro and tanked the
        # first eval. Expected ref position advances by the GT tempo_ratio.
        span_len = max(0.0, t.set_end_s - t.set_start_s)
        ratio = t.tempo_ratio if (t.tempo_ratio and 0.5 < t.tempo_ratio < 2.0) else 1.0
        probe_dt = max(0.0, min(8.0, span_len - args.window_s))
        probe_t = t.set_start_s + probe_dt
        expected_ref = t.ref_start_s + probe_dt * ratio
        a = int(probe_t * SR / HOP)
        n = int(args.window_s * SR / HOP)
        a = min(a, max(0, mc.shape[1] - n))
        win = mc[:, a:a + n]
        if win.shape[1] < n // 2:
            skipped += 1
            continue
        # grid-derived stretch, octave-folded (same as --grid-stretch)
        stretches = STRETCHES
        if t.recording_id in ref_series:
            j = int(np.searchsorted(mix_series.start_s, t.set_start_s))
            lo, hi = max(0, j - 2), min(mix_series.n_measures, j + 3)
            mix_bar = float(np.median(mix_series.end_s[lo:hi] - mix_series.start_s[lo:hi]))
            rser = ref_series[t.recording_id]
            ref_bar = float(np.median(rser.end_s - rser.start_s))
            if mix_bar > 0 and ref_bar > 0:
                e = ref_bar / mix_bar
                while e > 1.45:
                    e *= 0.5
                while e < 0.7:
                    e *= 2.0
                stretches = tuple(e * f for f in (0.96, 0.98, 1.0, 1.02, 1.04))
        jobs.append((i, str(ref_path), win.tolist(), stretches, expected_ref))
        meta.append((t, expected_ref))

    print(f"evaluating {len(jobs)} linear spans "
          f"(excluded: {nonlinear} loop/segment/odd-ratio, {skipped} without audio)…")
    res: dict[int, dict] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, r in enumerate(ex.map(_job, jobs, chunksize=2)):
            res[r["idx"]] = r
            if (k + 1) % 25 == 0:
                print(f"  {k + 1}/{len(jobs)}")

    rows = []
    for (i, _path, _w, _st, _gt), (t, expected_ref) in zip(jobs, meta):
        r = res[i]
        src = src_by_key.get((t.slot_label, round(t.set_start_s, 2)), "reference")
        err = abs(r["det"] - expected_ref)
        err_early = abs(r["earliest"] - expected_ref)
        # equivalent = GT position scores within delta of the found peak
        equiv = (err < 2.0) or (not np.isnan(r["score_gt"])
                                and r["score_gt"] >= r["peak"] - _EQ_DELTA)
        st_err = (abs(r["stretch"] - t.tempo_ratio) / t.tempo_ratio
                  if t.tempo_ratio else None)
        rows.append((err, t.slot_label, t.claimed_stem, src, r["peak"],
                     expected_ref, r["det"], equiv, err_early, st_err,
                     t.label or ""))

    def report(name: str, sel: list) -> None:
        if not sel:
            return
        e = np.array([r[0] for r in sel])
        eq = np.array([r[7] for r in sel])
        ee = np.array([r[8] for r in sel])
        print(f"  {name:24} n={len(e):3}  exact<2s: {100 * (e < 2).mean():3.0f}%  "
              f"equiv(repeat-aware): {100 * eq.mean():3.0f}%  "
              f"earliest-tiebreak<2s: {100 * (ee < 2).mean():3.0f}%")

    comparable = [r for r in rows if r[3] != "online_candidate"]
    candidate = [r for r in rows if r[3] == "online_candidate"]
    print(f"\n=== detection vs GT ===")
    report("ALL comparable", comparable)
    for stem in ("regular", "acappella", "instrumental"):
        report(f"  {stem}", [r for r in comparable if r[2] == stem])
    report("online_candidate (n/c)*", candidate)
    print("  * candidate spans were labeled against a different file — only the")
    print("    equiv column is meaningful there (content match, not offset match)")
    st_errs = np.array([r[9] for r in comparable if r[9] is not None])
    if st_errs.size:
        print(f"stretch err vs GT tempo_ratio (comparable): "
              f"median={100 * np.median(st_errs):.1f}%  "
              f"p90={100 * np.percentile(st_errs, 90):.1f}%")

    bad = sorted((r for r in comparable if not r[7]), reverse=True)
    print(f"\nnon-equivalent misses ({len(bad)}):")
    print(f"{'err_s':>7} {'slot':6} {'stem':11} {'src':9} {'peak':>5} "
          f"{'gt@':>7} {'det@':>7}  label")
    for err, slot, stem, src, peak, gt_at, det, _eq, _ee, _se, label in bad[:15]:
        print(f"{err:7.1f} {slot:6} {stem:11} {src:9} {peak:5.2f} "
              f"{gt_at:7.1f} {det:7.1f}  {label[:36]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
