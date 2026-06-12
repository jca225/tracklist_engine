#!/usr/bin/env python3
"""Acappella placement via instrumental-BPM stretch-lock (#1) + slope-constrained
subsequence DTW (#7).

The continuity stack assumes a single (offset, stretch) LINE per span. Acappellas
are sub-track WARPED — the DJ warps the vocal phrase-by-phrase to lock it to the
host beat — so the line is wrong and acappella exact<2s stalls (~43% vs ~67%
regular). Two domain facts fix it:

  #1 the host instrumental's BPM is fixed within a span and the acappella is
     beat-synced to it ⇒ the GLOBAL stretch is known from the beat grids
     (set_measures bar dur ÷ ref bar dur). Lock the slope; don't search it.
  #7 the residual is a NON-LINEAR warp ⇒ recover it with a DTW, not a rigid line.

Method per acappella span: pre-stretch the mix-vocal chroma by the grid-locked
ratio (into the ref timebase), then run a subsequence DTW against the ref vocal
stem with a tight Sakoe-Chiba band (slope ≈ 1 after the pre-stretch, the band is
just the warp tolerance). The path's start in the ref = clip ref_start.

`--eval` A/Bs DTW vs the continuity stack on the SAME acappella spans vs GT.

    venvs/audio/bin/python -m workspaces.alignment_prototype.acappella_dtw \\
        --eval [--band-rad 0.12] [--qwin-s 24] [--workers 8]
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

from workspaces.alignment_prototype.continuity_refine import (  # noqa: E402
    _grid_stretches, _probe_offsets, _stack_offset, _windows_from,
)
from workspaces.alignment_prototype.refine_ref_offsets import (  # noqa: E402
    HOP, SR, _MIX_SOURCE, chroma, find_aligning_dir,
)

_EQ_DELTA = 2.0  # exact tolerance (s)


def _resample_cols(c: np.ndarray, stretch: float) -> np.ndarray:
    m = c.shape[1]
    mm = max(4, int(round(m * stretch)))
    idx = np.clip((np.arange(mm) / stretch).astype(int), 0, m - 1)
    return c[:, idx]


def dtw_ref_start(query: np.ndarray, ref: np.ndarray, stretch: float,
                  band_rad: float) -> tuple[float, float] | None:
    """(clip_ref_start_s, mean_path_cost). query starts at the span's mix start;
    it is pre-stretched into the ref timebase by `stretch`, then aligned as a
    subsequence of `ref` with a band-limited DTW (the residual sub-track warp)."""
    import librosa
    q = _resample_cols(query, stretch)
    if ref.shape[1] <= q.shape[1] + 2:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        D, wp = librosa.sequence.dtw(X=q, Y=ref, metric="cosine",
                                     subseq=True, band_rad=band_rad)
    j_start = int(wp[-1][1])          # ref frame aligned to query[0] = span start
    total = float(D[wp[0][0], wp[0][1]])
    return j_start * HOP / SR, total / max(1, wp.shape[0])


def _job(args: tuple) -> dict:
    idx, ref_path, span_chroma, dts, n, qwin_f, stretches, stretch_c, band_rad = args
    import librosa
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref_y, _ = librosa.load(ref_path, sr=SR, mono=True)
    ref_f = chroma(ref_y)
    sc = np.asarray(span_chroma, dtype=np.float32)

    # COARSE: continuity stack (rigid line) on probe windows finds the region
    windows = [(int(round(dt * SR / HOP)), w)
               for dt in dts
               for w in [sc[:, int(dt * SR / HOP):int(dt * SR / HOP) + n]]
               if w.shape[1] >= n // 2]
    stack_r0 = float("nan")
    if windows:
        r0, _pk, _st = _stack_offset(windows, ref_f, tuple(stretches))
        stack_r0 = r0

    # FINE: grid-locked DTW, restricted to a window around the stack's region —
    # global subseq DTW on repetitive vocals locks onto the wrong occurrence, so
    # we only let the warp path refine inside the coarse region the stack found.
    q = sc[:, :qwin_f]
    dtw_r0 = stack_r0
    if not np.isnan(stack_r0):
        pad_s = 8.0
        lo = max(0, int((stack_r0 - pad_s) * SR / HOP))
        span_ref_f = int(qwin_f * stretch_c) + int(2 * pad_s * SR / HOP)
        sub = ref_f[:, lo:lo + span_ref_f]
        d = dtw_ref_start(q, sub, stretch_c, band_rad)
        if d is not None:
            dtw_r0 = lo * HOP / SR + d[0]
    return {"idx": idx, "stack": round(stack_r0, 3), "dtw": round(dtw_r0, 3)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--gt", type=Path,
                   default=_REPO / "labeling/fixtures/bb12_ground_truth.yaml")
    p.add_argument("--window-s", type=float, default=12.0, help="stack probe window")
    p.add_argument("--qwin-s", type=float, default=24.0, help="DTW query window")
    p.add_argument("--probes", type=int, default=5)
    p.add_argument("--band-rad", type=float, default=0.12, help="DTW warp tolerance")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)
    if not args.eval:
        p.error("only --eval is wired")

    import librosa
    from core.result import Err, Ok
    from workspaces.alignment_prototype.dataset import load_set
    from workspaces.alignment_prototype.mert_store import load_bb12_mert

    match load_set(args.gt):
        case Err(m): sys.exit(f"GT load failed: {m}")
        case Ok((gt, targets)): pass
    match load_bb12_mert(gt.set_id):
        case Err(m): sys.exit(f"MERT grids failed: {m}")
        case Ok((_s, mix_series, ref_series)): pass

    import yaml as _yaml
    raw = _yaml.safe_load(args.gt.read_text())
    linear = {}
    for row in raw.get("tracks", []):
        key = (str(row.get("slot_label")), round(float(row.get("set_start_s", -1)), 2))
        ratio = float(row.get("tempo_ratio") or 1.0)
        linear[key] = (not row.get("is_loop") and not row.get("ref_segments")
                       and 0.9 <= ratio <= 1.15 and not row.get("unalignable"))

    set_dir = find_aligning_dir(gt.set_id)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    for t in manifest["tracks"]:
        if t.get("recording_id"):
            by_tid.setdefault(t["recording_id"], t)

    fname = _MIX_SOURCE["acappella"][0]
    f = set_dir / fname
    if not f.is_file():
        sys.exit(f"missing {fname}")
    print(f"chroma({fname}) …")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _sr = librosa.load(str(f), sr=SR, mono=True)
    mix_vox = chroma(y)

    n = int(args.window_s * SR / HOP)
    qwin_f = int(args.qwin_s * SR / HOP)
    jobs, meta, skipped = [], [], 0
    for i, t in enumerate(targets):
        if t.claimed_stem != "acappella":
            continue
        if not linear.get((t.slot_label, round(t.set_start_s, 2)), True):
            continue
        track = by_tid.get(t.recording_id)
        if track is None:
            skipped += 1; continue
        base = Path(track["local_path"]).stem
        ref_path = set_dir / "stems" / base / "vocals.flac"
        if not ref_path.is_file():
            sp = (track.get("stems") or {}).get("vocals")
            ref_path = Path(sp) if sp and Path(sp).is_file() else None
        if ref_path is None:
            skipped += 1; continue
        a = int(t.set_start_s * SR / HOP)
        span_f = max(n, int((t.set_end_s - t.set_start_s) * SR / HOP))
        sc = mix_vox[:, a:a + span_f]
        if sc.shape[1] < n:
            skipped += 1; continue
        span_len = sc.shape[1] * HOP / SR
        dts = _probe_offsets(span_len, args.window_s, args.probes)
        stretches = _grid_stretches(t, mix_series, ref_series)
        stretch_c = float(np.median(stretches))
        jobs.append((i, str(ref_path), sc.tolist(), dts, n, qwin_f,
                     stretches, stretch_c, args.band_rad))
        meta.append(t)

    print(f"acappella spans: {len(jobs)} (skipped {skipped}); "
          f"DTW band_rad={args.band_rad}, qwin={args.qwin_s:.0f}s …")
    res = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, r in enumerate(ex.map(_job, jobs, chunksize=1)):
            res[r["idx"]] = r
            if (k + 1) % 5 == 0:
                print(f"  {k + 1}/{len(jobs)}")

    se, de, rows = [], [], []
    for (i, *_), t in zip(jobs, meta):
        r = res[i]
        s_err = abs(r["stack"] - t.ref_start_s)
        d_err = abs(r["dtw"] - t.ref_start_s)
        se.append(s_err); de.append(d_err)
        rows.append((t.slot_label, t.ref_start_s, r["stack"], s_err, r["dtw"], d_err,
                     (t.label or "")[:30]))
    se, de = np.array(se), np.array(de)
    print(f"\n=== acappella placement vs GT (n={len(rows)}) ===")
    print(f"  continuity stack : exact<2s {100*(se<2).mean():3.0f}%   median err {np.median(se):6.1f}s")
    print(f"  grid-lock + DTW  : exact<2s {100*(de<2).mean():3.0f}%   median err {np.median(de):6.1f}s")
    fixed = [r for r in rows if r[3] >= 2 and r[5] < 2]
    broke = [r for r in rows if r[3] < 2 and r[5] >= 2]
    print(f"\nDTW fixed {len(fixed)} (stack>2s → dtw<2s), broke {len(broke)}:")
    for slot, gt_s, s, se_, d, de_, lab in sorted(rows, key=lambda r: -r[5]):
        mark = "FIX " if (s and abs(s-gt_s) >= 2 and de_ < 2) else ("BRK " if (abs(s-gt_s) < 2 and de_ >= 2) else "    ")
        print(f"  {mark}{slot:6} gt={gt_s:7.1f} stack={s:7.1f}({se_:5.1f}) dtw={d:7.1f}({de_:5.1f})  {lab}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
