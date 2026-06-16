#!/usr/bin/env python3
"""v0.1 — the load-bearing experiment: does CHROMA deliver the leeway rescue?

v0 showed MERT is too self-similar (~0.92 floor) to define within-track section
equivalence. This re-runs the identical oracle-span scorecard with chroma —
the prior work's placement feature — for BOTH localization (detect_offset) and
the equivalence class (chroma self-similarity). If equivalence-aware placement
jumps toward the ~67% the prior work cites for regular stems, the user's
"allow leeway on identical parts" thesis is confirmed on a discriminative
feature, and the section HSMM has a real emission signal to stand on.

Reuses workspaces/alignment_prototype chroma + detect_offset read-only. Audio
comes from ~/aligning/<set>/ (mix + per-track stems). Chroma is cached under
workspaces/section_hsmm/.cache/ so threshold sweeps are cheap.

Usage:
    venvs/audio/bin/python -m workspaces.section_hsmm.v0_1_chroma_scorecard \
        --set-id 1fsnxchk [--thresh 0.80] [--tol-s 2.0]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.alignment_prototype.refine_ref_offsets import (  # noqa: E402
    HOP, SR, _MIX_SOURCE, chroma, detect_offset, find_aligning_dir, ref_audio_for,
)
from workspaces.section_hsmm.v0_scorecard import _straight_rows  # noqa: E402

_CACHE = Path(__file__).resolve().parent / ".cache"
FPS = SR / HOP  # chroma frames per second


def _load_chroma(audio_path: Path, cache_key: str) -> np.ndarray:
    _CACHE.mkdir(parents=True, exist_ok=True)
    cf = _CACHE / f"{cache_key}.npy"
    if cf.is_file():
        return np.load(cf)
    import librosa
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    c = chroma(y)
    np.save(cf, c)
    return c


def _l2cols(c: np.ndarray) -> np.ndarray:
    """L2-normalize each chroma column (librosa.util.normalize defaults to
    max-norm, NOT L2 — so a raw column dot is not a cosine)."""
    return c / np.clip(np.linalg.norm(c, axis=0, keepdims=True), 1e-8, None)


def _win_cos(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-frame cosine between two (12, w) chroma windows."""
    w = min(a.shape[1], b.shape[1])
    if w == 0:
        return 0.0
    a, b = _l2cols(a[:, :w]), _l2cols(b[:, :w])
    return float((a * b).sum(axis=0).mean())


def _floor(ref_c: np.ndarray, width: int) -> float:
    """Median off-diagonal window cosine — chance level for equivalence."""
    R = ref_c.shape[1]
    n = R - width + 1
    if n < 2:
        return float("nan")
    step = max(1, n // 24)
    starts = list(range(0, n, step))
    cs = [_win_cos(ref_c[:, i:i + width], ref_c[:, j:j + width])
          for ai, i in enumerate(starts) for j in starts[ai + 1:]
          if abs(i - j) >= width]
    return float(np.median(cs)) if cs else float("nan")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set-id", default="1fsnxchk")
    p.add_argument("--gt", type=Path,
                   default=_REPO / "labeling/fixtures/bb12_ground_truth.yaml")
    p.add_argument("--thresh", type=float, default=0.80)
    p.add_argument("--tol-s", type=float, default=2.0)
    p.add_argument("--max-win-s", type=float, default=20.0,
                   help="cap the query window length (long spans -> slow fftconvolve)")
    args = p.parse_args(argv)

    set_dir = find_aligning_dir(args.set_id)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    for t in manifest["tracks"]:
        if t.get("recording_id"):
            by_tid.setdefault(t["recording_id"], t)

    mix_c: dict[str, np.ndarray] = {}
    for stem, (fname, _) in _MIX_SOURCE.items():
        f = set_dir / fname
        if f.is_file():
            print(f"chroma({fname}) …", file=sys.stderr)
            mix_c[stem] = _load_chroma(f, f"{args.set_id}_mix_{stem}")

    rows = _straight_rows(args.gt)
    results: list[dict] = []
    skipped = 0
    for r in rows:
        tid = str(r["track_id"])
        t = by_tid.get(tid)
        stem = r.get("claimed_stem") or "regular"
        if t is None or stem not in mix_c:
            skipped += 1
            continue
        ref_path = ref_audio_for({"claimed_stem": stem}, t)
        if ref_path is None:
            skipped += 1
            continue

        set_start, set_end = float(r["set_start_s"]), float(r["set_end_s"])
        a = int(set_start * FPS)
        n = int(min(set_end - set_start, args.max_win_s) * FPS)
        mc = mix_c[stem]
        a = min(a, max(0, mc.shape[1] - n))
        win = mc[:, a:a + n]
        if win.shape[1] < n // 2 or win.shape[1] < 8:
            skipped += 1
            continue
        ref_c = _load_chroma(ref_path, f"ref_{tid}_{stem}")
        if ref_c.shape[1] <= win.shape[1]:
            skipped += 1
            continue

        pred_ref_start, peak, stretch = detect_offset(win, ref_c)
        gt_ref_start = float(r["ref_start_s"])
        err_s = abs(pred_ref_start - gt_ref_start)
        exact = err_s < args.tol_s

        w = win.shape[1]
        pf = int(round(pred_ref_start * FPS))
        gf = int(round(gt_ref_start * FPS))
        eq_cos = _win_cos(ref_c[:, pf:pf + w], ref_c[:, gf:gf + w])
        equiv = eq_cos >= args.thresh
        results.append({
            "stem": stem, "err_s": err_s, "exact": exact, "equiv": equiv,
            "eq_cos": eq_cos, "peak": peak, "floor": _floor(ref_c, w),
            "name": str(r.get("track", ""))[:40],
        })

    n = len(results)
    if n == 0:
        print("no scorable rows", file=sys.stderr)
        return 1

    def pct(key: str, sub: list[dict]) -> str:
        return f"{100 * np.mean([x[key] for x in sub]):3.0f}%" if sub else "   -"

    print(f"=== v0.1 chroma equivalence-aware scorecard ({args.set_id}) ===")
    print(f"straight clips scored: {n}  (skipped {skipped})  "
          f"thresh={args.thresh} tol={args.tol_s}s win<= {args.max_win_s:.0f}s")
    print(f"{'stem':13} {'n':>3}  {'exact<tol':>9}  {'equiv':>6}  {'rescued':>9}")
    for stem in ("regular", "acappella", "instrumental", None):
        sub = results if stem is None else [x for x in results if x["stem"] == stem]
        if not sub:
            continue
        resc = [x for x in sub if x["equiv"] and not x["exact"]]
        label = "ALL" if stem is None else stem
        print(f"{label:13} {len(sub):3}  {pct('exact', sub):>9}  "
              f"{pct('equiv', sub):>6}  {len(resc):3} ({100*len(resc)/len(sub):2.0f}%)")

    errs = np.array([x["err_s"] for x in results])
    peaks = np.array([x["peak"] for x in results])
    floors = np.array([x["floor"] for x in results if np.isfinite(x["floor"])])
    print(f"\nraw |pred-gt|: median={np.median(errs):.1f}s  p90={np.percentile(errs,90):.1f}s")
    print(f"matched-filter peak: median={np.median(peaks):.2f}  p10={np.percentile(peaks,10):.2f}")
    if floors.size:
        print(f"chroma self-sim floor: median={np.median(floors):.2f}  "
              f"-> thresh {args.thresh} is {'above' if args.thresh > np.median(floors) else 'BELOW'} chance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
