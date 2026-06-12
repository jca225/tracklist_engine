#!/usr/bin/env python3
"""Joint per-span ref decode: continuity + loops via a jump-Viterbi.

Single-window matched filtering (refine_ref_offsets) finds the right
CONTENT but coin-flips among self-similar repeats, and cannot represent
loops at all. This module does what the human annotator does: look at the
whole span and require the song to move forward sensibly.

Per span:
  * windows every HOP_S across the span (stem-routed mix chroma)
  * score matrix S[w, k] = matched-filter curve of window w over the whole
    ref (grid-derived stretch, octave-folded)
  * Viterbi path through S with two moves between consecutive windows:
      advance:  ref position += HOP_S * stretch (± tolerance)  — free
      jump:     anywhere in the song                            — penalty
    A backward jump is a LOOP; a forward jump is a DJ edit/cut.
  * output: ref_segments [(mix_start, ref_start, dur)] — the GT schema's
    shape — plus span-level ref_start (first segment) and path confidence.

Default mode updates out/<set_id>_predicted_timeline.json in place
(adds ref_segments + ref_path_conf; rewrites ref_start_s/ref_end_s from
the first/last segment).

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.joint_ref_decode \\
        --set-id 1fsnxchk [--hop-s 4] [--win-s 8] [--jump-penalty 0.25]
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
    HOP, SR, _MIX_SOURCE, _STEM_FILE, chroma, find_aligning_dir,
)

OUT_DIR = Path(__file__).resolve().parent / "out"

_ADV_TOL_S = 0.75      # slack around the expected advance per hop
_MAX_SPAN_S = 90.0     # decode at most this much of a span (cost guard)
_MIN_SEG_S = 3.0       # merge segments shorter than this into neighbors


def _scores_matrix(wins: np.ndarray, ref_f: np.ndarray, stretch: float) -> np.ndarray:
    """S[w, k] — normalized correlation of each window over the ref."""
    from scipy.signal import fftconvolve
    n = wins.shape[2]
    m = int(round(n * stretch))
    idx = np.clip((np.arange(m) / stretch).astype(int), 0, n - 1)
    if ref_f.shape[1] <= m:
        return np.zeros((wins.shape[0], 0), np.float32)
    e = np.concatenate([[0.0], np.cumsum((ref_f ** 2).sum(axis=0))])
    den = np.sqrt(np.maximum(e[m:] - e[:-m], 1e-9))
    rows = []
    for w in wins:
        ww = w[:, idx]
        ww = ww / (np.linalg.norm(ww) + 1e-9)
        num = fftconvolve(ref_f, ww[:, ::-1], mode="valid", axes=1).sum(axis=0)
        rows.append((num / den).astype(np.float32))
    return np.stack(rows)


def jump_viterbi(S: np.ndarray, adv: int, tol: int, jump_penalty: float
                 ) -> tuple[np.ndarray, float]:
    """Best path k(w) through S with advance-or-jump transitions."""
    from scipy.ndimage import maximum_filter1d
    W, K = S.shape
    V = S[0].astype(np.float64).copy()
    back = np.zeros((W, K), dtype=np.int32)
    idx = np.arange(K)
    for w in range(1, W):
        # advance: V[w-1] shifted by adv, sliding max over ±tol
        slid = maximum_filter1d(V, size=2 * tol + 1, mode="constant", cval=-1e9)
        prev_idx_adv = idx - adv
        adv_score = np.where(
            (prev_idx_adv >= 0) & (prev_idx_adv < K),
            slid[np.clip(prev_idx_adv, 0, K - 1)], -1e9,
        )
        # jump: global best minus penalty
        g = int(V.argmax())
        jump_score = V[g] - jump_penalty
        take_jump = jump_score > adv_score
        # backpointer: exact argmax within the advance window, or g
        for k in np.flatnonzero(~take_jump):
            lo = max(0, k - adv - tol)
            hi = min(K, k - adv + tol + 1)
            back[w, k] = lo + int(V[lo:hi].argmax())
        back[w, take_jump] = g
        V = S[w] + np.where(take_jump, jump_score, adv_score)
    k = int(V.argmax())
    path = np.zeros(W, dtype=np.int32)
    path[-1] = k
    for w in range(W - 1, 0, -1):
        path[w - 1] = back[w, path[w]]
    return path, float(V[k] / W)


def path_to_segments(path: np.ndarray, hop_s: float, stretch: float,
                     set_start: float, adv: int, tol: int) -> list[dict]:
    """Group the path into linear segments; a non-advance step starts one."""
    segs: list[dict] = []
    seg_w0 = 0
    for w in range(1, len(path)):
        if abs(int(path[w]) - int(path[w - 1]) - adv) > tol:
            segs.append({"w0": seg_w0, "w1": w - 1})
            seg_w0 = w
    segs.append({"w0": seg_w0, "w1": len(path) - 1})
    out = []
    for s in segs:
        dur = (s["w1"] - s["w0"] + 1) * hop_s
        if out and dur < _MIN_SEG_S:
            out[-1]["dur_s"] = round(out[-1]["dur_s"] + dur, 2)
            continue
        out.append({
            "mix_start_s": round(set_start + s["w0"] * hop_s, 2),
            "ref_start_s": round(float(path[s["w0"]]) * HOP / SR, 2),
            "dur_s": round(dur, 2),
        })
    return out


def _span_job(args: tuple) -> dict:
    (slot, ref_path, wins, stretch, set_start, hop_s, jump_penalty) = args
    import librosa
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref_y, _ = librosa.load(ref_path, sr=SR, mono=True)
    ref_f = chroma(ref_y)
    wins_f = np.asarray(wins, dtype=np.float32)
    S = _scores_matrix(wins_f, ref_f, stretch)
    if S.shape[1] == 0 or S.shape[0] < 2:
        return {"slot": slot, "segments": None, "conf": 0.0}
    adv = max(1, int(round(hop_s * stretch * SR / HOP)))
    tol = max(1, int(round(_ADV_TOL_S * SR / HOP)))
    path, conf = jump_viterbi(S, adv, tol, jump_penalty)
    segs = path_to_segments(path, hop_s, stretch, set_start, adv, tol)
    return {"slot": slot, "segments": segs, "conf": round(conf, 3)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", required=True)
    p.add_argument("--hop-s", type=float, default=4.0)
    p.add_argument("--win-s", type=float, default=8.0)
    p.add_argument("--jump-penalty", type=float, default=0.25)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)

    import librosa

    timeline_path = OUT_DIR / f"{args.set_id}_predicted_timeline.json"
    timeline = json.loads(timeline_path.read_text())
    spans = timeline["spans"]

    set_dir = find_aligning_dir(args.set_id)
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

    n_win_frames = int(args.win_s * SR / HOP)
    hop_frames = int(args.hop_s * SR / HOP)
    jobs = []
    for s in spans:
        t = by_tid.get(s["recording_id"])
        if t is None:
            continue
        stem = s.get("claimed_stem") or "regular"
        ref_path = None
        stem_key = _STEM_FILE.get(stem)
        if stem_key:
            sp = (t.get("stems") or {}).get(stem_key)
            if sp and Path(sp).is_file():
                ref_path = sp
        if ref_path is None:
            ref_path = t["local_path"]
        if not Path(ref_path).is_file():
            continue
        mc = mix_chroma.get(stem, mix_chroma["regular"])
        span_len = min(_MAX_SPAN_S, max(args.win_s, s["set_end_s"] - s["set_start_s"]))
        starts = np.arange(0.0, span_len - args.win_s + 1e-6, args.hop_s)
        wins = []
        for dt in starts:
            a = int((s["set_start_s"] + dt) * SR / HOP)
            a = min(a, max(0, mc.shape[1] - n_win_frames))
            wins.append(mc[:, a:a + n_win_frames])
        if len(wins) < 2:
            continue
        stretch = float(s.get("ref_stretch") or 1.0)
        jobs.append((s["slot_label"], str(ref_path),
                     np.stack(wins).tolist(), stretch,
                     s["set_start_s"], args.hop_s, args.jump_penalty))

    print(f"joint decode: {len(jobs)} spans, win={args.win_s:.0f}s hop={args.hop_s:.0f}s "
          f"jump_penalty={args.jump_penalty}, {args.workers} workers…")
    results: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(_span_job, jobs, chunksize=2)):
            results[r["slot"]] = r
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(jobs)}")

    n_multi, updated = 0, 0
    for s in spans:
        r = results.get(s["slot_label"])
        if not r or not r["segments"]:
            continue
        segs = r["segments"]
        s["ref_segments"] = segs
        s["ref_path_conf"] = r["conf"]
        s.setdefault("ref_start_detect", s["ref_start_s"])
        s["ref_start_s"] = segs[0]["ref_start_s"]
        last = segs[-1]
        stretch = float(s.get("ref_stretch") or 1.0)
        s["ref_end_s"] = round(last["ref_start_s"] + last["dur_s"] * stretch, 2)
        updated += 1
        if len(segs) > 1:
            n_multi += 1

    timeline["ref_decode"] = "joint jump-viterbi (joint_ref_decode)"
    timeline_path.write_text(json.dumps(timeline, indent=2))
    confs = [r["conf"] for r in results.values() if r["segments"]]
    print(f"\nupdated {updated} spans; multi-segment (loops/edits): {n_multi}; "
          f"path conf median={np.median(confs):.2f}")
    print(f"rewrote {timeline_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
