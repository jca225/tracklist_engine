#!/usr/bin/env python3
"""Beat-domain ref-offset detection (v2) — the instrumental-BPM-anchor
heuristic operationalized.

Domain facts (user, 2026-06-11): within a span the host instrumental's BPM
never changes, and overlaid acappellas are beat-synced to the host grid. So
tempo is not a free parameter: resample both signals onto their own measure
grids (mix: set_measures via the MERT bundle; ref: track_measures likewise)
and correlate in *bar space*, where stretch drops out entirely and candidate
offsets are bar-quantized (DJs sync on bars). The only residual freedom is
half/double-time, tested as discrete bar mappings {1:1, 2:1, 1:2}.

vs v1 (refine_ref_offsets.py, seconds-space stretch grid): no stretch grid
to saturate (v1 piled up at its 0.92/1.08 edges), ~100x fewer candidate
offsets, and the implied warp ratio comes from the grids themselves:
ratio = ref bar duration / mix bar duration at the matched bars.

**MEASURED RESULT (BB11, 2026-06-11): WORSE THAN v1 — do not --apply.**
Only 41/151 spans agreed with v1 within 2 s; the big disagreements almost
all chose the 0.5/2.0 mappings, i.e. coarse bar-locked chroma (8 samples/
bar) plus free octave mappings matches self-similar song repeats instead
of the true section, and v1's peak was usually higher where they differed.
The heuristic's correct application is refine_ref_offsets --grid-stretch:
keep v1's full-resolution chroma and use the grids only to FIX the stretch
(ratio = ref bar dur / mix bar dur, ±2% + octave checks). Kept for the
compare harness and as the recorded negative result.

Default mode COMPARES v2 against the offsets already in the timeline JSON
(does not write). --apply rewrites ref_start_s/ref_end_s like v1 does.

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.beat_domain_detect \\
        --set-id 2nvzlh2k [--bars 8] [--apply] [--workers 8]
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

OUT_DIR = Path(__file__).resolve().parent / "out"
ALIGNING_ROOT = Path.home() / "aligning"

SR = 22050
HOP = 512
SAMPLES_PER_BAR = 8          # chroma samples per bar after grid-locking
MAPPINGS = (1.0, 2.0, 0.5)   # ref bars per mix bar (half/double-time checks)

_MIX_SOURCE = {
    "regular": "mix.m4a",
    "acappella": "mix_vocals.flac",
    "instrumental": "mix_instrumental.flac",
}
_STEM_FILE = {"acappella": "vocals", "instrumental": "instrumental"}


def chroma(y: np.ndarray) -> np.ndarray:
    import librosa
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP)
    return librosa.util.normalize(c, axis=0).astype(np.float32)


def bar_locked(feat: np.ndarray, starts: np.ndarray, ends: np.ndarray,
               per_bar: int = SAMPLES_PER_BAR) -> np.ndarray:
    """(12, n_bars * per_bar) — chroma sampled at uniform fractions of each
    bar, so tempo is normalized away."""
    cols = []
    n = feat.shape[1]
    for t0, t1 in zip(starts, ends):
        ts = t0 + (np.arange(per_bar) + 0.5) / per_bar * max(t1 - t0, 1e-3)
        idx = np.clip((ts * SR / HOP).astype(int), 0, n - 1)
        cols.append(feat[:, idx])
    return np.concatenate(cols, axis=1) if cols else np.zeros((12, 0), np.float32)


def slide_scores(ref_bl: np.ndarray, win: np.ndarray, stride: int) -> np.ndarray:
    """Normalized correlation of win against ref_bl at every `stride` offset."""
    m = win.shape[1]
    if ref_bl.shape[1] < m:
        return np.zeros(0, np.float32)
    w = (win / (np.linalg.norm(win) + 1e-9)).ravel()
    n_off = (ref_bl.shape[1] - m) // stride + 1
    # windows as strided view -> (n_off, 12*m)
    out = np.empty(n_off, np.float32)
    for i in range(n_off):
        seg = ref_bl[:, i * stride: i * stride + m]
        out[i] = float(w @ (seg.ravel() / (np.linalg.norm(seg) + 1e-9)))
    return out


def detect_bar_offset(
    win_bl: np.ndarray,          # (12, W*per_bar) mix window, bar-locked
    ref_feat: np.ndarray,        # raw ref chroma
    ref_starts: np.ndarray,
    ref_ends: np.ndarray,
) -> tuple[int, float, float]:
    """(ref_bar_index, peak, mapping) best across half/double-time mappings."""
    W = win_bl.shape[1] // SAMPLES_PER_BAR
    best = (0, 0.0, 1.0)
    for mp in MAPPINGS:
        # mapping mp: one mix bar covers mp ref bars -> sample ref at
        # per_bar/mp per ref bar so W mix bars align with W*mp ref bars
        pb = max(2, int(round(SAMPLES_PER_BAR / mp)))
        ref_bl = bar_locked(ref_feat, ref_starts, ref_ends, per_bar=pb)
        m = W * SAMPLES_PER_BAR  # target window length in samples
        if ref_bl.shape[1] < m:
            continue
        scores = slide_scores(ref_bl, win_bl, stride=pb)
        if not scores.size:
            continue
        k = int(scores.argmax())
        if scores[k] > best[1]:
            best = (k, float(scores[k]), mp)
    return best


def _span_job(args: tuple) -> dict:
    span, ref_path, win_bl_list, ref_starts, ref_ends = args
    import librosa
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref_y, _ = librosa.load(ref_path, sr=SR, mono=True)
    ref_feat = chroma(ref_y)
    rs = np.asarray(ref_starts)
    re_ = np.asarray(ref_ends)
    win_bl = np.asarray(win_bl_list, dtype=np.float32)
    bar, peak, mp = detect_bar_offset(win_bl, ref_feat, rs, re_)
    ref_start = float(rs[bar]) if bar < len(rs) else 0.0
    # implied warp ratio from the grids at the match site
    i1 = min(bar + 4, len(rs) - 1)
    ref_bar_dur = (re_[bar:i1 + 1] - rs[bar:i1 + 1]).mean() if i1 >= bar else 2.0
    return {
        "slot_label": span["slot_label"],
        "ref_start_v2": round(ref_start, 3),
        "peak_v2": round(peak, 3),
        "mapping": mp,
        "ref_bar_dur": float(ref_bar_dur),
    }


def find_aligning_dir(set_id: str) -> Path:
    hits = sorted(ALIGNING_ROOT.glob(f"{set_id}__*"))
    if not hits:
        sys.exit(f"no ~/aligning folder for {set_id}")
    return hits[0]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", required=True)
    p.add_argument("--bars", type=int, default=8, help="mix window length in bars")
    p.add_argument("--min-peak", type=float, default=0.55)
    p.add_argument("--apply", action="store_true",
                   help="rewrite timeline ref offsets (default: compare only)")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)

    import librosa
    from core.result import Err, Ok
    from workspaces.alignment_prototype.mert_store import load_bb12_mert

    timeline_path = OUT_DIR / f"{args.set_id}_predicted_timeline.json"
    timeline = json.loads(timeline_path.read_text())
    spans = timeline["spans"]

    match load_bb12_mert(args.set_id):
        case Err(msg):
            sys.exit(f"MERT bundle (beat grids) load failed: {msg}")
        case Ok((_sid, mix_series, ref_series)):
            pass
    mix_starts, mix_ends = mix_series.start_s, mix_series.end_s

    set_dir = find_aligning_dir(args.set_id)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    for t in manifest["tracks"]:
        if t.get("recording_id"):
            by_tid.setdefault(t["recording_id"], t)

    mix_bl: dict[str, np.ndarray] = {}
    for stem, fname in _MIX_SOURCE.items():
        f = set_dir / fname
        if not f.is_file():
            continue
        print(f"bar-locked chroma({fname}) …")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, _sr = librosa.load(str(f), sr=SR, mono=True)
        mix_bl[stem] = bar_locked(chroma(y), mix_starts, mix_ends)

    jobs = []
    for s in spans:
        t = by_tid.get(s["recording_id"])
        if t is None or s["recording_id"] not in ref_series:
            continue
        stem = s.get("claimed_stem") or "regular"
        stem_key = _STEM_FILE.get(stem)
        ref_path = None
        if stem_key:
            sp = (t.get("stems") or {}).get(stem_key)
            if sp and Path(sp).is_file():
                ref_path = sp
        if ref_path is None:
            ref_path = t["local_path"]
        if not Path(ref_path).is_file():
            continue
        bl = mix_bl.get(stem, mix_bl["regular"])
        bar0 = int(np.searchsorted(mix_starts, s["set_start_s"]))
        bar0 = min(bar0, max(0, len(mix_starts) - args.bars))
        win = bl[:, bar0 * SAMPLES_PER_BAR:(bar0 + args.bars) * SAMPLES_PER_BAR]
        if win.shape[1] < args.bars * SAMPLES_PER_BAR:
            continue
        rser = ref_series[s["recording_id"]]
        jobs.append((s, ref_path, win.tolist(),
                     rser.start_s.tolist(), rser.end_s.tolist()))

    print(f"beat-domain detection: {len(jobs)} spans, {args.bars}-bar windows, "
          f"mappings {MAPPINGS}, {args.workers} workers…")
    results: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(_span_job, jobs, chunksize=2)):
            results[r["slot_label"]] = r
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(jobs)}")

    # ---- compare against current timeline offsets ---------------------------
    diffs, rows = [], []
    for s in spans:
        r = results.get(s["slot_label"])
        if r is None:
            continue
        d = abs(r["ref_start_v2"] - s["ref_start_s"])
        diffs.append(d)
        rows.append((d, s["slot_label"], s.get("ref_peak", 0.0), r["peak_v2"],
                     s["ref_start_s"], r["ref_start_v2"], r["mapping"],
                     s["name"][:40]))
    d = np.asarray(diffs)
    print(f"\nv2 vs current: n={len(d)} agree<2s: {(d < 2).sum()} "
          f"agree<5s: {(d < 5).sum()} median={np.median(d):.1f}s")
    rows.sort(reverse=True)
    print(f"\n{'Δs':>7} {'slot':6} {'v1pk':>5} {'v2pk':>5} {'v1@':>7} {'v2@':>7} map  name")
    for dd, slot, p1, p2, a1, a2, mp, name in rows[:20]:
        print(f"{dd:7.1f} {slot:6} {p1:5.2f} {p2:5.2f} {a1:7.1f} {a2:7.1f} {mp:3.1f}  {name}")

    if args.apply:
        updated = 0
        for s in spans:
            r = results.get(s["slot_label"])
            if r is None or r["peak_v2"] < args.min_peak:
                continue
            s.setdefault("ref_start_decode", s["ref_start_s"])
            span_len = s["set_end_s"] - s["set_start_s"]
            mix_bar = float(np.mean(mix_ends[:8] - mix_starts[:8]))
            i = int(np.searchsorted(mix_starts, s["set_start_s"]))
            i = min(i, len(mix_starts) - 1)
            mix_bar = float(mix_ends[i] - mix_starts[i]) or mix_bar
            ratio = r["ref_bar_dur"] * r["mapping"] / mix_bar
            s["ref_start_s"] = r["ref_start_v2"]
            s["ref_end_s"] = round(r["ref_start_v2"] + span_len * ratio, 3)
            s["ref_peak"] = r["peak_v2"]
            s["ref_stretch"] = round(ratio, 4)
            s["ref_detector"] = "beat_domain_v2"
            updated += 1
        timeline_path.write_text(json.dumps(timeline, indent=2))
        print(f"\napplied: {updated} spans rewritten -> {timeline_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
