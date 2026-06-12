#!/usr/bin/env python3
"""Temporal detection of ref offsets: matched-filter each span's mix window
against the full reference track.

The cross-set decode places spans in the mix (cue-anchored) but has no
working signal for WHERE IN THE SONG the span comes from — predicted
ref_start collapses to ~0 (track intro). Empirically (BB11, 2026-06-11) the
true offsets are 30-120 s in, and a 12 s chroma matched filter finds them
with peaks 0.67-0.99 while the predicted offsets score far lower.

Per span:
  * mix window  <- chroma of the mix at the predicted set position
                   (stem-routed: acappella spans use the roformer
                   mix_vocals stem vs the ref's Demucs vocals, instrumental
                   spans the instrumental pair)
  * search      <- FFT cross-correlation of the window against the whole
                   ref at stretch factors 0.92-1.08; best (offset, stretch)
                   wins. stretch == ref-seconds per mix-second, so it also
                   yields the warp ratio for the .als / review player.
  * output      <- ref_start_s / ref_end_s rewritten in the timeline JSON,
                   plus ref_peak + ref_start_decode (provenance). Spans with
                   peak < --min-peak keep the decode value and are flagged.

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.refine_ref_offsets \\
        --set-id 2nvzlh2k [--window-s 12] [--workers 8]
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
STRETCHES = (0.92, 0.95, 0.98, 1.0, 1.02, 1.05, 1.08)

_MIX_SOURCE = {  # claimed_stem -> (mix file, ref stem key)
    "regular": ("mix.m4a", None),
    "acappella": ("mix_vocals.flac", "vocals"),
    "instrumental": ("mix_instrumental.flac", "instrumental"),
}


def chroma(y: np.ndarray) -> np.ndarray:
    import librosa
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP)
    return librosa.util.normalize(c, axis=0).astype(np.float32)


def correlate_window(wf: np.ndarray, rf: np.ndarray) -> tuple[int, float]:
    """Best (frame, normalized score) of window wf sliding over ref rf."""
    from scipy.signal import fftconvolve
    m = wf.shape[1]
    if rf.shape[1] <= m:
        return 0, 0.0
    w = wf / (np.linalg.norm(wf) + 1e-9)
    # correlation = convolution with time-reversed kernel, summed over chroma
    num = fftconvolve(rf, w[:, ::-1], mode="valid", axes=1).sum(axis=0)
    # sliding L2 norm of ref windows
    e = np.concatenate([[0.0], np.cumsum((rf ** 2).sum(axis=0))])
    den = np.sqrt(np.maximum(e[m:] - e[:-m], 1e-9))
    scores = num / den
    k = int(scores.argmax())
    return k, float(scores[k])


def detect_offset(
    win_f: np.ndarray, ref_f: np.ndarray,
) -> tuple[float, float, float]:
    """(ref_start_s, peak, stretch) — search all stretch factors.

    stretch = ref seconds per mix second: the mix window is resampled to
    stretch*len before matching, so a hit at stretch s means the DJ played
    the song at 1/s speed."""
    n = win_f.shape[1]
    best = (0.0, 0.0, 1.0)
    for st in STRETCHES:
        m = int(round(n * st))
        idx = np.clip((np.arange(m) / st).astype(int), 0, n - 1)
        k, score = correlate_window(win_f[:, idx], ref_f)
        if score > best[1]:
            best = (k * HOP / SR, score, st)
    return best


def _span_job(args: tuple) -> dict:
    """Worker: load ref audio, chroma, detect. Returns updates for the span."""
    span, ref_path, win = args
    import librosa
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref_y, _ = librosa.load(ref_path, sr=SR, mono=True)
    ref_f = chroma(ref_y)
    win_f = np.asarray(win, dtype=np.float32)
    ref_start, peak, stretch = detect_offset(win_f, ref_f)
    return {
        "slot_label": span["slot_label"],
        "ref_start_s": round(ref_start, 3),
        "ref_peak": round(peak, 3),
        "ref_stretch": stretch,
    }


def find_aligning_dir(set_id: str) -> Path:
    hits = sorted(ALIGNING_ROOT.glob(f"{set_id}__*"))
    if not hits:
        sys.exit(f"no ~/aligning folder for {set_id}")
    return hits[0]


_STEM_FILE = {"acappella": "vocals", "instrumental": "instrumental"}


def ref_audio_for(span: dict, track: dict) -> Path | None:
    stem_key = _STEM_FILE.get(span.get("claimed_stem") or "regular")
    if stem_key:
        p = (track.get("stems") or {}).get(stem_key)
        if p and Path(p).is_file():
            return Path(p)
    p = Path(track["local_path"])
    return p if p.is_file() else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", required=True)
    p.add_argument("--window-s", type=float, default=12.0)
    p.add_argument("--min-peak", type=float, default=0.55,
                   help="below this, keep the decode offset and flag the span")
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

    # chroma of each mix source once; windows are sliced per span
    mix_chroma: dict[str, np.ndarray] = {}
    for stem, (fname, _) in _MIX_SOURCE.items():
        f = set_dir / fname
        if f.is_file():
            print(f"chroma({fname}) …")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y, _sr = librosa.load(str(f), sr=SR, mono=True)
            mix_chroma[stem] = chroma(y)
    if "regular" not in mix_chroma:
        sys.exit("mix.m4a missing")

    jobs, meta = [], []
    for s in spans:
        t = by_tid.get(s["recording_id"])
        ref = ref_audio_for(s, t) if t else None
        if ref is None:
            continue
        stem = s.get("claimed_stem") or "regular"
        mc = mix_chroma.get(stem, mix_chroma["regular"])
        a = int(s["set_start_s"] * SR / HOP)
        n = int(args.window_s * SR / HOP)
        a = min(a, max(0, mc.shape[1] - n))
        win = mc[:, a:a + n]
        if win.shape[1] < n // 2:
            continue
        jobs.append((s, str(ref), win.tolist()))
        meta.append(s)

    print(f"detecting ref offsets for {len(jobs)} spans "
          f"(window={args.window_s:.0f}s, {len(STRETCHES)} stretches, "
          f"{args.workers} workers)…")
    results: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(_span_job, jobs, chunksize=2)):
            results[r["slot_label"]] = r
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(jobs)}")

    updated, weak = 0, []
    for s in spans:
        r = results.get(s["slot_label"])
        if r is None:
            continue
        s["ref_start_decode"] = s["ref_start_s"]
        s["ref_peak"] = r["ref_peak"]
        if r["ref_peak"] >= args.min_peak:
            span_len = s["set_end_s"] - s["set_start_s"]
            s["ref_start_s"] = r["ref_start_s"]
            s["ref_end_s"] = round(r["ref_start_s"] + span_len * r["ref_stretch"], 3)
            s["ref_stretch"] = r["ref_stretch"]
            updated += 1
        else:
            weak.append((s["slot_label"], r["ref_peak"], s["name"][:45]))

    timeline["ref_offsets"] = "matched-filter chroma detection (refine_ref_offsets)"
    timeline_path.write_text(json.dumps(timeline, indent=2))

    peaks = np.array([r["ref_peak"] for r in results.values()])
    print(f"\nupdated {updated}/{len(results)} spans "
          f"(peak median={np.median(peaks):.2f} p10={np.percentile(peaks, 10):.2f})")
    if weak:
        print(f"{len(weak)} weak spans kept decode offsets (peak < {args.min_peak}):")
        for slot, pk, name in weak:
            print(f"  {slot:6} peak={pk:.2f}  {name}")
    print(f"rewrote {timeline_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
