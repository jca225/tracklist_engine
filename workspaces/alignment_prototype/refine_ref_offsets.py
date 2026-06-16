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
    e = np.concatenate([[0.0], np.cumsum((rf**2).sum(axis=0))])
    den = np.sqrt(np.maximum(e[m:] - e[:-m], 1e-9))
    scores = num / den
    k = int(scores.argmax())
    return k, float(scores[k])


def detect_offset(
    win_f: np.ndarray,
    ref_f: np.ndarray,
    stretches: tuple[float, ...] = STRETCHES,
) -> tuple[float, float, float]:
    """(ref_start_s, peak, stretch) — search the given stretch factors.

    stretch = ref seconds per mix second: the mix window is resampled to
    stretch*len before matching, so a hit at stretch s means the DJ played
    the song at 1/s speed."""
    n = win_f.shape[1]
    best = (0.0, 0.0, 1.0)
    for st in stretches:
        m = int(round(n * st))
        idx = np.clip((np.arange(m) / st).astype(int), 0, n - 1)
        k, score = correlate_window(win_f[:, idx], ref_f)
        if score > best[1]:
            best = (k * HOP / SR, score, st)
    return best


def _span_job(args: tuple) -> dict:
    """Worker: load ref audio, chroma, detect. Returns updates for the span.

    When fp_cfg is set, ALSO run a landmark-fingerprint offset on the mix-audio
    window vs the same ref audio — a high-vote fingerprint hit is a sharper,
    higher-precision localizer than the chroma matched filter and recovers the
    wrong-content errors it makes (validated by fp_fuse: +6pp fiber-aware)."""
    span, ref_path, win, stretches, fp_cfg = args
    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref_y, _ = librosa.load(ref_path, sr=SR, mono=True)
    ref_f = chroma(ref_y)
    win_f = np.asarray(win, dtype=np.float32)
    ref_start, peak, stretch = detect_offset(win_f, ref_f, tuple(stretches))
    out = {
        "slot_label": span["slot_label"],
        "ref_start_s": round(ref_start, 3),
        "ref_peak": round(peak, 3),
        "ref_stretch": stretch,
        "fp_votes": 0,
    }
    if fp_cfg is not None:
        from workspaces.alignment_prototype.fp_probe import fp_offset

        mix_audio, s0, fp_win, fp_stretches = fp_cfg
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mw, _ = librosa.load(
                mix_audio, sr=SR, mono=True, offset=s0, duration=fp_win
            )
        fp_off, votes, _ = fp_offset(mw, ref_y, tuple(fp_stretches))
        out["fp_ref_start"] = round(fp_off, 3)
        out["fp_votes"] = int(votes)
    return out


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
    p.add_argument(
        "--min-peak",
        type=float,
        default=0.55,
        help="below this, keep the decode offset and flag the span",
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--grid-stretch",
        action="store_true",
        help="fix stretch from beat grids (instrumental-BPM-anchor "
        "heuristic): ratio = ref bar dur / mix bar dur, "
        "searched only ±2%% + half/double-time octaves",
    )
    p.add_argument(
        "--fingerprint",
        action="store_true",
        help="also run a landmark-fingerprint offset per span and OVERRIDE the "
        "chroma match when its vote count is high (recovers wrong-content "
        "errors the matched filter makes; fp_fuse validated +6pp fiber-aware)",
    )
    p.add_argument(
        "--fp-votes",
        type=int,
        default=40,
        help="min fingerprint votes to override the chroma offset",
    )
    p.add_argument(
        "--fp-win-s",
        type=float,
        default=15.0,
        help="mix-audio probe window for the fingerprint",
    )
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

    grids = None
    if args.grid_stretch:
        from core.result import Err, Ok
        from workspaces.alignment_prototype.mert_store import load_bb12_mert

        match load_bb12_mert(args.set_id):
            case Err(msg):
                sys.exit(f"--grid-stretch needs the MERT bundle (beat grids): {msg}")
            case Ok((_sid, mix_series, ref_series)):
                grids = (mix_series, ref_series)

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
        win = mc[:, a : a + n]
        if win.shape[1] < n // 2:
            continue
        stretches = STRETCHES
        if grids is not None and s["recording_id"] in grids[1]:
            mix_series, ref_series = grids
            i = int(np.searchsorted(mix_series.start_s, s["set_start_s"]))
            lo, hi = max(0, i - 2), min(mix_series.n_measures, i + 3)
            mix_bar = float(
                np.median(mix_series.end_s[lo:hi] - mix_series.start_s[lo:hi])
            )
            rser = ref_series[s["recording_id"]]
            ref_bar = float(np.median(rser.end_s - rser.start_s))
            if mix_bar > 0 and ref_bar > 0:
                e = ref_bar / mix_bar
                # Fold metrical-level (octave) mismatch between the two
                # beat_this grids: the SECONDS-stretch is near 1 regardless
                # of how each side tagged its bar level. No octave
                # candidates in the search — comparing matched-filter peaks
                # across 2x-different query lengths is length-biased (the
                # shorter query always wins; measured BB11 2026-06-11).
                while e > 1.45:
                    e *= 0.5
                while e < 0.7:
                    e *= 2.0
                stretches = tuple(e * f for f in (0.96, 0.98, 1.0, 1.02, 1.04))
        fp_cfg = None
        if args.fingerprint:
            mix_file = set_dir / _MIX_SOURCE.get(stem, _MIX_SOURCE["regular"])[0]
            if not mix_file.is_file():
                mix_file = set_dir / "mix.m4a"
            fp_win = min(args.fp_win_s, s["set_end_s"] - s["set_start_s"])
            fp_cfg = (str(mix_file), s["set_start_s"], fp_win, (0.98, 1.0, 1.02))
        jobs.append((s, str(ref), win.tolist(), stretches, fp_cfg))
        meta.append(s)

    print(
        f"detecting ref offsets for {len(jobs)} spans "
        f"(window={args.window_s:.0f}s, {len(STRETCHES)} stretches, "
        f"{args.workers} workers)…"
    )
    results: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(_span_job, jobs, chunksize=2)):
            results[r["slot_label"]] = r
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(jobs)}")

    updated, weak, fp_overrides = 0, [], 0
    for s in spans:
        r = results.get(s["slot_label"])
        if r is None:
            continue
        s.setdefault(
            "ref_start_decode", s["ref_start_s"]
        )  # keep original provenance on re-runs
        s["ref_peak"] = r["ref_peak"]
        span_len = s["set_end_s"] - s["set_start_s"]
        # high-vote fingerprint OVERRIDES the chroma match (sharper, recovers
        # wrong-content); else fall back to the matched-filter offset.
        if args.fingerprint and r.get("fp_votes", 0) >= args.fp_votes:
            s["ref_start_fp"] = r["fp_ref_start"]
            s["ref_fp_votes"] = r["fp_votes"]
            s["ref_start_s"] = r["fp_ref_start"]
            s["ref_end_s"] = round(r["fp_ref_start"] + span_len * r["ref_stretch"], 3)
            s["ref_stretch"] = r["ref_stretch"]
            s["ref_source_method"] = "fingerprint"
            updated += 1
            fp_overrides += 1
        elif r["ref_peak"] >= args.min_peak:
            s["ref_start_s"] = r["ref_start_s"]
            s["ref_end_s"] = round(r["ref_start_s"] + span_len * r["ref_stretch"], 3)
            s["ref_stretch"] = r["ref_stretch"]
            s["ref_source_method"] = "chroma"
            updated += 1
        else:
            weak.append((s["slot_label"], r["ref_peak"], s["name"][:45]))

    method = (
        "matched-filter chroma + fingerprint override"
        if args.fingerprint
        else "matched-filter chroma detection"
    )
    timeline["ref_offsets"] = f"{method} (refine_ref_offsets)"
    timeline_path.write_text(json.dumps(timeline, indent=2))

    peaks = np.array([r["ref_peak"] for r in results.values()])
    print(
        f"\nupdated {updated}/{len(results)} spans "
        f"(peak median={np.median(peaks):.2f} p10={np.percentile(peaks, 10):.2f})"
    )
    if args.fingerprint:
        print(f"  fingerprint overrode {fp_overrides} spans (votes >= {args.fp_votes})")
    if weak:
        print(f"{len(weak)} weak spans kept decode offsets (peak < {args.min_peak}):")
        for slot, pk, name in weak:
            print(f"  {slot:6} peak={pk:.2f}  {name}")
    print(f"rewrote {timeline_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
