#!/usr/bin/env python3
"""Landmark-fingerprint localizer — the sharp signal for the wrong-CLASS ~30%.

The chroma/HuBERT matched filter localizes by smooth similarity, so it can land
on the wrong content entirely (path_decode picked ref 147 s when GT was 48 s on
006w2). A landmark constellation (Shazam-style spectral-peak pairs) localizes by
EXACT transient coincidence: it recovered 48 s with 387 offset votes where the
matched filter missed. Fibers ([[project_fibers]]) handle within-class repeat
ambiguity; this attacks the genuinely-wrong-content cases.

DJ tempo-warp breaks vanilla fingerprinting (landmark time-deltas scale), so we
search a small stretch band (the instrumental-anchor tempo could feed this) and
time-stretch the ref before hashing. The offset HISTOGRAM is the localizer: every
non-repeating transient votes for the one true offset; repeats split their votes,
so it's far sharper than cosine argmax.

NB the schema (track_fingerprints / set_fingerprint_hits) names "chromaprint",
which is tempo-rigid and wrong for warped mixes — landmark hashing replaces it.

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.fp_probe \
        --eval [--stems regular,instrumental] [--max-win-s 15] [--workers 6]
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
    SR,
    _STEM_FILE,
    find_aligning_dir,
)

_NFFT = 2048
_FHOP = 512
_FPS = SR / _FHOP


def constellation(y: np.ndarray, peak_size: int = 19, db_floor: float = 60.0):
    """(time_frames, freq_bins) of spectral-peak landmarks (local maxima)."""
    import librosa
    from scipy.ndimage import maximum_filter

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = librosa.amplitude_to_db(
            np.abs(librosa.stft(y, n_fft=_NFFT, hop_length=_FHOP))
        )
    mx = maximum_filter(s, size=(peak_size, peak_size))
    pk = (s == mx) & (s > s.max() - db_floor)
    fb, tf = np.where(pk)
    return tf.astype(np.int32), fb.astype(np.int32)


def hashes(tf: np.ndarray, fb: np.ndarray, fan: int = 8, dt_max: int = 80) -> dict:
    """{(f1, f2, dt): [anchor_time_frames]} — peak-pair landmark hashes."""
    order = np.argsort(tf)
    tf, fb = tf[order], fb[order]
    h: dict = {}
    n = len(tf)
    for i in range(n):
        for j in range(i + 1, min(i + 1 + fan, n)):
            dt = int(tf[j] - tf[i])
            if 1 <= dt <= dt_max:
                h.setdefault((int(fb[i]) // 2, int(fb[j]) // 2, dt), []).append(
                    int(tf[i])
                )
    return h


def fp_offset(
    mix_y: np.ndarray, ref_y: np.ndarray, stretches: tuple[float, ...]
) -> tuple[float, int, float]:
    """Backward-compatible wrapper (sharpness dropped)."""
    from workspaces.alignment_prototype.landmark_fp import fp_offset as _fp

    off, votes, st, _sharp = _fp(mix_y, ref_y, stretches=stretchs)
    return off, votes, st


def _job(args: tuple) -> dict:
    import librosa

    idx, mix_path, s0, span, ref_path, stretches = args
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mix, _ = librosa.load(mix_path, sr=SR, mono=True, offset=s0, duration=span)
        ref, _ = librosa.load(ref_path, sr=SR, mono=True)
    off, v, st = fp_offset(mix, ref, tuple(stretches))
    return {"idx": idx, "ref_start": round(off, 2), "votes": v, "stretch": st}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval", action="store_true")
    p.add_argument(
        "--gt", type=Path, default=_REPO / "labeling/fixtures/bb12_ground_truth.yaml"
    )
    p.add_argument("--set-id", default="1fsnxchk")
    p.add_argument("--stems", default="regular,instrumental")
    p.add_argument("--max-win-s", type=float, default=15.0)
    p.add_argument("--stretches", default="0.98,1.0,1.02")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args(argv)
    if not args.eval:
        p.error("only --eval is wired")
    want = {s.strip() for s in args.stems.split(",") if s.strip()}
    stretches = tuple(float(x) for x in args.stretches.split(","))

    import yaml

    set_dir = find_aligning_dir(args.set_id)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    for t in manifest["tracks"]:
        if t.get("recording_id"):
            by_tid.setdefault(t["recording_id"], t)
    mix_path = str(set_dir / "mix.m4a")

    rows = [
        r
        for r in yaml.safe_load(args.gt.read_text())["tracks"]
        if r.get("track_id")
        and (r.get("claimed_stem") or "regular") in want
        and r.get("ref_source") != "online_candidate"
        and not r.get("is_loop")
        and not r.get("ref_segments")  # single-line spans: one GT ref_start
    ]
    jobs, meta, skipped = [], [], 0
    for i, r in enumerate(rows):
        tr = by_tid.get(str(r["track_id"])) or by_tid.get(r.get("recording_id"))
        if not tr:
            skipped += 1
            continue
        stem = r.get("claimed_stem") or "regular"
        sk = _STEM_FILE.get(stem)
        ref = (tr.get("stems") or {}).get(sk) if sk else tr.get("local_path")
        if not ref or not Path(ref).is_file():
            ref = tr.get("local_path")
        if not ref or not Path(ref).is_file():
            skipped += 1
            continue
        s0 = float(r["set_start_s"])
        span = min(args.max_win_s, float(r["set_end_s"]) - s0)
        if span < 4:
            skipped += 1
            continue
        jobs.append((i, mix_path, s0, span, str(ref), stretches))
        meta.append(r)

    print(f"fingerprinting {len(jobs)} single-line spans ({skipped} skipped)…")
    res: dict[int, dict] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for k, rr in enumerate(ex.map(_job, jobs, chunksize=1)):
            res[rr["idx"]] = rr
            if (k + 1) % 20 == 0:
                print(f"  {k + 1}/{len(jobs)}")

    by_stem: dict[str, list] = {}
    allrows = []
    for (i, *_), r in zip(jobs, meta):
        rr = res[i]
        err = abs(rr["ref_start"] - float(r["ref_start_s"]))
        stem = r.get("claimed_stem") or "regular"
        rec = (err, rr["votes"], stem, r.get("slot_label"), r.get("label") or "")
        by_stem.setdefault(stem, []).append(rec)
        allrows.append(rec)

    def rep(name, sel):
        if not sel:
            return
        e = np.array([x[0] for x in sel])
        v = np.array([x[1] for x in sel])
        print(
            f"  {name:14} n={len(sel):3}  exact<2s {100 * (e < 2).mean():3.0f}%  "
            f"<5s {100 * (e < 5).mean():3.0f}%  median votes {int(np.median(v))}"
        )

    print("\n=== fingerprint ref-offset vs GT (single-line spans) ===")
    rep("ALL", allrows)
    for stem in ("regular", "instrumental", "acappella"):
        rep(stem, by_stem.get(stem, []))
    # confidence: do votes separate hits from misses?
    hits = [x[1] for x in allrows if x[0] < 2]
    miss = [x[1] for x in allrows if x[0] >= 2]
    if hits and miss:
        print(
            f"votes: hits median {int(np.median(hits))}  vs  miss median "
            f"{int(np.median(miss))}  (a usable abstention signal if separated)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
