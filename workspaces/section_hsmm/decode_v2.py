#!/usr/bin/env python3
"""v2 — two-channel HSMM decode with data-derived transitions.

The BB12 taxonomy (gt_taxonomy.py) said v1 was wrong in two ways: the mix is
2-3 layers deep (one state per frame can't represent it) and the within-song
section-jump is HALF of all plays (so it must be cheap, not a penalty). v2 fixes
both:

  * two channels, decoded independently then read as layers:
      bed     = mix_instrumental  vs  instrumental/regular refs (harmonic bed)
      overlay = mix_vocals        vs  acappella refs            (vocal payload)
  * transitions from the data:
      advance (+1)            free        — linear play
      forward section-jump    fwd_pen     — cheap; the "make it shorter" op
      backward loop-jump      back_pen    — cheap; loop a phrase
      switch song             switch_pen  — change track (overlap allowed by the
                                            two-channel split: bed + overlay
                                            run concurrently)

Single overlay per frame for now (27% of mix has 2+ acappellas at once —
multi-overlay is v3). Emissions are chroma cosine. Evaluated PER CHANNEL against
the GT rows of that channel's stems.

Usage:
    venvs/audio/bin/python -m workspaces.section_hsmm.decode_v2 \
        --set-id 1fsnxchk [--frame-s 2.0] [--fwd-pen 0.15] [--back-pen 0.3] \
        [--switch-pen 0.5]
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
    HOP, SR, chroma, find_aligning_dir,
)
from workspaces.section_hsmm.decode_hsmm import FPS, NEG, Vocab, _gt_track_ids  # noqa: E402
from workspaces.section_hsmm.v0_1_chroma_scorecard import _CACHE, _l2cols  # noqa: E402

CHANNELS = {
    "bed": {"mix_file": "mix_instrumental.flac", "mix_key": "mix_instrumental",
            "stems": ("regular", "instrumental"), "ref_stem": "instrumental"},
    "overlay": {"mix_file": "mix_vocals.flac", "mix_key": "mix_vocals",
                "stems": ("acappella",), "ref_stem": "vocals"},
}


def _pooled(audio_path: Path | None, frame_key: str, pool_key: str,
            frame_s: float) -> np.ndarray:
    """Pooled L2-normed chroma (n,12). Reuses a frame-level cache (from v0.1)
    when present so we never re-decode the 771MB mix_instrumental."""
    _CACHE.mkdir(parents=True, exist_ok=True)
    pf = _CACHE / f"{pool_key}.npy"
    if pf.is_file():
        return np.load(pf)
    ff = _CACHE / f"{frame_key}.npy"
    if ff.is_file():
        c = np.load(ff)
    else:
        if audio_path is None or not audio_path.is_file():
            return np.zeros((0, 12), dtype=np.float32)
        import librosa
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
        c = chroma(y)
        np.save(ff, c)
    w = max(1, int(round(frame_s * FPS)))
    n = c.shape[1] // w
    if n == 0:
        return np.zeros((0, 12), dtype=np.float32)
    pooled = c[:, : n * w].reshape(12, n, w).mean(axis=2)
    out = _l2cols(pooled).T.astype(np.float32)
    np.save(pf, out)
    return out


def _ref_audio(track: dict, ref_stem: str) -> Path | None:
    p = (track.get("stems") or {}).get(ref_stem)
    if p and Path(p).is_file():
        return Path(p)
    p = track.get("local_path")
    return Path(p) if p and Path(p).is_file() else None


def build_channel_vocab(channel: str, by_tid: dict, gt_tids: list[str],
                        frame_s: float) -> Vocab:
    cfg = CHANNELS[channel]
    # which GT track ids belong to this channel (by their claimed_stem)
    chan_tids = [t for t in gt_tids
                 if (by_tid.get(t, {}).get("_stem") in cfg["stems"])]
    tids, refs, track_of, ref_frame, slices = [], [], [], [], []
    cur = 0
    for tid in chan_tids:
        t = by_tid.get(tid)
        ap = _ref_audio(t, cfg["ref_stem"]) if t else None
        c = _pooled(ap, f"ref_{tid}_{channel}", f"ref_{tid}_{channel}_pool{frame_s}", frame_s)
        if c.shape[0] < 2:
            continue
        k = len(tids)
        tids.append(tid)
        refs.append(c)
        slices.append((cur, cur + c.shape[0]))
        track_of.extend([k] * c.shape[0])
        ref_frame.extend(range(c.shape[0]))
        cur += c.shape[0]
    if not refs:
        return Vocab([], np.array([]), np.array([]), [], np.zeros((0, 12), np.float32))
    return Vocab(tids, np.array(ref_frame, np.int32), np.array(track_of, np.int32),
                 slices, np.concatenate(refs, axis=0).astype(np.float32))


def viterbi_v2(emis: np.ndarray, vocab: Vocab, *, fwd_pen: float, back_pen: float,
               switch_pen: float, hold_pen: float, skip_pen: float) -> np.ndarray:
    """Like v1 but the same-track jump is direction-split: landing ref-forward
    of the prior best state costs fwd_pen (the shorten op), backward costs
    back_pen (loop). fwd_pen < back_pen encodes Two Friends' forward bias."""
    T, S = emis.shape
    V = emis[0].copy()
    bp = np.empty((T, S), dtype=np.int32)
    bp[0] = -1
    is_start = np.zeros(S, dtype=bool)
    start2 = np.zeros(S, dtype=bool)
    for lo, _hi in vocab.slices:
        is_start[lo] = True
        if lo + 1 < S:
            start2[lo + 1] = True
    idx = np.arange(S)
    for t in range(1, T):
        e = emis[t]
        adv = np.full(S, NEG); adv[1:] = V[:-1]; adv[is_start] = NEG
        hold = V - hold_pen
        skip = np.full(S, NEG); skip[2:] = V[:-2] - skip_pen
        skip[is_start] = NEG; skip[start2] = NEG
        local = np.stack([adv, hold, skip])
        local_src = np.stack([idx - 1, idx, idx - 2])
        li = local.argmax(0)
        local_best = np.take_along_axis(local, li[None], 0)[0]
        local_bp = np.take_along_axis(local_src, li[None], 0)[0]
        # direction-split same-track jump from the track's best prior state
        jump_best = np.full(S, NEG); jump_bp = np.full(S, -1, np.int32)
        for lo, hi in vocab.slices:
            seg = V[lo:hi]
            a = int(seg.argmax())
            li_loc = np.arange(hi - lo)
            pen = np.where(li_loc >= a, fwd_pen, back_pen)
            jump_best[lo:hi] = seg[a] - pen
            jump_bp[lo:hi] = lo + a
        # switch to any other track
        g = int(V.argmax())
        switch = np.full(S, V[g] - switch_pen)
        cand = np.stack([local_best, jump_best, switch])
        cand_bp = np.stack([local_bp, jump_bp, np.full(S, g, np.int32)])
        ci = cand.argmax(0)
        V = np.take_along_axis(cand, ci[None], 0)[0] + e
        bp[t] = np.take_along_axis(cand_bp, ci[None], 0)[0]
    path = np.empty(T, dtype=np.int32)
    path[-1] = int(V.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return path


def _eval_channel(channel: str, dec_tid: list[str], frame_s: float, gt_rows: list[dict]) -> None:
    stems = CHANNELS[channel]["stems"]
    rows = [r for r in gt_rows if (r.get("claimed_stem") or "regular") in stems]

    def active(ts: float) -> set[str]:
        return {str(r["track_id"]) for r in rows
                if float(r["set_start_s"]) - 2 <= ts <= float(r["set_end_s"]) + 2}

    hit = cov = 0
    T = len(dec_tid)
    for i in range(T):
        ts = (i + 0.5) * frame_s
        act = active(ts)
        if not act:
            continue
        cov += 1
        if dec_tid[i] in act:
            hit += 1
    switches = sum(1 for i in range(1, T) if dec_tid[i] != dec_tid[i - 1])
    print(f"  [{channel:7}] identity-over-time: {hit}/{cov} = {100*hit/max(cov,1):.0f}%  "
          f"| switches: {switches}  | GT rows: {len(rows)}  | frames w/o GT: {T-cov}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set-id", default="1fsnxchk")
    p.add_argument("--frame-s", type=float, default=2.0)
    p.add_argument("--fwd-pen", type=float, default=0.15)
    p.add_argument("--back-pen", type=float, default=0.30)
    p.add_argument("--switch-pen", type=float, default=0.50)
    p.add_argument("--hold-pen", type=float, default=0.10)
    p.add_argument("--skip-pen", type=float, default=0.10)
    p.add_argument("--max-mix-s", type=float, default=0.0)
    args = p.parse_args(argv)

    set_dir = find_aligning_dir(args.set_id)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    for t in manifest["tracks"]:
        if t.get("recording_id"):
            by_tid.setdefault(t["recording_id"], t)
    # tag each track with its GT claimed_stem (majority) for channel routing
    import yaml
    gt_rows = [r for r in yaml.safe_load(
        (_REPO / "labeling/fixtures/bb12_ground_truth.yaml").read_text())["tracks"]
        if str(r.get("slot_label")) != "mix" and r.get("track_id")]
    stem_by_tid: dict[str, str] = {}
    for r in gt_rows:
        stem_by_tid.setdefault(str(r["track_id"]), r.get("claimed_stem") or "regular")
    for tid, t in by_tid.items():
        t["_stem"] = stem_by_tid.get(tid, "regular")
    gt_tids = _gt_track_ids(_REPO / "labeling/fixtures/bb12_ground_truth.yaml")

    print(f"=== v2 two-channel HSMM decode ({args.set_id}) ===")
    print(f"frame={args.frame_s}s  fwd={args.fwd_pen} back={args.back_pen} "
          f"switch={args.switch_pen}")
    for channel, cfg in CHANNELS.items():
        vocab = build_channel_vocab(channel, by_tid, gt_tids, args.frame_s)
        if not vocab.tids:
            print(f"  [{channel}] no vocab", file=sys.stderr)
            continue
        mix = _pooled(set_dir / cfg["mix_file"], f"{args.set_id}_{cfg['mix_key']}",
                      f"{args.set_id}_{cfg['mix_key']}_pool{args.frame_s}", args.frame_s)
        if args.max_mix_s > 0:
            mix = mix[: int(args.max_mix_s / args.frame_s)]
        print(f"  [{channel}] {len(vocab.tids)} tracks, {vocab.emit_ref.shape[0]} states, "
              f"{mix.shape[0]} mix frames", file=sys.stderr)
        emis = (mix @ vocab.emit_ref.T).astype(np.float64)
        path = viterbi_v2(emis, vocab, fwd_pen=args.fwd_pen, back_pen=args.back_pen,
                          switch_pen=args.switch_pen, hold_pen=args.hold_pen,
                          skip_pen=args.skip_pen)
        dec_tid = [vocab.tids[k] for k in vocab.track_of[path]]
        _eval_channel(channel, dec_tid, args.frame_s, gt_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
