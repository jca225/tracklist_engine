#!/usr/bin/env python3
"""v1 — whole-mix HSMM decode over (track, ref-position) states.

The user's mental model made literal: a DJ set is a hidden Markov chain whose
state is (which song, where in the song). We decode the whole mix at once with
three transitions, exactly the three things a DJ does:

    advance  (k, j) -> (k, j+1)      continue playing the song   (free)
    jump     (k, j) -> (k, j')       loop / skip within the song (JUMP_PEN)
    switch   (k, j) -> (k', j')      change to another song      (SWITCH_PEN)

Emissions are chroma cosine (the feature v0.1 proved localizes), states are
restricted to the closed tracklist vocabulary (the GT track ids — this is why
it is tractable). Viterbi runs in O(T * S) via the per-track-max (jump) and
global-max (switch) entry trick, so it scales to the whole mix.

The point vs per-span argmax: a single repeated chorus is ambiguous in
isolation, but the best *path* through the whole mix breaks the tie — the
surrounding frames pick which chorus, and switches are penalized so the decode
prefers staying coherent. This targets the documented p90=100s repeat tail, not
the median.

Usage:
    venvs/audio/bin/python -m workspaces.section_hsmm.decode_hsmm \
        --set-id 1fsnxchk [--frame-s 2.0] [--switch-pen 0.5] [--jump-pen 0.3]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.alignment_prototype.refine_ref_offsets import (  # noqa: E402
    HOP, SR, chroma, find_aligning_dir,
)
from workspaces.section_hsmm.v0_1_chroma_scorecard import _CACHE, _l2cols  # noqa: E402

FPS = SR / HOP
NEG = -1e9


def _pooled_chroma(audio_path: Path, cache_key: str, frame_s: float) -> np.ndarray:
    """L2-normalized chroma pooled into frame_s-second columns -> (n, 12)."""
    cf = _CACHE / f"{cache_key}.npy"
    if cf.is_file():
        c = np.load(cf)
    else:
        import librosa
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
        c = chroma(y)
        _CACHE.mkdir(parents=True, exist_ok=True)
        np.save(cf, c)
    w = max(1, int(round(frame_s * FPS)))
    n = c.shape[1] // w
    if n == 0:
        return np.zeros((0, 12), dtype=np.float32)
    pooled = c[:, : n * w].reshape(12, n, w).mean(axis=2)
    return _l2cols(pooled).T.astype(np.float32)  # (n, 12)


@dataclass(frozen=True)
class Vocab:
    tids: list[str]                  # track id per vocab entry
    ref_frame: np.ndarray            # (S,) ref-frame index within its track
    track_of: np.ndarray             # (S,) vocab-track index
    slices: list[tuple[int, int]]    # (lo, hi) state range per track
    emit_ref: np.ndarray             # (S, 12) ref chroma stacked


def _build_vocab(set_id: str, frame_s: float) -> tuple[Vocab, dict]:
    set_dir = find_aligning_dir(set_id)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    gt_tids = _gt_track_ids(_REPO / "labeling/fixtures/bb12_ground_truth.yaml")
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    for t in manifest["tracks"]:
        if t.get("recording_id"):
            by_tid.setdefault(t["recording_id"], t)

    tids: list[str] = []
    refs: list[np.ndarray] = []
    track_of, ref_frame, slices = [], [], []
    cur = 0
    for tid in gt_tids:
        t = by_tid.get(tid)
        if not t:
            continue
        path = Path(t["local_path"])
        if not path.is_file():
            continue
        c = _pooled_chroma(path, f"ref_{tid}_pool{frame_s}", frame_s)
        if c.shape[0] < 2:
            continue
        k = len(tids)
        tids.append(tid)
        refs.append(c)
        slices.append((cur, cur + c.shape[0]))
        track_of.extend([k] * c.shape[0])
        ref_frame.extend(range(c.shape[0]))
        cur += c.shape[0]
    vocab = Vocab(
        tids=tids,
        ref_frame=np.array(ref_frame, dtype=np.int32),
        track_of=np.array(track_of, dtype=np.int32),
        slices=slices,
        emit_ref=np.concatenate(refs, axis=0).astype(np.float32),
    )
    return vocab, by_tid


def _gt_track_ids(gt_path: Path) -> list[str]:
    import yaml
    seen: dict[str, None] = {}
    for r in yaml.safe_load(gt_path.read_text())["tracks"]:
        if str(r.get("slot_label")) != "mix" and r.get("track_id"):
            seen.setdefault(str(r["track_id"]), None)
    return list(seen)


def viterbi(emis: np.ndarray, vocab: Vocab, *, jump_pen: float, switch_pen: float,
            hold_pen: float, skip_pen: float) -> np.ndarray:
    """emis: (T, S) chroma cosine. Returns decoded state index per mix frame."""
    T, S = emis.shape
    V = emis[0].copy()
    bp = np.empty((T, S), dtype=np.int32)
    bp[0] = -1
    is_track_start = np.zeros(S, dtype=bool)
    for lo, _hi in vocab.slices:
        is_track_start[lo] = True
    idx = np.arange(S)
    for t in range(1, T):
        e = emis[t]
        # advance (+1) within same track
        adv = np.full(S, NEG, dtype=np.float64)
        adv[1:] = V[:-1]
        adv[is_track_start] = NEG
        adv_src = idx - 1
        # hold (same j) — absorbs minor tempo
        hold = V - hold_pen
        # skip (+2) within same track
        skip = np.full(S, NEG, dtype=np.float64)
        skip[2:] = V[:-2] - skip_pen
        skip[is_track_start] = NEG
        start2 = np.zeros(S, dtype=bool)
        for lo, _hi in vocab.slices:
            if lo + 1 < S:
                start2[lo + 1] = True
        skip[start2] = NEG
        # local (advance/hold/skip) best
        local = np.stack([adv, hold, skip])
        local_src = np.stack([adv_src, idx, idx - 2])
        li = local.argmax(axis=0)
        local_best = np.take_along_axis(local, li[None], 0)[0]
        local_bp = np.take_along_axis(local_src, li[None], 0)[0]
        # jump within same track: best previous state of that track
        jump_best = np.full(S, NEG, dtype=np.float64)
        jump_bp = np.full(S, -1, dtype=np.int32)
        for lo, hi in vocab.slices:
            seg = V[lo:hi]
            a = int(seg.argmax())
            jump_best[lo:hi] = seg[a] - jump_pen
            jump_bp[lo:hi] = lo + a
        # switch to any track: global best previous state
        g = int(V.argmax())
        switch_val = V[g] - switch_pen
        # combine
        cand = np.stack([local_best, jump_best, np.full(S, switch_val)])
        cand_bp = np.stack([local_bp, jump_bp, np.full(S, g, dtype=np.int32)])
        ci = cand.argmax(axis=0)
        V = np.take_along_axis(cand, ci[None], 0)[0] + e
        bp[t] = np.take_along_axis(cand_bp, ci[None], 0)[0]
    # backtrace
    path = np.empty(T, dtype=np.int32)
    path[-1] = int(V.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set-id", default="1fsnxchk")
    p.add_argument("--frame-s", type=float, default=2.0)
    p.add_argument("--jump-pen", type=float, default=0.3)
    p.add_argument("--switch-pen", type=float, default=0.5)
    p.add_argument("--hold-pen", type=float, default=0.1)
    p.add_argument("--skip-pen", type=float, default=0.1)
    p.add_argument("--max-mix-s", type=float, default=0.0, help="0 = full mix")
    args = p.parse_args(argv)

    set_dir = find_aligning_dir(args.set_id)
    print("building closed-vocab ref states …", file=sys.stderr)
    vocab, _ = _build_vocab(args.set_id, args.frame_s)
    print(f"  {len(vocab.tids)} tracks, {vocab.emit_ref.shape[0]} states", file=sys.stderr)

    mix = _pooled_chroma(set_dir / "mix.m4a", f"{args.set_id}_mixpool{args.frame_s}", args.frame_s)
    if args.max_mix_s > 0:
        mix = mix[: int(args.max_mix_s / args.frame_s)]
    print(f"mix frames: {mix.shape[0]} ({mix.shape[0]*args.frame_s:.0f}s)", file=sys.stderr)

    emis = (mix @ vocab.emit_ref.T).astype(np.float64)  # (T, S) cosine
    path = viterbi(emis, vocab, jump_pen=args.jump_pen, switch_pen=args.switch_pen,
                   hold_pen=args.hold_pen, skip_pen=args.skip_pen)

    dec_track = vocab.track_of[path]
    dec_tid = [vocab.tids[k] for k in dec_track]
    _evaluate(args, dec_tid, mix.shape[0])
    return 0


def _evaluate(args, dec_tid: list[str], T: int) -> None:
    import yaml
    gt_rows = [r for r in yaml.safe_load(
        (_REPO / "labeling/fixtures/bb12_ground_truth.yaml").read_text())["tracks"]
        if str(r.get("slot_label")) != "mix" and r.get("track_id")]

    def active(t_s: float) -> set[str]:
        return {str(r["track_id"]) for r in gt_rows
                if float(r["set_start_s"]) - 2 <= t_s <= float(r["set_end_s"]) + 2}

    hit = cov = 0
    for i in range(T):
        ts = (i + 0.5) * args.frame_s
        act = active(ts)
        if not act:
            continue
        cov += 1
        if dec_tid[i] in act:
            hit += 1
    # decoded switch count vs GT span count
    switches = sum(1 for i in range(1, T) if dec_tid[i] != dec_tid[i - 1])
    print(f"=== v1 whole-mix HSMM decode ({args.set_id}) ===")
    print(f"frame={args.frame_s}s  jump={args.jump_pen} switch={args.switch_pen} "
          f"hold={args.hold_pen} skip={args.skip_pen}")
    print(f"identity-over-time: {hit}/{cov} = {100*hit/max(cov,1):.0f}%  "
          f"(decoded track in GT-active set; {T-cov} frames had no GT)")
    print(f"decoded track switches: {switches}  (GT spans: {len(gt_rows)})")


if __name__ == "__main__":
    sys.exit(main())
