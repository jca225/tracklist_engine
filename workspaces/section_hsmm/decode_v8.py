#!/usr/bin/env python3
"""v8 — vocal-specific NULL-default Viterbi (1s frames, MFCC, event-scored).

The vocal-event taxonomy showed the acappella channel inverts the bed: silence
is the resting state (51/99 acap->acap transitions go through none) and stabs are
sub-5s, so 2s frames smear them. v8 = the v7 MFCC channel at 1s frames with
NULL-default dynamics (cheap to enter silence) and optional 2-layer peeling for
concurrent vocals. Scored at EVENT level (did we catch each vocal burst with the
right identity + onset timing), the right metric for an intermittent channel.

Usage:
    venvs/audio/bin/python -m workspaces.section_hsmm.decode_v8 --set-id 1fsnxchk \
        [--frame-s 1.0] [--beta -0.08] [--layers 2]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.alignment_prototype.refine_ref_offsets import find_aligning_dir  # noqa: E402
from workspaces.section_hsmm.decode_hsmm import NEG  # noqa: E402
from workspaces.section_hsmm.decode_null import PENS, viterbi_null  # noqa: E402
from workspaces.section_hsmm.decode_v7 import build_vocal_vocab  # noqa: E402
from workspaces.section_hsmm.mfcc_emit import pooled_mfcc  # noqa: E402


def spans_from_path(path, vocab, frame_s, min_frames=2):
    """Contiguous non-NULL same-track runs -> (tid, start_s, end_s)."""
    S = vocab.emit_ref.shape[0]
    out, T, i = [], len(path), 0
    while i < T:
        if int(path[i]) == S:
            i += 1; continue
        k = int(vocab.track_of[path[i]]); j = i
        while j < T and int(path[j]) != S and int(vocab.track_of[path[j]]) == k:
            j += 1
        if j - i >= min_frames:
            out.append((vocab.tids[k], i * frame_s, j * frame_s))
        i = j
    return out


def peel(emis, vocab, K, beta, null_enter_pen, frame_s):
    S = vocab.emit_ref.shape[0]
    masked = emis.copy()
    spans = []
    for _ in range(K):
        path = viterbi_null(masked, vocab, beta=beta,
                            null_enter_pen=null_enter_pen, **PENS)
        spans.extend(spans_from_path(path, vocab, frame_s))
        for t, st in enumerate(path):
            if int(st) != S:
                lo, hi = vocab.slices[int(vocab.track_of[st])]
                masked[t, lo:hi] = NEG
    return spans


def event_score(spans, gt_rows, tol=3.0):
    ev = [(str(r["track_id"]), float(r["set_start_s"]), float(r["set_end_s"]))
          for r in gt_rows if (r.get("claimed_stem") or "regular") == "acappella"]

    def ov(a0, a1, b0, b1):
        return a0 < b1 + tol and a1 > b0 - tol

    caught, onsets = 0, []
    for tid, s0, s1 in ev:
        hits = [(sp0) for ptid, sp0, sp1 in spans if ptid == tid and ov(sp0, sp1, s0, s1)]
        if hits:
            caught += 1
            onsets.append(min(abs(h - s0) for h in hits))
    recall = caught / max(len(ev), 1)
    good = sum(1 for ptid, sp0, sp1 in spans
               if any(ptid == tid and ov(sp0, sp1, s0, s1) for tid, s0, s1 in ev))
    prec = good / max(len(spans), 1)
    return recall, prec, len(ev), len(spans), onsets


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", default="1fsnxchk")
    p.add_argument("--frame-s", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=-0.08)
    p.add_argument("--layers", type=int, default=2)
    args = p.parse_args(argv)

    set_dir = find_aligning_dir(args.set_id)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    for t in manifest["tracks"]:
        if t.get("recording_id"):
            by_tid.setdefault(t["recording_id"], t)
    import yaml
    gt_rows = [r for r in yaml.safe_load(
        (_REPO / "labeling/fixtures/bb12_ground_truth.yaml").read_text())["tracks"]
        if str(r.get("slot_label")) != "mix" and r.get("track_id")]

    print(f"building vocal vocab @ {args.frame_s}s frames …", file=sys.stderr)
    vocab = build_vocal_vocab(by_tid, args.frame_s)
    mix = pooled_mfcc(set_dir / "mix_vocals.flac", f"{args.set_id}_mix_vocals",
                      f"{args.set_id}_mix_vocals_pool{args.frame_s}", args.frame_s)
    print(f"  {len(vocab.tids)} tracks, {vocab.emit_ref.shape[0]} states, "
          f"{mix.shape[0]} mix frames", file=sys.stderr)
    emis = (mix @ vocab.emit_ref.T).astype(np.float32)

    n_ev = sum(1 for r in gt_rows if (r.get("claimed_stem") or "regular") == "acappella")
    print(f"=== v8 vocal NULL-default Viterbi ({args.set_id}, {args.frame_s}s, "
          f"{n_ev} GT vocal events) ===")
    print(f"  {'mode':>10} {'nullPen':>8} {'evRecall':>9} {'evPrec':>7} {'onset_med':>10}")
    for K in (1, args.layers):
        for nep in (0.0, 0.02, 0.05):
            spans = peel(emis, vocab, K, args.beta, nep, args.frame_s)
            rec, prec, nev, nsp, onsets = event_score(spans, gt_rows)
            om = f"{np.median(onsets):.1f}s" if onsets else "-"
            print(f"  {('K=%d' % K):>10} {nep:8.2f} {100*rec:7.0f}% {100*prec:6.0f}% "
                  f"{om:>10}  ({nsp} spans)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
