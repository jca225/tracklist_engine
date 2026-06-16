#!/usr/bin/env python3
"""v7 — label-agnostic vocal-verification overlay channel (MFCC).

Replaces the overlay identity-decode with MFCC vocal matching (the pre-test
feature: 70% retrieval@1 vs chroma chance). Candidate vocab is EVERY set track
with a vocal stem — not the scraped 'acappella' tag (John: it's missing/messy) —
so the audio decides where a vocal appears. Runs the NULL-state abstaining
Viterbi over MFCC emissions and reports margin precision @ coverage, vs the v3
MERT overlay (23% no-abstain / 73% @ 22%).

Usage:
    venvs/audio/bin/python -m workspaces.section_hsmm.decode_v7 --set-id 1fsnxchk
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

from workspaces.alignment_prototype.refine_ref_offsets import find_aligning_dir, ref_audio_for  # noqa: E402
from workspaces.section_hsmm.decode_hsmm import Vocab  # noqa: E402
from workspaces.section_hsmm.decode_null import PENS, viterbi_null  # noqa: E402
from workspaces.section_hsmm.mfcc_emit import pooled_mfcc  # noqa: E402


def build_vocal_vocab(by_tid: dict, frame_s: float) -> Vocab:
    """Label-agnostic: every set track with a usable vocal stem."""
    tids, refs, track_of, ref_frame, slices = [], [], [], [], []
    cur = 0
    seen = set()
    for tid, t in by_tid.items():
        if tid in seen:
            continue
        seen.add(tid)
        ap = ref_audio_for({"claimed_stem": "acappella"}, t)
        if ap is None:
            continue
        c = pooled_mfcc(ap, f"ref_{tid}_voc", f"ref_{tid}_voc_pool{frame_s}", frame_s)
        if c.shape[0] < 2:
            continue
        k = len(tids)
        tids.append(tid); refs.append(c)
        slices.append((cur, cur + c.shape[0]))
        track_of.extend([k] * c.shape[0]); ref_frame.extend(range(c.shape[0]))
        cur += c.shape[0]
    return Vocab(tids, np.array(ref_frame, np.int32), np.array(track_of, np.int32),
                 slices, np.concatenate(refs, axis=0).astype(np.float32))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", default="1fsnxchk")
    p.add_argument("--frame-s", type=float, default=2.0)
    p.add_argument("--null-enter-pen", type=float, default=0.05)
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

    print(f"building label-agnostic vocal vocab (all set tracks) …", file=sys.stderr)
    vocab = build_vocal_vocab(by_tid, args.frame_s)
    mix = pooled_mfcc(set_dir / "mix_vocals.flac", f"{args.set_id}_mix_vocals",
                      f"{args.set_id}_mix_vocals_pool{args.frame_s}", args.frame_s)
    print(f"  {len(vocab.tids)} candidate tracks, {vocab.emit_ref.shape[0]} states, "
          f"{mix.shape[0]} mix frames", file=sys.stderr)
    emis = (mix @ vocab.emit_ref.T).astype(np.float64)

    # eval vs acappella GT (label-agnostic vocab, but GT acappella rows are truth)
    ac = [r for r in gt_rows if (r.get("claimed_stem") or "regular") == "acappella"]
    starts = np.array([float(r["set_start_s"]) for r in ac])
    ends = np.array([float(r["set_end_s"]) for r in ac])
    tids_gt = np.array([str(r["track_id"]) for r in ac])
    S = vocab.emit_ref.shape[0]

    print(f"=== v7 MFCC vocal-verification overlay ({args.set_id}) ===")
    print(f"  {'beta':>6} {'coverage':>9} {'precision':>10}")
    for beta in (-0.30, -0.20, -0.15, -0.10, -0.05, 0.0):
        path = viterbi_null(emis, vocab, beta=beta,
                            null_enter_pen=args.null_enter_pen, **PENS)
        pred = corr = den = 0
        for i, st in enumerate(path):
            ts = (i + 0.5) * args.frame_s
            A = set(tids_gt[(starts - 2 <= ts) & (ts <= ends + 2)])
            if not A:
                continue
            den += 1
            if int(st) == S:
                continue
            pred += 1
            if vocab.tids[int(vocab.track_of[st])] in A:
                corr += 1
        cov = pred / max(den, 1)
        prec = corr / max(pred, 1)
        ps = f"{100*prec:8.0f}%" if pred else "       -"
        print(f"  {beta:6.2f} {100*cov:7.0f}%  {ps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
