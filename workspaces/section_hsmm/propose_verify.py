#!/usr/bin/env python3
"""Propose-then-verify: loose decode proposes regions, strong matched filter
re-IDs and gates them.

The diagnostic showed the blind decode (60%) is near its verifiable ceiling
(66%), with ~12 verifiable-but-missed events recoverable. This loosens the
decode to PROPOSE spans generously (high recall), then RE-VERIFIES each with the
strong stretch-search matched filter (the given-region primitive) against the
decoded track + top-K alternatives — reassigning identity to the best match and
dropping spans no candidate confirms. Decode proposes, verifier disposes.

Usage:
    venvs/audio/bin/python -m workspaces.section_hsmm.propose_verify --set-id 1fsnxchk
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

from workspaces.alignment_prototype.refine_ref_offsets import (  # noqa: E402
    STRETCHES, detect_offset, find_aligning_dir, ref_audio_for,
)
from workspaces.section_hsmm.decode_null import PENS, viterbi_null  # noqa: E402
from workspaces.section_hsmm.decode_v7 import build_vocal_vocab  # noqa: E402
from workspaces.section_hsmm.decode_v8 import event_score  # noqa: E402
from workspaces.section_hsmm.decode_v9 import diag_smooth_multislope  # noqa: E402
from workspaces.section_hsmm.mfcc_emit import pooled_mfcc  # noqa: E402
from workspaces.section_hsmm.similarity_probe import FPS, _feat  # noqa: E402

SLOPES = (0.7, 0.85, 1.0, 1.18, 1.4)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", default="1fsnxchk")
    p.add_argument("--frame-s", type=float, default=1.0)
    p.add_argument("--win-frames", type=int, default=6)
    p.add_argument("--propose-beta", type=float, default=-0.15)
    p.add_argument("--topk", type=int, default=15)
    p.add_argument("--max-win-s", type=float, default=15.0)
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

    vocab = build_vocal_vocab(by_tid, args.frame_s)
    mixp = pooled_mfcc(set_dir / "mix_vocals.flac", f"{args.set_id}_mix_vocals",
                       f"{args.set_id}_mix_vocals_pool{args.frame_s}", args.frame_s)
    emis = diag_smooth_multislope((mixp @ vocab.emit_ref.T).astype(np.float32),
                                  vocab.slices, args.win_frames, SLOPES)
    S, T = vocab.emit_ref.shape[0], emis.shape[0]
    K = len(vocab.slices)
    track_max = np.full((T, K), -1e9, np.float32)
    for k, (lo, hi) in enumerate(vocab.slices):
        track_max[:, k] = emis[:, lo:hi].max(axis=1)

    # propose: loose decode -> frame-indexed spans
    path = viterbi_null(emis, vocab, beta=args.propose_beta, null_enter_pen=0.0, **PENS)
    proposals, i = [], 0
    while i < T:
        if int(path[i]) == S:
            i += 1; continue
        j = i
        while j < T and int(path[j]) != S and int(vocab.track_of[path[j]]) == int(vocab.track_of[path[i]]):
            j += 1
        if j - i >= 2:
            proposals.append((int(vocab.track_of[path[i]]), i, j))
        i = j
    print(f"proposed {len(proposals)} spans (beta={args.propose_beta}) — verifying …",
          file=sys.stderr)

    # verify each proposal with the strong stretch-search matched filter
    mixf = _feat(set_dir / "mix_vocals.flac", f"{args.set_id}_mix_vocals", "mfcc")
    ref_cache: dict[str, np.ndarray] = {}

    def ref_feat(tid):
        if tid not in ref_cache:
            ref_cache[tid] = _feat(ref_audio_for({"claimed_stem": "acappella"}, by_tid[tid]),
                                   f"ref_{tid}_voc", "mfcc")
        return ref_cache[tid]

    for tau in (0.0, 0.55, 0.60, 0.65):
        verified = []
        for kdec, i0, i1 in proposals:
            s0, s1 = i0 * args.frame_s, i1 * args.frame_s
            a = int(s0 * FPS); n = int(min(s1 - s0, args.max_win_s) * FPS)
            a = min(a, max(0, mixf.shape[1] - n))
            win = mixf[:, a:a + n]
            if win.shape[1] < 8:
                continue
            cand_idx = list({kdec, *np.argsort(track_max[i0:i1].mean(0))[::-1][:args.topk]})
            best_tid, best_peak = None, -1.0
            for k in cand_idx:
                rf = ref_feat(vocab.tids[k])
                if rf.shape[1] > win.shape[1]:
                    pk = detect_offset(win, rf, STRETCHES)[1]
                    if pk > best_peak:
                        best_peak, best_tid = pk, vocab.tids[k]
            if best_tid is not None and best_peak >= tau:
                verified.append((best_tid, s0, s1))
        rec, prec, nev, nsp, onsets = event_score(verified, gt_rows)
        om = f"{np.median(onsets):.1f}s" if onsets else "-"
        print(f"  tau={tau:.2f}: recall {100*rec:.0f}% precision {100*prec:.0f}% "
              f"onset {om}  ({nsp} verified spans of {len(proposals)} proposed)")
    print("  (v9 baseline: recall 55% / precision 60%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
