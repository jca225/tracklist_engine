#!/usr/bin/env python3
"""VAD-gated vocal decode: force abstain where vocal activity is low.

Wires the vocal-activity signal (vad_probe.py) into the vocal NULL-default
Viterbi as an "informed NULL": at decode frames whose energy/voiced activity is
below a threshold, all real states are suppressed so the path MUST abstain —
killing the sticky-distractor false positives that the matcher hallucinates onto
separation artifacts. Sweeps the gate percentile and reports recall / precision /
false-positive count vs the ungated baseline.

Usage:
    venvs/audio/bin/python -m workspaces.section_hsmm.vad_gate --set-id 1fsnxchk
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

from workspaces.alignment_prototype.refine_ref_offsets import SR, find_aligning_dir  # noqa: E402
from workspaces.section_hsmm.decode_hsmm import NEG  # noqa: E402
from workspaces.section_hsmm.decode_null import PENS, viterbi_null  # noqa: E402
from workspaces.section_hsmm.decode_v7 import build_vocal_vocab  # noqa: E402
from workspaces.section_hsmm.decode_v8 import event_score, spans_from_path  # noqa: E402
from workspaces.section_hsmm.decode_v9 import diag_smooth_multislope  # noqa: E402
from workspaces.section_hsmm.mfcc_emit import pooled_mfcc  # noqa: E402
from workspaces.section_hsmm.v0_1_chroma_scorecard import _CACHE  # noqa: E402

VAD_HOP = 1024
SLOPES = (0.7, 0.85, 1.0, 1.18, 1.4)


def _norm(x):
    lo, hi = np.percentile(x, 5), np.percentile(x, 95)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)


def _min_duration(mask, min_frames):
    """Remove gated runs shorter than min_frames — only SUSTAINED low-activity
    (real instrumental sections) gates; brief dips (breaths/syllables) don't
    fragment a vocal span. This is John's 'long section vs phrase break' rule."""
    out = mask.copy()
    i, n = 0, len(mask)
    while i < n:
        if out[i]:
            j = i
            while j < n and out[j]:
                j += 1
            if j - i < min_frames:
                out[i:j] = False
            i = j
        else:
            i += 1
    return out


def _activity_per_frame(set_id, n_frames, frame_s):
    z = np.load(_CACHE / f"{set_id}_vad_feats.npz")
    rms, vprob = z["rms"], z["vprob"]
    act = 0.5 * _norm(rms) + 0.5 * _norm(vprob)
    fps = SR / VAD_HOP
    out = np.zeros(n_frames)
    for i in range(n_frames):
        a, b = int(i * frame_s * fps), int((i + 1) * frame_s * fps)
        out[i] = act[a:b].mean() if b > a and a < len(act) else 0.0
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", default="1fsnxchk")
    p.add_argument("--frame-s", type=float, default=1.0)
    p.add_argument("--win-frames", type=int, default=6)
    p.add_argument("--beta", type=float, default=-0.05)
    args = p.parse_args(argv)

    set_dir = find_aligning_dir(args.set_id)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    for t in manifest["tracks"]:
        if t.get("recording_id"):
            by_tid.setdefault(t["recording_id"], t)
    import yaml
    gt = [r for r in yaml.safe_load(
        (_REPO / "labeling/fixtures/bb12_ground_truth.yaml").read_text())["tracks"]
        if str(r.get("slot_label")) != "mix" and r.get("track_id")]

    vocab = build_vocal_vocab(by_tid, args.frame_s)
    mix = pooled_mfcc(set_dir / "mix_vocals.flac", f"{args.set_id}_mix_vocals",
                      f"{args.set_id}_mix_vocals_pool{args.frame_s}", args.frame_s)
    emis0 = diag_smooth_multislope((mix @ vocab.emit_ref.T).astype(np.float32),
                                   vocab.slices, args.win_frames, SLOPES)
    T = emis0.shape[0]
    act = _activity_per_frame(args.set_id, T, args.frame_s)

    ac = [r for r in gt if (r.get("claimed_stem") or "regular") == "acappella"]

    def fp_count(spans):
        def ov(sp):
            return any(sp[0] == str(r["track_id"]) and sp[1] < float(r["set_end_s"]) + 3
                       and sp[2] > float(r["set_start_s"]) - 3 for r in ac)
        return sum(1 for sp in spans if not ov(sp))

    # decode ONCE (ungated), then drop spans whose mean activity is low
    path = viterbi_null(emis0, vocab, beta=args.beta, null_enter_pen=0.0, **PENS)
    base = spans_from_path(path, vocab, args.frame_s)  # (tid, s0, s1)

    def span_act(s0, s1):
        a, b = int(s0 / args.frame_s), int(s1 / args.frame_s)
        return act[a:b].mean() if b > a and a < T else 0.0

    sp_act = [span_act(s0, s1) for _, s0, s1 in base]
    print(f"=== VAD-gated vocal decode ({args.set_id}, {len(ac)} GT events) ===")
    print(f"  {'drop <%ile':>10} {'kept':>5} {'recall':>7} {'precision':>10} {'false pos':>10}")
    for q in (0, 20, 35, 50, 65):
        thr = np.percentile(sp_act, q) if q > 0 else -1
        kept = [s for s, a in zip(base, sp_act) if a >= thr]
        rec, prec, _, nsp, _ = event_score(kept, gt)
        print(f"  {q:9d}% {len(kept):5d} {100*rec:6.0f}% {100*prec:9.0f}% {fp_count(kept):6d}")
    print("\ndrop 0% = ungated baseline (89 spans). Higher %ile drops more "
          "low-activity spans. Want: false pos down faster than recall.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
