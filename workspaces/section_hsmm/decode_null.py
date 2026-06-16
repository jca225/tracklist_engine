#!/usr/bin/env python3
"""v4 — Viterbi with a real NULL (abstain) state the path routes through.

abstain_eval proved margin (decoded vs best-OTHER track) is the right confidence
signal and absolute cosine is useless. This bakes margin into the decode: a NULL
state whose per-frame emission is the runner-up track's max score + beta. A real
state only beats NULL when it beats the *second-best track* by margin beta — but
because NULL competes on accumulated path score (not just the frame), continuity
keeps a confident track alive through a brief dip instead of dropping out, which
post-hoc gating can't do. Sweeping beta traces precision @ coverage; compare to
the post-hoc curve from abstain_eval.

Usage:
    venvs/audio/bin/python -m workspaces.section_hsmm.decode_null --set-id 1fsnxchk
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
from workspaces.section_hsmm.decode_hsmm import NEG, Vocab, _gt_track_ids  # noqa: E402
from workspaces.section_hsmm.decode_v2 import CHANNELS, _pooled, build_channel_vocab  # noqa: E402
from workspaces.section_hsmm.decode_v3 import _assemble_vocab, build_overlay_vocab_mert  # noqa: E402
from workspaces.section_hsmm.mert_emit import ensure_mert_cache, load_pooled_mert  # noqa: E402

PENS = dict(fwd_pen=0.15, back_pen=0.30, switch_pen=0.50, hold_pen=0.10, skip_pen=0.10)


def viterbi_null(emis: np.ndarray, vocab: Vocab, *, beta: float, null_enter_pen: float,
                 fwd_pen: float, back_pen: float, switch_pen: float,
                 hold_pen: float, skip_pen: float) -> np.ndarray:
    """Returns decoded state per frame; state == S means ABSTAIN (NULL)."""
    T, S = emis.shape
    K = len(vocab.slices)
    # per-frame runner-up track score -> NULL emission
    track_max = np.full((T, K), NEG)
    for k, (lo, hi) in enumerate(vocab.slices):
        track_max[:, k] = emis[:, lo:hi].max(axis=1)
    part = np.partition(track_max, K - 2, axis=1) if K >= 2 else track_max
    top2 = part[:, -2] if K >= 2 else np.full(T, NEG)
    null_emit = top2 + beta

    V = np.empty(S + 1)
    V[:S] = emis[0]
    V[S] = null_emit[0]
    bp = np.empty((T, S + 1), dtype=np.int32)
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
        Vr = V[:S]
        adv = np.full(S, NEG); adv[1:] = Vr[:-1]; adv[is_start] = NEG
        hold = Vr - hold_pen
        skip = np.full(S, NEG); skip[2:] = Vr[:-2] - skip_pen
        skip[is_start] = NEG; skip[start2] = NEG
        local = np.stack([adv, hold, skip])
        local_src = np.stack([idx - 1, idx, idx - 2])
        li = local.argmax(0)
        local_best = np.take_along_axis(local, li[None], 0)[0]
        local_bp = np.take_along_axis(local_src, li[None], 0)[0]
        jump_best = np.full(S, NEG); jump_bp = np.full(S, -1, np.int32)
        for lo, hi in vocab.slices:
            seg = Vr[lo:hi]
            a = int(seg.argmax())
            pen = np.where(np.arange(hi - lo) >= a, fwd_pen, back_pen)
            jump_best[lo:hi] = seg[a] - pen
            jump_bp[lo:hi] = lo + a
        g = int(V.argmax())                       # global best incl NULL (leave abstain)
        switch = np.full(S, V[g] - switch_pen)
        cand = np.stack([local_best, jump_best, switch])
        cand_bp = np.stack([local_bp, jump_bp, np.full(S, g, np.int32)])
        ci = cand.argmax(0)
        real_new = np.take_along_axis(cand, ci[None], 0)[0] + e
        real_bp = np.take_along_axis(cand_bp, ci[None], 0)[0]
        # NULL: self-hold (free) or enter from best real (penalty)
        gr = int(Vr.argmax())
        null_self, null_from = V[S], Vr[gr] - null_enter_pen
        if null_self >= null_from:
            null_new, null_bp = null_self + null_emit[t], S
        else:
            null_new, null_bp = null_from + null_emit[t], gr
        V = np.concatenate([real_new, [null_new]])
        bp[t, :S] = real_bp
        bp[t, S] = null_bp
    path = np.empty(T, dtype=np.int32)
    path[-1] = int(V.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return path


def _channel_emis(channel, set_dir, by_tid, gt_tids, frame_s, layer):
    if channel == "bed":
        vocab = build_channel_vocab("bed", by_tid, gt_tids, frame_s)
        cfg = CHANNELS["bed"]
        mix = _pooled(set_dir / cfg["mix_file"], f"1fsnxchk_{cfg['mix_key']}",
                      f"1fsnxchk_{cfg['mix_key']}_pool{frame_s}", frame_s)
    else:
        key_of, items = build_overlay_vocab_mert(by_tid, gt_tids, frame_s, layer)
        ensure_mert_cache(items + [(set_dir / CHANNELS["overlay"]["mix_file"],
                                    "1fsnxchk_mix_vocals")], frame_s, layer)
        vocab = _assemble_vocab(key_of, frame_s, layer)
        mix = load_pooled_mert("1fsnxchk_mix_vocals", frame_s, layer)
    return vocab, (mix @ vocab.emit_ref.T).astype(np.float64)


def _eval(channel, vocab, path, frame_s, gt_rows):
    S = vocab.emit_ref.shape[0]
    stems = CHANNELS[channel]["stems"]
    rows = [r for r in gt_rows if (r.get("claimed_stem") or "regular") in stems]

    def active(ts):
        return {str(r["track_id"]) for r in rows
                if float(r["set_start_s"]) - 2 <= ts <= float(r["set_end_s"]) + 2}

    pred = corr = cov_den = 0
    for i, st in enumerate(path):
        if not active((i + 0.5) * frame_s):
            continue
        cov_den += 1
        if st == S:           # abstained
            continue
        pred += 1
        if vocab.tids[vocab.track_of[st]] in active((i + 0.5) * frame_s):
            corr += 1
    cov = pred / max(cov_den, 1)
    prec = corr / max(pred, 1)
    return cov, prec, cov_den


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", default="1fsnxchk")
    p.add_argument("--frame-s", type=float, default=2.0)
    p.add_argument("--mert-layer", type=int, default=6)
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
    stem_by_tid = {}
    for r in gt_rows:
        stem_by_tid.setdefault(str(r["track_id"]), r.get("claimed_stem") or "regular")
    for tid, t in by_tid.items():
        t["_stem"] = stem_by_tid.get(tid, "regular")
    gt_tids = _gt_track_ids(_REPO / "labeling/fixtures/bb12_ground_truth.yaml")

    print(f"=== v4 NULL-state Viterbi (route-through abstain) ({args.set_id}) ===")
    for channel in ("bed", "overlay"):
        vocab, emis = _channel_emis(channel, set_dir, by_tid, gt_tids,
                                    args.frame_s, args.mert_layer)
        print(f"\n[{channel}]  {'beta':>6} {'coverage':>9} {'precision':>10}")
        for beta in (-0.12, -0.10, -0.08, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01):
            path = viterbi_null(emis, vocab, beta=beta,
                                null_enter_pen=args.null_enter_pen, **PENS)
            cov, prec, _ = _eval(channel, vocab, path, args.frame_s, gt_rows)
            ps = f"{100*prec:8.0f}%" if cov > 0 else "       -"
            print(f"        {beta:6.2f} {100*cov:7.0f}%  {ps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
