#!/usr/bin/env python3
"""v6 — multi-overlay: recover 2+ concurrent acappellas by peeling.

The single overlay channel names at most one acappella per frame, but the
taxonomy showed the mix is layered — multiple acappellas often stack. This
decodes K overlay layers by peeling: run the NULL-state abstaining Viterbi,
mask the chosen track at the frames it claimed, re-decode for the next layer.
Each layer abstains independently, so layer 2/3 only fire where a second/third
acappella genuinely competes.

Scored as per-frame SET prediction against the GT-active acappella set:
  recall    = |predicted ∩ active| / |active|   (did we recover the stack?)
  precision = |predicted ∩ active| / |predicted| (were our names right?)

Usage:
    venvs/audio/bin/python -m workspaces.section_hsmm.decode_multi \
        --set-id 1fsnxchk [--overlay-beta -0.05] [--max-layers 3]
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
from workspaces.section_hsmm.decode_hsmm import NEG, _gt_track_ids  # noqa: E402
from workspaces.section_hsmm.decode_null import PENS, _channel_emis, viterbi_null  # noqa: E402


def decode_layers(emis, vocab, K, beta, null_enter_pen):
    """Peeling: decode, mask chosen track at claimed frames, repeat."""
    S = vocab.emit_ref.shape[0]
    masked = emis.copy()
    layers = []
    for _ in range(K):
        path = viterbi_null(masked, vocab, beta=beta,
                            null_enter_pen=null_enter_pen, **PENS)
        layers.append(path)
        for t, st in enumerate(path):
            if int(st) != S:
                lo, hi = vocab.slices[int(vocab.track_of[st])]
                masked[t, lo:hi] = NEG     # forbid this track here on the next pass
    return layers


def evaluate(layers, vocab, frame_s, gt_rows):
    S = vocab.emit_ref.shape[0]
    rows = [r for r in gt_rows if (r.get("claimed_stem") or "regular") == "acappella"]
    starts = np.array([float(r["set_start_s"]) for r in rows])
    ends = np.array([float(r["set_end_s"]) for r in rows])
    tids = np.array([str(r["track_id"]) for r in rows])
    T = len(layers[0])
    rec_n = rec_d = pre_n = pre_d = 0
    conc = []
    for t in range(T):
        ts = (t + 0.5) * frame_s
        m = (starts - 2 <= ts) & (ts <= ends + 2)
        A = set(tids[m])
        if not A:
            continue
        conc.append(len(A))
        P = {vocab.tids[int(vocab.track_of[p[t]])] for p in layers if int(p[t]) != S}
        inter = len(P & A)
        rec_n += inter; rec_d += len(A)
        pre_n += inter; pre_d += len(P)
    recall = rec_n / max(rec_d, 1)
    prec = pre_n / max(pre_d, 1)
    return recall, prec, np.array(conc)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", default="1fsnxchk")
    p.add_argument("--frame-s", type=float, default=2.0)
    p.add_argument("--mert-layer", type=int, default=6)
    p.add_argument("--overlay-beta", type=float, default=-0.05)
    p.add_argument("--null-enter-pen", type=float, default=0.05)
    p.add_argument("--max-layers", type=int, default=3)
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

    vocab, emis = _channel_emis("overlay", set_dir, by_tid, gt_tids,
                                args.frame_s, args.mert_layer)
    layers = decode_layers(emis, vocab, args.max_layers, args.overlay_beta,
                           args.null_enter_pen)

    print(f"=== v6 multi-overlay (peeling) ({args.set_id}, beta={args.overlay_beta}) ===")
    _, _, conc = evaluate(layers[:1], vocab, args.frame_s, gt_rows)
    if conc.size:
        hist = {k: int((conc == k).sum()) for k in sorted(set(conc.tolist()))}
        print(f"GT concurrent-acappella frames: mean={conc.mean():.2f} max={conc.max()}  "
              f"dist={hist}")
    print(f"{'layers':>7} {'recall':>7} {'precision':>10}")
    for k in range(1, args.max_layers + 1):
        recall, prec, _ = evaluate(layers[:k], vocab, args.frame_s, gt_rows)
        print(f"{k:7} {100*recall:6.0f}%  {100*prec:8.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
