#!/usr/bin/env python3
"""v3 — two-channel decode with MERT identity on the overlay channel.

v2 fixed the layering but the overlay (acappella) channel topped out at 25%
because vocal chroma is not discriminative for *which* acappella. v3 keeps
chroma on the bed (where it scored 79%) and swaps MERT — the identity feature —
into the overlay channel. Same Viterbi + data-derived transitions as v2.

Needs a one-time per-frame vocal MERT pass (Mac MPS, ~5-10 min) over the
overlay ref vocal stems + mix_vocals; cached thereafter.

Usage:
    venvs/audio/bin/python -m workspaces.section_hsmm.decode_v3 \
        --set-id 1fsnxchk [--frame-s 2.0] [--mert-layer 6] [--switch-pen 0.5]
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
from workspaces.section_hsmm.decode_hsmm import Vocab, _gt_track_ids  # noqa: E402
from workspaces.section_hsmm.decode_v2 import (  # noqa: E402
    CHANNELS, _eval_channel, _pooled, _ref_audio, build_channel_vocab, viterbi_v2,
)
from workspaces.section_hsmm.mert_emit import ensure_mert_cache, load_pooled_mert  # noqa: E402


def build_overlay_vocab_mert(by_tid: dict, gt_tids: list[str], frame_s: float,
                             layer: int) -> tuple[Vocab, list[tuple[Path, str]]]:
    cfg = CHANNELS["overlay"]
    chan_tids = [t for t in gt_tids if by_tid.get(t, {}).get("_stem") in cfg["stems"]]
    items: list[tuple[Path, str]] = []
    key_of: dict[str, str] = {}
    for tid in chan_tids:
        t = by_tid.get(tid)
        ap = _ref_audio(t, cfg["ref_stem"]) if t else None
        if ap is None:
            continue
        key = f"ref_{tid}_overlay"
        items.append((ap, key))
        key_of[tid] = key
    return key_of, items  # vocab built after cache ensured


def _assemble_vocab(key_of: dict[str, str], frame_s: float, layer: int) -> Vocab:
    tids, refs, track_of, ref_frame, slices = [], [], [], [], []
    cur = 0
    for tid, key in key_of.items():
        c = load_pooled_mert(key, frame_s, layer)
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
        return Vocab([], np.array([]), np.array([]), [], np.zeros((0, 1024), np.float32))
    return Vocab(tids, np.array(ref_frame, np.int32), np.array(track_of, np.int32),
                 slices, np.concatenate(refs, axis=0).astype(np.float32))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set-id", default="1fsnxchk")
    p.add_argument("--frame-s", type=float, default=2.0)
    p.add_argument("--mert-layer", type=int, default=6)
    p.add_argument("--fwd-pen", type=float, default=0.15)
    p.add_argument("--back-pen", type=float, default=0.30)
    p.add_argument("--switch-pen", type=float, default=0.50)
    p.add_argument("--hold-pen", type=float, default=0.10)
    p.add_argument("--skip-pen", type=float, default=0.10)
    p.add_argument("--max-mix-s", type=float, default=0.0)
    p.add_argument("--bed", action="store_true", help="also run the chroma bed channel")
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
    stem_by_tid: dict[str, str] = {}
    for r in gt_rows:
        stem_by_tid.setdefault(str(r["track_id"]), r.get("claimed_stem") or "regular")
    for tid, t in by_tid.items():
        t["_stem"] = stem_by_tid.get(tid, "regular")
    gt_tids = _gt_track_ids(_REPO / "labeling/fixtures/bb12_ground_truth.yaml")

    print(f"=== v3 overlay=MERT(L{args.mert_layer}) decode ({args.set_id}) ===")
    print(f"frame={args.frame_s}s  fwd={args.fwd_pen} back={args.back_pen} "
          f"switch={args.switch_pen}")

    # optional bed channel (chroma) — unchanged from v2, for side-by-side
    if args.bed:
        vocab = build_channel_vocab("bed", by_tid, gt_tids, args.frame_s)
        cfg = CHANNELS["bed"]
        mix = _pooled(set_dir / cfg["mix_file"], f"{args.set_id}_{cfg['mix_key']}",
                      f"{args.set_id}_{cfg['mix_key']}_pool{args.frame_s}", args.frame_s)
        if args.max_mix_s > 0:
            mix = mix[: int(args.max_mix_s / args.frame_s)]
        emis = (mix @ vocab.emit_ref.T).astype(np.float64)
        path = viterbi_v2(emis, vocab, fwd_pen=args.fwd_pen, back_pen=args.back_pen,
                          switch_pen=args.switch_pen, hold_pen=args.hold_pen,
                          skip_pen=args.skip_pen)
        dec = [vocab.tids[k] for k in vocab.track_of[path]]
        _eval_channel("bed", dec, args.frame_s, gt_rows)

    # overlay channel (MERT)
    key_of, items = build_overlay_vocab_mert(by_tid, gt_tids, args.frame_s, args.mert_layer)
    mix_vocals = set_dir / CHANNELS["overlay"]["mix_file"]
    ensure_mert_cache(items + [(mix_vocals, f"{args.set_id}_mix_vocals")],
                      args.frame_s, args.mert_layer)
    vocab = _assemble_vocab(key_of, args.frame_s, args.mert_layer)
    if not vocab.tids:
        print("  [overlay] no MERT vocab", file=sys.stderr)
        return 1
    mix = load_pooled_mert(f"{args.set_id}_mix_vocals", args.frame_s, args.mert_layer)
    if args.max_mix_s > 0:
        mix = mix[: int(args.max_mix_s / args.frame_s)]
    print(f"  [overlay] {len(vocab.tids)} tracks, {vocab.emit_ref.shape[0]} states, "
          f"{mix.shape[0]} mix frames", file=sys.stderr)
    emis = (mix @ vocab.emit_ref.T).astype(np.float64)
    path = viterbi_v2(emis, vocab, fwd_pen=args.fwd_pen, back_pen=args.back_pen,
                      switch_pen=args.switch_pen, hold_pen=args.hold_pen,
                      skip_pen=args.skip_pen)
    dec = [vocab.tids[k] for k in vocab.track_of[path]]
    _eval_channel("overlay", dec, args.frame_s, gt_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
