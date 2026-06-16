#!/usr/bin/env python3
"""Abstention as precision-at-coverage: predict only where confident.

The GT itself abstains (3% of BB12 is left unlabeled as 'original mix', plus
self-reference host rows). So the aligner should too: a frame gets a prediction
only if its decoded state's emission (chroma/MERT cosine) beats a threshold tau;
otherwise it abstains. Sweeping tau traces precision (on predicted frames) vs
coverage (fraction of GT-active frames we chose to predict).

This reuses the v2 bed (chroma) and v3 overlay (MERT) decodes — no recompute
beyond the cached emissions.

Usage:
    venvs/audio/bin/python -m workspaces.section_hsmm.abstain_eval --set-id 1fsnxchk
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
from workspaces.section_hsmm.decode_hsmm import _gt_track_ids  # noqa: E402
from workspaces.section_hsmm.decode_v2 import (  # noqa: E402
    CHANNELS, _pooled, build_channel_vocab, viterbi_v2,
)
from workspaces.section_hsmm.decode_v3 import build_overlay_vocab_mert, _assemble_vocab  # noqa: E402
from workspaces.section_hsmm.mert_emit import ensure_mert_cache, load_pooled_mert  # noqa: E402

PENS = dict(fwd_pen=0.15, back_pen=0.30, switch_pen=0.50, hold_pen=0.10, skip_pen=0.10)


def _decode_channel(channel, set_dir, by_tid, gt_tids, frame_s, layer):
    if channel == "bed":
        vocab = build_channel_vocab("bed", by_tid, gt_tids, frame_s)
        cfg = CHANNELS["bed"]
        mix = _pooled(set_dir / cfg["mix_file"], f"1fsnxchk_{cfg['mix_key']}",
                      f"1fsnxchk_{cfg['mix_key']}_pool{frame_s}", frame_s)
    else:
        key_of, items = build_overlay_vocab_mert(by_tid, gt_tids, frame_s, layer)
        ensure_mert_cache(items + [(set_dir / CHANNELS["overlay"]["mix_file"],
                                    f"1fsnxchk_mix_vocals")], frame_s, layer)
        vocab = _assemble_vocab(key_of, frame_s, layer)
        mix = load_pooled_mert("1fsnxchk_mix_vocals", frame_s, layer)
    emis = (mix @ vocab.emit_ref.T).astype(np.float64)
    path = viterbi_v2(emis, vocab, **PENS)
    T = len(path)
    # per-track max emission per frame, then margin of the decoded track vs the
    # best OTHER track — "how clearly is it this song, not another?"
    K = len(vocab.tids)
    track_max = np.full((T, K), -1e9)
    for k, (lo, hi) in enumerate(vocab.slices):
        track_max[:, k] = emis[:, lo:hi].max(axis=1)
    dk = vocab.track_of[path]
    own = track_max[np.arange(T), dk]
    tm = track_max.copy()
    tm[np.arange(T), dk] = -1e9
    margin = own - tm.max(axis=1)
    dec_tid = [vocab.tids[k] for k in dk]
    return dec_tid, margin


def _sweep(channel, dec_tid, conf, frame_s, gt_rows):
    stems = CHANNELS[channel]["stems"]
    rows = [r for r in gt_rows if (r.get("claimed_stem") or "regular") in stems]

    def active(ts):
        return {str(r["track_id"]) for r in rows
                if float(r["set_start_s"]) - 2 <= ts <= float(r["set_end_s"]) + 2}

    frames = []  # (correct, margin) for GT-active frames only
    for i, tid in enumerate(dec_tid):
        act = active((i + 0.5) * frame_s)
        if act:
            frames.append((tid in act, float(conf[i])))
    n = len(frames)
    print(f"\n[{channel}] {n} GT-active frames  (no-abstain accuracy = "
          f"{100*np.mean([c for c,_ in frames]):.0f}%)")
    print(f"  {'margin>=':>8} {'coverage':>9} {'precision':>10}")
    for tau in (-1.0, 0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
        pred = [(c, cf) for c, cf in frames if cf >= tau]
        if not pred:
            print(f"  {tau:8.2f}      0%        -")
            continue
        cov = len(pred) / n
        prec = np.mean([c for c, _ in pred])
        print(f"  {tau:8.2f} {100*cov:7.0f}%  {100*prec:8.0f}%")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", default="1fsnxchk")
    p.add_argument("--frame-s", type=float, default=2.0)
    p.add_argument("--mert-layer", type=int, default=6)
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

    print(f"=== abstention precision@coverage ({args.set_id}) ===")
    for channel in ("bed", "overlay"):
        dec_tid, conf = _decode_channel(channel, set_dir, by_tid, gt_tids,
                                        args.frame_s, args.mert_layer)
        _sweep(channel, dec_tid, conf, args.frame_s, gt_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
