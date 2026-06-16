#!/usr/bin/env python3
"""v5+v9 — fuse bed (chroma) + vocal (windowed-MFCC) into one layered timeline.

Two channels, each at its own resolution and feature, both abstaining via the
NULL-state Viterbi:
  bed     = chroma @ 2s, all instrumental/regular refs   (94% span-identity)
  overlay = windowed multislope MFCC @ 1s, label-agnostic vocal refs (v9 —
            55% event recall / 60% precision, warp-tolerant)
Collapses each per-frame path into contiguous (track, ref-offset) spans
(abstains -> gaps) and writes one layered timeline JSON.

Output: workspaces/section_hsmm/out/<set_id>_fused_timeline.json

Usage:
    venvs/audio/bin/python -m workspaces.section_hsmm.fuse --set-id 1fsnxchk
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
from workspaces.section_hsmm.decode_null import PENS, viterbi_null  # noqa: E402
from workspaces.section_hsmm.decode_v2 import CHANNELS, _pooled, build_channel_vocab  # noqa: E402
from workspaces.section_hsmm.decode_v7 import build_vocal_vocab  # noqa: E402
from workspaces.section_hsmm.decode_v8 import event_score  # noqa: E402
from workspaces.section_hsmm.decode_v9 import diag_smooth_multislope  # noqa: E402
from workspaces.section_hsmm.mfcc_emit import pooled_mfcc  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "out"
SLOPES = (0.7, 0.85, 1.0, 1.18, 1.4)


def spans_from_path(path, vocab, frame_s, channel, by_tid, min_frames=2):
    S = vocab.emit_ref.shape[0]
    spans, T, i = [], len(path), 0
    while i < T:
        st = int(path[i])
        if st == S:
            i += 1
            continue
        k = int(vocab.track_of[st])
        ref0 = float(vocab.ref_frame[st]) * frame_s
        last_ref, jumped, j = ref0, False, i
        while j < T and int(path[j]) != S and int(vocab.track_of[path[j]]) == k:
            rf = float(vocab.ref_frame[path[j]]) * frame_s
            if j > i and abs(rf - last_ref) > 2 * frame_s + 0.5:
                jumped = True
            last_ref = rf
            j += 1
        if j - i >= min_frames:
            t = by_tid.get(vocab.tids[k], {})
            spans.append({
                "channel": channel, "recording_id": vocab.tids[k],
                "name": f"{t.get('artist','?')} - {t.get('title','?')}",
                "claimed_stem": t.get("_stem", "regular"),
                "set_start_s": round(i * frame_s, 1), "set_end_s": round(j * frame_s, 1),
                "ref_start_s": round(ref0, 1), "ref_end_s": round(last_ref, 1),
                "n_frames": j - i, "had_section_jump": jumped,
            })
        i = j
    return spans


def _bed_spans(set_dir, by_tid, gt_tids, frame_s, beta, by_name) -> list[dict]:
    vocab = build_channel_vocab("bed", by_tid, gt_tids, frame_s)
    cfg = CHANNELS["bed"]
    mix = _pooled(set_dir / cfg["mix_file"], f"1fsnxchk_{cfg['mix_key']}",
                  f"1fsnxchk_{cfg['mix_key']}_pool{frame_s}", frame_s)
    emis = (mix @ vocab.emit_ref.T).astype(np.float64)
    path = viterbi_null(emis, vocab, beta=beta, null_enter_pen=0.05, **PENS)
    return spans_from_path(path, vocab, frame_s, "bed", by_tid)


def _vocal_spans(set_dir, by_tid, frame_s, win_frames, beta) -> list[dict]:
    vocab = build_vocal_vocab(by_tid, frame_s)
    mix = pooled_mfcc(set_dir / "mix_vocals.flac", "1fsnxchk_mix_vocals",
                      f"1fsnxchk_mix_vocals_pool{frame_s}", frame_s)
    emis = (mix @ vocab.emit_ref.T).astype(np.float32)
    emis = diag_smooth_multislope(emis, vocab.slices, win_frames, SLOPES)
    path = viterbi_null(emis, vocab, beta=beta, null_enter_pen=0.0, **PENS)
    return spans_from_path(path, vocab, frame_s, "overlay", by_tid)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", default="1fsnxchk")
    p.add_argument("--bed-frame-s", type=float, default=2.0)
    p.add_argument("--bed-beta", type=float, default=-0.04)
    p.add_argument("--overlay-frame-s", type=float, default=1.0)
    p.add_argument("--overlay-win-frames", type=int, default=6)
    p.add_argument("--overlay-beta", type=float, default=-0.05)
    args = p.parse_args(argv)

    set_dir = find_aligning_dir(args.set_id)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    mix_dur = float(manifest.get("mix_duration_s") or 0.0)
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

    print(f"=== fused layered timeline ({args.set_id}) ===", file=sys.stderr)
    bed = _bed_spans(set_dir, by_tid, gt_tids, args.bed_frame_s, args.bed_beta, by_tid)
    voc = _vocal_spans(set_dir, by_tid, args.overlay_frame_s,
                       args.overlay_win_frames, args.overlay_beta)

    # scorecards: bed span-identity, overlay event-level
    bed_ok = sum(1 for s in bed if _span_correct(s, gt_rows, ("regular", "instrumental")))
    print(f"=== fused layered timeline ({args.set_id}) ===")
    print(f"  [bed    ] {len(bed)} spans  span-identity {bed_ok}/{len(bed)} = "
          f"{100*bed_ok/max(len(bed),1):.0f}%")
    vtuples = [(s["recording_id"], s["set_start_s"], s["set_end_s"]) for s in voc]
    rec, prec, nev, nsp, onsets = event_score(vtuples, gt_rows)
    om = f"{np.median(onsets):.1f}s" if onsets else "-"
    print(f"  [overlay] {len(voc)} spans  event recall {100*rec:.0f}% / "
          f"precision {100*prec:.0f}% / onset {om}  ({nev} GT vocal events)")

    all_spans = sorted(bed + voc, key=lambda s: (s["set_start_s"], s["channel"]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.set_id}_fused_timeline.json"
    out.write_text(json.dumps({
        "set_id": args.set_id, "mix_duration_s": mix_dur,
        "bed": {"frame_s": args.bed_frame_s, "feature": "chroma", "beta": args.bed_beta},
        "overlay": {"frame_s": args.overlay_frame_s, "feature": "mfcc-windowed",
                    "win_frames": args.overlay_win_frames, "beta": args.overlay_beta},
        "n_spans": len(all_spans), "spans": all_spans,
    }, indent=2))
    print(f"\nwrote {out}  ({len(all_spans)} spans: {len(bed)} bed + {len(voc)} vocal)")

    print("\ntimeline head (first 6 min):")
    for s in all_spans:
        if s["set_start_s"] > 360:
            break
        jp = " [jump]" if s["had_section_jump"] else ""
        print(f"  {s['set_start_s']/60:4.1f}-{s['set_end_s']/60:4.1f}m  {s['channel']:7} "
              f"{s['name'][:42]:42} @ref {s['ref_start_s']:.0f}-{s['ref_end_s']:.0f}s{jp}")
    return 0


def _span_correct(s, gt_rows, stems) -> bool:
    rows = [r for r in gt_rows if (r.get("claimed_stem") or "regular") in stems]
    mid = 0.5 * (s["set_start_s"] + s["set_end_s"])
    return s["recording_id"] in {str(r["track_id"]) for r in rows
                                 if float(r["set_start_s"]) - 3 <= mid <= float(r["set_end_s"]) + 3}


if __name__ == "__main__":
    sys.exit(main())
