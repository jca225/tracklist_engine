#!/usr/bin/env python3
"""Set-id-general pre-seed: run the fused aligner on ANY pulled set (no GT) and
write a predicted_timeline.json that seed_als_from_timeline.py turns into a
draft Ableton project for the human to correct.

Unlike fuse.py (BB12-wired: cache keys + GT-derived vocab), this:
  * takes any --set-id; cache keys are set-id-namespaced
  * vocab comes from the set's MANIFEST (the tracklist), not a GT yaml
  * _stem read from the scraped track name (acappella/instrumental/regular)
  * no GT scoring (the set has none yet) — it just emits predictions

Output: workspaces/alignment_prototype/out/<set_id>_predicted_timeline.json
        (the path seed_als_from_timeline.py reads)

Usage:
    venvs/audio/bin/python -m workspaces.section_hsmm.preseed --set-id 2nvzlh2k
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
from workspaces.section_hsmm.decode_null import PENS, viterbi_null  # noqa: E402
from workspaces.section_hsmm.decode_v2 import CHANNELS, _pooled, build_channel_vocab  # noqa: E402
from workspaces.section_hsmm.decode_v7 import build_vocal_vocab  # noqa: E402
from workspaces.section_hsmm.decode_v9 import diag_smooth_multislope  # noqa: E402
from workspaces.section_hsmm.mfcc_emit import pooled_mfcc  # noqa: E402

SEED_OUT = _REPO / "workspaces/alignment_prototype/out"
SLOPES = (0.7, 0.85, 1.0, 1.18, 1.4)


def _scrape_stem(track: dict) -> str:
    name = (Path(track.get("local_path", "")).name or "").lower()
    if any(s in name for s in ("acap", "a cappella", "vocals only")):
        return "acappella"
    if "instrumental" in name:
        return "instrumental"
    return "regular"


def _spans(path, vocab, frame_s, channel, by_tid):
    S = vocab.emit_ref.shape[0]
    out, T, i = [], len(path), 0
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
        if j - i >= 2:
            tid = vocab.tids[k]
            t = by_tid.get(tid, {})
            stem = "acappella" if channel == "overlay" else (t.get("_stem") or "regular")
            out.append({
                "slot_label": str(t.get("label") or tid), "recording_id": tid,
                "name": f"{t.get('artist','?')} - {t.get('title','?')}",
                "claimed_stem": stem, "channel": channel,
                "set_start_s": round(i * frame_s, 2), "set_end_s": round(j * frame_s, 2),
                "ref_start_s": round(ref0, 2), "ref_end_s": round(last_ref, 2),
                "had_section_jump": jumped,
                # no tracklist cue anchor in a blind pre-seed -> flagged for review
                "cue_anchor_s": None,
            })
        i = j
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", required=True)
    p.add_argument("--bed-frame-s", type=float, default=2.0)
    p.add_argument("--bed-beta", type=float, default=-0.04)
    p.add_argument("--overlay-frame-s", type=float, default=1.0)
    p.add_argument("--overlay-win-frames", type=int, default=6)
    p.add_argument("--overlay-beta", type=float, default=-0.05)
    args = p.parse_args(argv)
    sid = args.set_id

    set_dir = find_aligning_dir(sid)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    for t in manifest["tracks"]:
        if t.get("recording_id"):
            by_tid.setdefault(t["recording_id"], t)
    for tid, t in by_tid.items():
        t["_stem"] = _scrape_stem(t)
    all_tids = list(dict.fromkeys(by_tid.keys()))

    # --- bed channel (chroma vs mix_instrumental) ---
    print("bed: building vocab + features (first run computes chroma) …", file=sys.stderr)
    bed_vocab = build_channel_vocab("bed", by_tid, all_tids, args.bed_frame_s)
    cfg = CHANNELS["bed"]
    bmix = _pooled(set_dir / cfg["mix_file"], f"{sid}_{cfg['mix_key']}",
                   f"{sid}_{cfg['mix_key']}_pool{args.bed_frame_s}", args.bed_frame_s)
    bed = []
    if bed_vocab.tids and bmix.shape[0]:
        emis = (bmix @ bed_vocab.emit_ref.T).astype(np.float64)
        path = viterbi_null(emis, bed_vocab, beta=args.bed_beta, null_enter_pen=0.05, **PENS)
        bed = _spans(path, bed_vocab, args.bed_frame_s, "bed", by_tid)
    print(f"  bed: {len(bed_vocab.tids)} tracks -> {len(bed)} spans", file=sys.stderr)

    # --- overlay channel (windowed MFCC vs mix_vocals) ---
    print("overlay: building vocab + features (first run computes MFCC) …", file=sys.stderr)
    ov_vocab = build_vocal_vocab(by_tid, args.overlay_frame_s)
    omix = pooled_mfcc(set_dir / "mix_vocals.flac", f"{sid}_mix_vocals",
                       f"{sid}_mix_vocals_pool{args.overlay_frame_s}", args.overlay_frame_s)
    voc = []
    if ov_vocab.tids and omix.shape[0]:
        emis = (omix @ ov_vocab.emit_ref.T).astype(np.float32)
        emis = diag_smooth_multislope(emis, ov_vocab.slices, args.overlay_win_frames, SLOPES)
        path = viterbi_null(emis, ov_vocab, beta=args.overlay_beta, null_enter_pen=0.0, **PENS)
        voc = _spans(path, ov_vocab, args.overlay_frame_s, "overlay", by_tid)
    print(f"  overlay: {len(ov_vocab.tids)} tracks -> {len(voc)} spans", file=sys.stderr)

    spans = sorted(bed + voc, key=lambda s: (s["set_start_s"], s["channel"]))
    SEED_OUT.mkdir(parents=True, exist_ok=True)
    out = SEED_OUT / f"{sid}_predicted_timeline.json"
    out.write_text(json.dumps({"set_id": sid, "n_spans": len(spans), "spans": spans}, indent=2))
    print(f"\nwrote {out}  ({len(spans)} spans: {len(bed)} bed + {len(voc)} vocal)")
    print("timeline head:")
    for s in spans[:14]:
        jp = " [jump]" if s["had_section_jump"] else ""
        print(f"  {s['set_start_s']/60:5.1f}m {s['channel']:7} {s['name'][:40]:40} "
              f"@ref {s['ref_start_s']:.0f}s{jp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
