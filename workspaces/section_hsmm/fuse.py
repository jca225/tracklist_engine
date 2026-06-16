#!/usr/bin/env python3
"""v5 — fuse bed + overlay NULL-state decodes into one layered timeline.

Runs the abstaining NULL-state Viterbi on both channels, collapses each per-frame
path into contiguous (track, ref-offset) spans (abstained frames become gaps),
and writes a single layered timeline JSON — the actual aligner output, with two
layers (harmonic bed + acappella overlay) that abstain independently. Also prints
a span-level scorecard vs GT and a readable timeline head.

Output: workspaces/section_hsmm/out/<set_id>_fused_timeline.json

Usage:
    venvs/audio/bin/python -m workspaces.section_hsmm.fuse --set-id 1fsnxchk \
        [--bed-beta -0.04] [--overlay-beta -0.04]
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
from workspaces.section_hsmm.decode_null import PENS, _channel_emis, viterbi_null  # noqa: E402
from workspaces.section_hsmm.decode_v2 import CHANNELS  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "out"


def spans_from_path(path: np.ndarray, vocab, frame_s: float, channel: str,
                    by_tid: dict) -> list[dict]:
    S = vocab.emit_ref.shape[0]
    spans: list[dict] = []
    T = len(path)
    i = 0
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
        tid = vocab.tids[k]
        t = by_tid.get(tid, {})
        name = f"{t.get('artist','?')} - {t.get('title','?')}"
        spans.append({
            "channel": channel,
            "recording_id": tid,
            "name": name,
            "claimed_stem": t.get("_stem", "regular"),
            "set_start_s": round(i * frame_s, 1),
            "set_end_s": round(j * frame_s, 1),
            "ref_start_s": round(ref0, 1),
            "ref_end_s": round(last_ref, 1),
            "n_frames": j - i,
            "had_section_jump": jumped,
        })
        i = j
    return spans


def score_spans(spans: list[dict], gt_rows: list[dict], stems: tuple[str, ...]) -> tuple[int, int]:
    rows = [r for r in gt_rows if (r.get("claimed_stem") or "regular") in stems]
    ok = 0
    for s in spans:
        mid = 0.5 * (s["set_start_s"] + s["set_end_s"])
        active = {str(r["track_id"]) for r in rows
                  if float(r["set_start_s"]) - 3 <= mid <= float(r["set_end_s"]) + 3}
        if s["recording_id"] in active:
            ok += 1
    return ok, len(spans)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", default="1fsnxchk")
    p.add_argument("--frame-s", type=float, default=2.0)
    p.add_argument("--mert-layer", type=int, default=6)
    p.add_argument("--bed-beta", type=float, default=-0.04)
    p.add_argument("--overlay-beta", type=float, default=-0.04)
    p.add_argument("--null-enter-pen", type=float, default=0.05)
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

    betas = {"bed": args.bed_beta, "overlay": args.overlay_beta}
    all_spans: list[dict] = []
    print(f"=== v5 fused layered timeline ({args.set_id}) ===")
    for channel in ("bed", "overlay"):
        vocab, emis = _channel_emis(channel, set_dir, by_tid, gt_tids,
                                    args.frame_s, args.mert_layer)
        path = viterbi_null(emis, vocab, beta=betas[channel],
                            null_enter_pen=args.null_enter_pen, **PENS)
        spans = spans_from_path(path, vocab, args.frame_s, channel, by_tid)
        cov = 100 * (path != vocab.emit_ref.shape[0]).mean()
        ok, n = score_spans(spans, gt_rows, CHANNELS[channel]["stems"])
        jumps = sum(1 for s in spans if s["had_section_jump"])
        print(f"  [{channel:7}] beta={betas[channel]:+.2f}  {n} spans  "
              f"({jumps} w/ section-jump)  frame-coverage {cov:.0f}%  "
              f"span-identity {ok}/{n} = {100*ok/max(n,1):.0f}%")
        all_spans.extend(spans)

    all_spans.sort(key=lambda s: (s["set_start_s"], s["channel"]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.set_id}_fused_timeline.json"
    out.write_text(json.dumps({
        "set_id": args.set_id, "mix_duration_s": mix_dur,
        "frame_s": args.frame_s, "betas": betas,
        "n_spans": len(all_spans), "spans": all_spans,
    }, indent=2))
    print(f"\nwrote {out}  ({len(all_spans)} spans)")

    print("\ntimeline head (first 6 min):")
    for s in all_spans:
        if s["set_start_s"] > 360:
            break
        m0, m1 = s["set_start_s"] / 60, s["set_end_s"] / 60
        jp = " [jump]" if s["had_section_jump"] else ""
        print(f"  {m0:4.1f}-{m1:4.1f}m  {s['channel']:7} {s['name'][:44]:44} "
              f"@ref {s['ref_start_s']:.0f}-{s['ref_end_s']:.0f}s{jp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
