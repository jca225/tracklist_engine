#!/usr/bin/env python3
"""Score a predicted timeline (infer + refine_ref_offsets) against GT.

End-to-end pipeline scorecard — unlike eval_ref_detection (which probes with
GT set positions), this scores the actual pipeline output: identity, set
placement, and ref offsets, per stem channel. Ref offsets are scored only on
straight-clip GT rows (loops/segments aren't representable by the current
single-(ref_start, stretch) span output — counted separately).

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.score_timeline_vs_gt \\
        --set-id 1fsnxchk [--gt labeling/fixtures/bb12_ground_truth.yaml]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.alignment_prototype.path_decode import FPS, _span_class, trajectory_acc
from workspaces.alignment_prototype.refine_ref_offsets import (
    _STEM_FILE,
    find_aligning_dir,
)

OUT_DIR = Path(__file__).resolve().parent / "out"


def norm_slot(s: str) -> str:
    """'006w2' -> '6w2', '013' -> '13' — GT zero-pads, set_track_slots doesn't."""
    m = re.match(r"^0*(\d+)(w\d+)?$", str(s).strip())
    return f"{m.group(1)}{m.group(2) or ''}" if m else str(s).strip()


def _pred_segs_from_span(s: dict) -> list[tuple[float, float, float]]:
    """Predicted ref_segments -> decode_path convention: [(mix_start_REL, ref_start,
    ref_end)] (mix-start span-relative, ref absolute). Absorbs both segment schemas
    (joint_ref_decode legacy {mix_start_s ABS, dur_s} and the GT/decode_path
    {mix_start_s ABS, ref_end_s}); falls back to a one-segment straight line when the
    span carries only a scalar ref_start_s (measures the headroom lost without segments)."""
    s0 = float(s["set_start_s"])
    stretch = float(s.get("ref_stretch") or 1.0)
    segs = s.get("ref_segments")
    if segs:
        out = []
        for seg in segs:
            rs = float(seg["ref_start_s"])
            if "ref_end_s" in seg:
                re_ = float(seg["ref_end_s"])
            else:  # legacy dur_s (mix seconds) -> ref_end via stretch
                re_ = rs + float(seg["dur_s"]) * stretch
            out.append((float(seg["mix_start_s"]) - s0, rs, re_))
        return out
    rs = float(s["ref_start_s"])
    re_ = (
        float(s["ref_end_s"])
        if s.get("ref_end_s") is not None
        else (rs + (float(s["set_end_s"]) - s0) * stretch)
    )
    return [(0.0, rs, re_)]


def _resolve_ref_audio(span: dict, track: dict | None) -> str | None:
    """Stem-routed reference audio path for a span (vocals/instrumental stem or the
    full track), for HuBERT fiber computation."""
    if track is None:
        return None
    stem_key = _STEM_FILE.get(span.get("claimed_stem") or "regular")
    if stem_key:
        p = (track.get("stems") or {}).get(stem_key)
        if p and Path(p).is_file():
            return p
    p = track.get("local_path")
    return p if p and Path(p).is_file() else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", required=True)
    p.add_argument(
        "--gt", type=Path, default=_REPO / "labeling/fixtures/bb12_ground_truth.yaml"
    )
    p.add_argument(
        "--fibers",
        action="store_true",
        help="fiber-aware trajectory scoring (HuBERT repeat classes; one HuBERT pass "
        "per ref — expensive)",
    )
    p.add_argument("--hubert-layer", type=int, default=9)
    p.add_argument(
        "--timeline",
        type=Path,
        default=None,
        help="score an arbitrary timeline JSON (default: out/<set-id>_predicted_timeline.json)",
    )
    args = p.parse_args(argv)

    tl_path = args.timeline or (OUT_DIR / f"{args.set_id}_predicted_timeline.json")
    timeline = json.loads(Path(tl_path).read_text())
    # manifest by track_id — only needed for fiber ref-audio resolution
    by_tid: dict[str, dict] = {}
    if args.fibers:
        set_dir = find_aligning_dir(args.set_id)
        for t in json.loads((set_dir / "manifest.json").read_text())["tracks"]:
            by_tid[t["track_id"]] = t
            if t.get("recording_id"):
                by_tid.setdefault(t["recording_id"], t)
    gt_rows = [
        r
        for r in yaml.safe_load(args.gt.read_text())["tracks"]
        if str(r.get("slot_label")) != "mix"
    ]
    # GT slot labels are the HUMAN's section numbering (002-155 on BB12),
    # not the tracklist's slot space — match by recording + time, never by
    # slot label.
    gt_by_tid: dict[str, list[dict]] = {}
    for r in gt_rows:
        if r.get("track_id"):
            gt_by_tid.setdefault(str(r["track_id"]), []).append(r)

    fiber_cache: dict[str, tuple] = {}

    def fibers_for(ref_audio: str | None):
        if not args.fibers or ref_audio is None:
            return None
        if ref_audio not in fiber_cache:
            from workspaces.alignment_prototype.path_decode import _ensure_feat
            from workspaces.alignment_prototype.ref_fibers import compute_fibers

            hf = np.load(
                _ensure_feat(ref_audio, ref_audio, "hubert", args.hubert_layer)
            )
            fiber_cache[ref_audio] = compute_fibers(hf, FPS, audio_path=ref_audio)
        return fiber_cache[ref_audio]

    id_ok, id_bad, no_gt = 0, [], 0
    place_errs, ref_rows, traj = [], [], []
    loops_hit = 0
    for s in timeline["spans"]:
        slot = norm_slot(s["slot_label"])
        # identity: any GT row overlapping the predicted span in time whose
        # track matches? (GT rows without track_id can't vote)
        overlapping = [
            r
            for r in gt_rows
            if r.get("track_id")
            and float(r["set_start_s"]) < s["set_end_s"] + 5
            and float(r["set_end_s"]) > s["set_start_s"] - 5
        ]
        if overlapping:
            if any(str(r["track_id"]) == s["recording_id"] for r in overlapping):
                id_ok += 1
            else:
                id_bad.append(
                    (
                        slot,
                        s["recording_id"],
                        sorted({str(r["track_id"]) for r in overlapping})[:3],
                        s["name"][:36],
                    )
                )
        # placement + ref: nearest same-recording GT row
        rows = gt_by_tid.get(s["recording_id"])
        if not rows:
            no_gt += 1
            continue
        g = min(rows, key=lambda r: abs(float(r["set_start_s"]) - s["set_start_s"]))
        place_errs.append(
            (
                abs(float(g["set_start_s"]) - s["set_start_s"]),
                slot,
                s["set_start_s"],
                float(g["set_start_s"]),
                s["name"][:36],
            )
        )
        # trajectory accuracy: scores ref(mix_t) coverage for EVERY span class
        # (linear / multiseg / loop / oddratio), the metric that was previously
        # excluded for loops/segments. strict = fraction of mix-time within 2s of
        # GT ref; fiber-aware credits a content-identical repeat.
        fib = fibers_for(_resolve_ref_audio(s, by_tid.get(s["recording_id"])))
        strict, _npred, facc = trajectory_acc(_pred_segs_from_span(s), g, fiber=fib)
        traj.append((_span_class(g), s.get("claimed_stem") or "regular", strict, facc))
        if g.get("is_loop") or g.get("ref_segments"):
            loops_hit += 1
            continue
        ratio = float(g.get("tempo_ratio") or 1.0)
        if not (0.9 <= ratio <= 1.15):
            loops_hit += 1
            continue
        expected = (
            float(g["ref_start_s"])
            + (s["set_start_s"] - float(g["set_start_s"])) * ratio
        )
        ref_rows.append(
            (
                abs(s["ref_start_s"] - expected),
                slot,
                s.get("claimed_stem") or "regular",
                s["ref_start_s"],
                expected,
                s["name"][:36],
            )
        )

    n = len(timeline["spans"])
    print(f"=== end-to-end pipeline vs GT ({args.set_id}, {n} predicted spans) ===")
    nid = id_ok + len(id_bad)
    print(
        f"identity: {id_ok}/{nid} ({100 * id_ok / max(nid, 1):.0f}%)  "
        f"[{no_gt} spans had no same-slot GT row]"
    )
    pe = np.array([r[0] for r in place_errs])
    print(
        f"set placement |pred-gt|: median={np.median(pe):.1f}s  "
        f"<5s: {100 * (pe < 5).mean():.0f}%  <15s: {100 * (pe < 15).mean():.0f}%  "
        f"p90={np.percentile(pe, 90):.1f}s  (n={len(pe)})"
    )
    re_ = np.array([r[0] for r in ref_rows])
    if re_.size:
        print(
            f"ref offset |pred-gt| (straight clips, n={len(re_)}; "
            f"{loops_hit} loop/segment spans excluded): "
            f"median={np.median(re_):.1f}s  <2s: {100 * (re_ < 2).mean():.0f}%  "
            f"<5s: {100 * (re_ < 5).mean():.0f}%  p90={np.percentile(re_, 90):.1f}s"
        )
        for stem in ("regular", "acappella", "instrumental"):
            e = np.array([r[0] for r in ref_rows if r[2] == stem])
            if e.size:
                print(
                    f"  {stem:13} n={len(e):3} median={np.median(e):.1f}s  "
                    f"<2s: {100 * (e < 2).mean():.0f}%  <5s: {100 * (e < 5).mean():.0f}%"
                )

    # trajectory accuracy (ref structure, ALL span classes) — the headline for
    # segment output: previously loop/segment spans were unscored.
    if traj:
        which = "fiber-aware" if args.fibers else "strict"

        def _ta(rows_):
            st = np.array([r[2] for r in rows_])
            fa = np.array([r[3] for r in rows_])
            v = fa if args.fibers else st
            return f"n={len(v):3} traj-acc={100 * v.mean():.0f}%  >=80%covered: {100 * (v >= 0.8).mean():.0f}%"

        print(
            f"\nref trajectory ({which}, ref(mix_t) within 2s; n={len(traj)} matched):"
        )
        for cls in ("linear", "multiseg", "loop", "oddratio"):
            rc = [r for r in traj if r[0] == cls]
            if rc:
                print(f"  class {cls:9} {_ta(rc)}")
        for stem in ("regular", "acappella", "instrumental"):
            rs = [r for r in traj if r[1] == stem]
            if rs:
                print(f"  stem  {stem:9} {_ta(rs)}")
        nonlin = [r for r in traj if r[0] in ("multiseg", "loop")]
        if nonlin:
            print(f"  HEADLINE multiseg+loop {_ta(nonlin)}")

    print("\nworst placement:")
    for err, slot, pred, gt_v, name in sorted(place_errs, reverse=True)[:8]:
        print(f"  {err:7.1f}s {slot:6} pred={pred:8.1f} gt={gt_v:8.1f}  {name}")
    print("\nworst ref offset:")
    for err, slot, _stem, pred, exp, name in sorted(ref_rows, reverse=True)[:8]:
        print(f"  {err:7.1f}s {slot:6} pred={pred:8.1f} gt={exp:8.1f}  {name}")
    if id_bad:
        print(f"\nidentity misses ({len(id_bad)}):")
        for slot, rid, tids, name in id_bad[:10]:
            print(f"  {slot:6} pred={rid} gt={','.join(tids[:3])}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
