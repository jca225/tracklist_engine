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

from workspaces.alignment_prototype.path_decode import (
    FPS,
    _gt_pieces,
    _pieces,
    _ref_at,
    _span_class,
    trajectory_acc,
)
from workspaces.alignment_prototype.refine_ref_offsets import (
    _STEM_FILE,
    find_aligning_dir,
)

OUT_DIR = Path(__file__).resolve().parent / "out"


def norm_slot(s: str) -> str:
    """'006w2' -> '6w2', '013' -> '13' — GT zero-pads, set_track_slots doesn't."""
    m = re.match(r"^0*(\d+)(w\d+)?$", str(s).strip())
    return f"{m.group(1)}{m.group(2) or ''}" if m else str(s).strip()


def _decompose_span(pred_segs, row) -> tuple[int, int, int]:
    """(n_seconds, n_outside_decode, n_inside_correct) — mirrors
    trajectory_acc's sampling exactly (same helpers, same inputs) and adds
    the coverage split: a sampled GT second is OUTSIDE when it falls before
    the first decoded segment or after the last segment's own extent (the
    piecewise interpolation extrapolates there; those seconds measure
    window/extent misses, not decode quality)."""
    s0, s1 = float(row["set_start_s"]), float(row["set_end_s"])
    if s1 <= s0 or not pred_segs:
        n = max(0, int(np.ceil(s1 - s0)))
        return n, n, 0
    gt = _gt_pieces(row)
    slope = float(row.get("tempo_ratio") or 1.0)
    pred = _pieces([(s0 + ms, rs, re) for (ms, rs, re) in pred_segs], s0, s1, slope)
    first = pred[0][0]
    lm0, lm1, lrs, lsl = pred[-1]
    last_end = lm0 + max(lm1 - lm0, 0.0)
    n_out = n_ok = n_tot = 0
    for t in np.arange(s0, s1, 1.0):
        n_tot += 1
        pr = _ref_at(pred, float(t))
        gr = _ref_at(gt, float(t))
        if t < first or t > last_end:
            n_out += 1
        elif abs(pr - gr) < 2.0:
            n_ok += 1
    return n_tot, n_out, n_ok


def _pred_segs_from_span(
    s: dict, anchor_s: float | None = None
) -> list[tuple[float, float, float]]:
    """Predicted ref_segments -> decode_path convention: [(mix_start_REL, ref_start,
    ref_end)] (mix-start span-relative, ref absolute). Absorbs both segment schemas
    (joint_ref_decode legacy {mix_start_s ABS, dur_s} and the GT/decode_path
    {mix_start_s ABS, ref_end_s}); falls back to a one-segment straight line when the
    span carries only a scalar ref_start_s (measures the headroom lost without segments).

    anchor_s: the reference start the caller will re-add (trajectory_acc adds the GT
    row's set_start). Timeline ref_segments carry ABSOLUTE mix positions, so anchoring
    at the SPAN's own start and re-adding the GT start TRANSLATES the segments by the
    placement error — double-counting it (measured: BB12 acappella 6%% vs 18%% on
    identical predictions). Pass the matched GT row's set_start for absolute scoring."""
    s0 = float(s["set_start_s"]) if anchor_s is None else float(anchor_s)
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


def _resolve_ref_audio(
    span: dict, track: dict | None, stem: str | None = None
) -> str | None:
    """Stem-routed reference audio path for a span (vocals/instrumental stem or the
    full track), for HuBERT fiber computation. `stem` overrides the span's own
    `claimed_stem` (which is the materialized set_track_slots value — stale on
    pre-888aca timelines; the matched GT row is authoritative)."""
    if track is None:
        return None
    stem_key = _STEM_FILE.get(stem or span.get("claimed_stem") or "regular")
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
        "--gt",
        type=Path,
        default=None,
        help="GT yaml (default: the labeling/fixtures/*_ground_truth.yaml whose "
        "set_id matches --set-id; error if none matches — never a wrong-set GT)",
    )
    p.add_argument(
        "--fibers",
        action="store_true",
        help="fiber-aware trajectory scoring (HuBERT repeat classes; one HuBERT pass "
        "per ref — expensive)",
    )
    p.add_argument("--hubert-layer", type=int, default=9)
    p.add_argument(
        "--decompose",
        action="store_true",
        help="per-second gap attribution INSIDE the scoring loop: "
        "outside-decode-window vs in-window accuracy (+GT-side match audit)",
    )
    p.add_argument(
        "--timeline",
        type=Path,
        default=None,
        help="score an arbitrary timeline JSON (default: out/<set-id>_predicted_timeline.json)",
    )
    args = p.parse_args(argv)

    if args.gt is None:
        # Resolve GT by set_id — a hardcoded default once scored BB11 against
        # BB12's GT (silent 84%→0% "catastrophe", 2026-07-09).
        fixtures = sorted((_REPO / "labeling" / "fixtures").glob("*_ground_truth.yaml"))
        matches = [
            f
            for f in fixtures
            if yaml.safe_load(f.read_text()).get("set_id") == args.set_id
        ]
        if len(matches) != 1:
            p.error(
                f"--gt not given and {len(matches)} GT fixtures match "
                f"set_id={args.set_id} (looked in labeling/fixtures/)"
            )
        args.gt = matches[0]
        print(f"(gt: {args.gt.name})")

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
    # Identity is scored in the CANONICAL recording_id namespace. GT rows that
    # path-matched the pull manifest carry scrape-namespace tlp* ids; map them
    # through labeling/fixtures/id_maps/<set>.json (tlp_id -> recording_id,
    # from set_track_slots) so namespace mismatch can't masquerade as an
    # identity miss.
    id_map_path = _REPO / "labeling" / "fixtures" / "id_maps" / f"{args.set_id}.json"
    id_map: dict[str, str] = (
        json.loads(id_map_path.read_text()) if id_map_path.exists() else {}
    )
    if id_map:
        mapped = 0
        for r in gt_rows:
            tid = str(r.get("track_id") or "")
            if tid in id_map and id_map[tid] != tid:
                r["track_id"] = id_map[tid]
                mapped += 1
        print(f"(id map: {mapped} GT ids normalized via {id_map_path.name})")
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
    decomp: list[tuple] = []
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
        # Axis from the matched GT row, NEVER the timeline span: the span's
        # claimed_stem is the materialized set_track_slots value, corrupted by
        # the row-text drop bug on pre-888aca timelines (BB12 showed 2
        # instrumentals vs 25 in GT) — see eda/alignment/failure_analysis.
        gstem = g.get("claimed_stem") or s.get("claimed_stem") or "regular"
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
        fib = fibers_for(
            _resolve_ref_audio(s, by_tid.get(s["recording_id"]), stem=gstem)
        )
        strict, _npred, facc = trajectory_acc(
            _pred_segs_from_span(s, anchor_s=float(g["set_start_s"])), g, fiber=fib
        )
        traj.append((_span_class(g), gstem, strict, facc))
        if args.decompose:
            nt, no, nk = _decompose_span(
                _pred_segs_from_span(s, anchor_s=float(g["set_start_s"])), g
            )
            decomp.append((gstem, nt, no, nk, strict))
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
                gstem,
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
    if args.decompose and decomp:
        print("\ngap decomposition (per sampled GT second):")
        for stem in ("acappella", "regular"):
            rows_ = [d for d in decomp if d[0] == stem]
            if not rows_:
                continue
            nt = sum(d[1] for d in rows_)
            no = sum(d[2] for d in rows_)
            nk = sum(d[3] for d in rows_)
            n_in = nt - no
            # faithfulness cross-check: per-span mean of in+out accuracy vs metric
            per_span = float(np.mean([(d[3] / d[1]) if d[1] else 0.0 for d in rows_]))
            metric = float(np.mean([d[4] for d in rows_]))
            print(
                f"  {stem:10} seconds={nt}  outside-decode {100 * no / nt:.0f}%  "
                f"in-window acc {100 * nk / max(1, n_in):.0f}%  "
                f"(xcheck strict-per-span {100 * per_span:.0f}% vs metric {100 * metric:.0f}%)"
            )
        # GT-side: acappella rows never matched by any timeline span
        matched_tids = {s2["recording_id"] for s2 in timeline["spans"]}
        gt_aca = [
            r
            for r in gt_rows
            if r.get("claimed_stem") == "acappella" and r.get("track_id")
        ]
        unmatched = [r for r in gt_aca if str(r["track_id"]) not in matched_tids]
        print(
            f"  GT acappella rows: {len(gt_aca)}; recording matched by SOME timeline span: "
            f"{len(gt_aca) - len(unmatched)}; NEVER matched (invisible to metric): {len(unmatched)}"
        )

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
