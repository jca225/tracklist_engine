#!/usr/bin/env python3
"""Score a predicted timeline (infer + refine_ref_offsets) against GT.

End-to-end pipeline scorecard — unlike eval_ref_detection (which probes with
GT set positions), this scores the actual pipeline output: identity, set
placement, and ref offsets, per stem channel. Ref offsets are scored only on
straight-clip GT rows (loops/segments aren't representable by the current
single-(ref_start, stretch) span output — counted separately).

Usage:
    venvs/audio/bin/python -m alignment.score_timeline_vs_gt \\
        --set-id 1fsnxchk [--gt labeling/fixtures/bb12_ground_truth.yaml]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from core.contracts import join_guard, load_manifest, load_timeline

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from alignment.path_decode import (
    FPS,
    _audible_intervals,
    _gt_pieces,
    _pieces,
    _ref_at,
    _span_class,
    gt_placement_onset,
    trajectory_acc,
)
from alignment.identity_bridge import (
    canonicalize_gt_rows,
    load_identity_map,
)
from alignment.never_matched import (
    identity_recall,
    unmatched_gt_forms,
)
from alignment.refine_ref_offsets import (
    _STEM_FILE,
    find_aligning_dir,
)
from eda.alignment.spectrogram_review.source_audio import run_audio_preflight

OUT_DIR = Path(__file__).resolve().parent / "out"


@dataclass(frozen=True)
class SpanScore:
    slot: str
    recording_id: str
    stem: str | None  # matched-GT claimed_stem (axis), None if no same-rec GT
    span_class: str | None  # linear/multiseg/loop/oddratio, None if no same-rec GT
    id_correct: bool | None  # None if no overlapping GT row
    place_err_s: float | None
    strict: float | None
    fiber: float | None
    ref_err_s: float | None  # straight clips only
    density: int | None


def norm_slot(s: str) -> str:
    """'006w2' -> '6w2', '013' -> '13' — GT zero-pads, set_track_slots doesn't."""
    m = re.match(r"^0*(\d+)(w\d+)?$", str(s).strip())
    return f"{m.group(1)}{m.group(2) or ''}" if m else str(s).strip()


def _row_overlaps(row: dict, lo: float, hi: float) -> bool:
    """Whether any actually-played GT interval overlaps ``[lo, hi]``."""
    return any(a < hi and b > lo for a, b in _audible_intervals(row))


def _row_active_at(row: dict, t: float) -> bool:
    return any(a <= t < b for a, b in _audible_intervals(row))


def _sample_played_times(row: dict, n: int = 15) -> np.ndarray:
    """Even samples over played time, excluding multisegment silent gaps."""
    intervals = _audible_intervals(row)
    lengths = [max(0.0, b - a) for a, b in intervals]
    total = sum(lengths)
    if total <= 0:
        return np.array([], dtype=float)
    offsets = np.linspace(0.0, total, n, endpoint=False) + total / (2 * n)
    out: list[float] = []
    for off in offsets:
        cursor = off
        for (a, _b), length in zip(intervals, lengths):
            if cursor < length - 1e-9:
                out.append(a + cursor)
                break
            cursor -= length
    return np.asarray(out, dtype=float)


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


def score_spans(
    set_id: str,
    timeline_path: Path,
    *,
    fibers: bool = False,
    hubert_layer: int = 9,
    gt_path: Path | None = None,
) -> list[SpanScore]:
    """Score every span in *timeline_path* against GT; return one SpanScore per span.

    Spans with no same-recording GT row are emitted with stem/span_class/
    place_err_s/strict/fiber/ref_err_s/density=None.  id_correct is set from
    the overlapping-GT block when present, else None.
    """
    # --- GT resolution (mirrors main() setup) ---
    if gt_path is None:
        fixtures = sorted((_REPO / "labeling" / "fixtures").glob("*_ground_truth.yaml"))
        matches = [
            f for f in fixtures if yaml.safe_load(f.read_text()).get("set_id") == set_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"--gt not given and {len(matches)} GT fixtures match "
                f"set_id={set_id} (looked in labeling/fixtures/)"
            )
        gt_path = matches[0]

    tl_record = load_timeline(timeline_path)
    join_guard(tl_record.sid, set_id, context="timeline vs set_id")
    timeline = json.loads(Path(timeline_path).read_text())

    # manifest by track_id — only needed for fiber ref-audio resolution
    by_tid: dict[str, dict] = {}
    if fibers:
        set_dir = find_aligning_dir(set_id)
        manifest = load_manifest(set_dir / "manifest.json")
        join_guard(manifest.sid, tl_record.sid, context="manifest vs timeline")
        by_tid = {
            k: {"track_id": r.track_id, "stems": r.stems, "local_path": r.local_path}
            for k, r in manifest.by_track_id().items()
        }

    gt_doc = yaml.safe_load(gt_path.read_text())
    join_guard(tl_record.sid, gt_doc.get("set_id") or "", context="timeline vs GT yaml")
    gt_rows = [
        r
        for r in canonicalize_gt_rows(set_id, gt_doc["tracks"])
        if str(r.get("slot_label")) != "mix"
    ]

    gt_by_tid: dict[str, list[dict]] = {}
    for r in gt_rows:
        if r.get("track_id"):
            gt_by_tid.setdefault(str(r["track_id"]), []).append(r)

    fiber_cache: dict[str, tuple] = {}

    def _fibers_for(ref_audio: str | None):
        if not fibers or ref_audio is None:
            return None
        if ref_audio not in fiber_cache:
            from alignment.path_decode import _ensure_feat
            from alignment.ref_fibers import compute_fibers

            hf = np.load(_ensure_feat(ref_audio, ref_audio, "hubert", hubert_layer))
            fiber_cache[ref_audio] = compute_fibers(hf, FPS, audio_path=ref_audio)
        return fiber_cache[ref_audio]

    # --- per-span loop ---
    results: list[SpanScore] = []
    for s in timeline["spans"]:
        slot = norm_slot(s["slot_label"])
        recording_id: str = s["recording_id"]

        # identity: any GT row overlapping the predicted span in time?
        overlapping = [
            r
            for r in gt_rows
            if r.get("track_id")
            and _row_overlaps(
                r, float(s["set_start_s"]) - 5.0, float(s["set_end_s"]) + 5.0
            )
        ]
        if overlapping:
            id_correct: bool | None = any(
                str(r["track_id"]) == recording_id for r in overlapping
            )
        else:
            id_correct = None

        # placement + ref: nearest same-recording GT row
        rows = gt_by_tid.get(recording_id)
        if not rows:
            results.append(
                SpanScore(
                    slot=slot,
                    recording_id=recording_id,
                    stem=None,
                    span_class=None,
                    id_correct=id_correct,
                    place_err_s=None,
                    strict=None,
                    fiber=None,
                    ref_err_s=None,
                    density=None,
                )
            )
            continue

        g = min(rows, key=lambda r: abs(float(r["set_start_s"]) - s["set_start_s"]))
        gstem = g.get("claimed_stem") or s.get("claimed_stem") or "regular"
        # Placement is measured against the AUDIBLE entry, not the clip extent: a
        # track that fades in under a crossfade is placed silently at set_start
        # but enters at audible_start_s, and the aligner locks the audible entry.
        place_err_s = abs(gt_placement_onset(g) - s["set_start_s"])

        structure_abstained = bool(s.get("structure_abstained"))
        if structure_abstained:
            strict = None
            fiber_val = None
        else:
            fib = _fibers_for(
                _resolve_ref_audio(s, by_tid.get(recording_id), stem=gstem)
            )
            strict, _npred, facc = trajectory_acc(
                _pred_segs_from_span(s, anchor_s=float(g["set_start_s"])),
                g,
                fiber=fib,
            )
            fiber_val = facc if fibers else strict

        # overlay density
        density_times = _sample_played_times(g)
        density = (
            int(
                np.median(
                    [
                        sum(1 for r in gt_rows if _row_active_at(r, float(t)))
                        for t in density_times
                    ]
                )
            )
            if density_times.size
            else 0
        )

        # ref_err_s: straight clips only (same exclusion logic as main())
        ref_err_s: float | None = None
        if not structure_abstained and not (
            g.get("is_loop") or g.get("ref_segments")
        ):
            ratio = float(g.get("tempo_ratio") or 1.0)
            if 0.9 <= ratio <= 1.15:
                expected = (
                    float(g["ref_start_s"])
                    + (s["set_start_s"] - float(g["set_start_s"])) * ratio
                )
                ref_err_s = abs(s["ref_start_s"] - expected)

        results.append(
            SpanScore(
                slot=slot,
                recording_id=recording_id,
                stem=gstem,
                span_class=_span_class(g),
                id_correct=id_correct,
                place_err_s=place_err_s,
                strict=strict,
                fiber=fiber_val,
                ref_err_s=ref_err_s,
                density=density,
            )
        )

    return results


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
    p.add_argument(
        "--axis-belief-bundle",
        type=Path,
        default=None,
        help="DEFAULT OFF: decode calibrated PLACEMENT/STRUCTURE beliefs into an "
        "evaluation-only timeline before running the canonical scorer",
    )
    p.add_argument(
        "--axis-belief-eval-output",
        type=Path,
        default=None,
        help="path for the generated evaluation timeline (requires "
        "--axis-belief-bundle; default: <timeline>.axis-beliefs-eval.json)",
    )
    p.add_argument(
        "--emit-never-matched",
        type=str,
        default=None,
        help="write never-matched GT recordings to this JSON path",
    )
    p.add_argument(
        "--strict-inventory",
        action="store_true",
        help="abort when GT rows lack resolvable ref audio (excluding unalignable)",
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
    if args.axis_belief_eval_output is not None and args.axis_belief_bundle is None:
        p.error("--axis-belief-eval-output requires --axis-belief-bundle")
    if args.axis_belief_bundle is not None:
        from alignment.belief_timeline import (
            prepare_belief_evaluation_timeline,
        )

        output = args.axis_belief_eval_output or tl_path.with_name(
            f"{tl_path.stem}.axis-beliefs-eval.json"
        )
        coverage = prepare_belief_evaluation_timeline(
            tl_path,
            args.axis_belief_bundle,
            output,
            expected_set_id=args.set_id,
        )
        tl_path = output
        print(
            "(axis beliefs: "
            f"placement {coverage['placement_accepted']}/"
            f"{coverage['source_spans']} accepted; "
            f"structure abstained "
            f"{coverage['structure_abstained_among_placement_accepted']}; "
            f"evaluation timeline {output})"
        )

    # Print id-map normalization count before score_spans (mirrors old behaviour).
    id_map_path = _REPO / "labeling" / "fixtures" / "id_maps" / f"{args.set_id}.json"
    id_map = load_identity_map(args.set_id)
    if id_map:
        gt_doc_pre = yaml.safe_load(args.gt.read_text())
        gt_rows_pre = [
            r for r in gt_doc_pre["tracks"] if str(r.get("slot_label")) != "mix"
        ]
        mapped = sum(
            1
            for r in gt_rows_pre
            if (tid := str(r.get("track_id") or ""))
            and tid in id_map
            and id_map[tid] != tid
        )
        if mapped or id_map:
            print(f"(id map: {mapped} GT ids normalized via {id_map_path.name})")

    gt_doc_pre = yaml.safe_load(args.gt.read_text())
    gt_rows_preflight = [
        r for r in gt_doc_pre["tracks"] if str(r.get("slot_label")) != "mix"
    ]
    if run_audio_preflight(
        args.set_id, gt_rows_preflight, strict=args.strict_inventory
    ):
        return 1

    span_scores = score_spans(
        args.set_id,
        tl_path,
        fibers=args.fibers,
        hubert_layer=args.hubert_layer,
        gt_path=args.gt,
    )

    # Reconstruct aggregates from span_scores + raw timeline (display only).
    timeline = json.loads(Path(tl_path).read_text())
    spans = timeline["spans"]

    # Reload GT rows (with id-map applied) for decompose + identity-miss display.
    gt_doc = yaml.safe_load(args.gt.read_text())
    gt_rows = [
        r
        for r in canonicalize_gt_rows(args.set_id, gt_doc["tracks"])
        if str(r.get("slot_label")) != "mix"
    ]
    gt_by_tid2: dict[str, list[dict]] = {}
    for r in gt_rows:
        if r.get("track_id"):
            gt_by_tid2.setdefault(str(r["track_id"]), []).append(r)

    # Identity aggregates
    id_ok = sum(1 for sc in span_scores if sc.id_correct is True)
    id_bad = []
    for sc, s in zip(span_scores, spans):
        if sc.id_correct is False:
            overlapping_tids = sorted(
                {
                    str(r["track_id"])
                    for r in gt_rows
                    if r.get("track_id")
                    and _row_overlaps(
                        r,
                        float(s["set_start_s"]) - 5.0,
                        float(s["set_end_s"]) + 5.0,
                    )
                }
            )[:3]
            id_bad.append((sc.slot, sc.recording_id, overlapping_tids, s["name"][:36]))
    no_gt = sum(1 for sc in span_scores if sc.place_err_s is None)

    # Placement + ref aggregates (for stats and worst lists)
    place_errs = []
    ref_rows = []
    loops_hit = 0
    for sc, s in zip(span_scores, spans):
        if sc.place_err_s is None:
            continue
        place_errs.append(
            (
                sc.place_err_s,
                sc.slot,
                s["set_start_s"],
                s["set_start_s"] - sc.place_err_s
                if s["set_start_s"] >= sc.place_err_s
                else s["set_start_s"] + sc.place_err_s,
                s["name"][:36],
            )
        )
        if s.get("structure_abstained"):
            continue
        # Recover the actual GT set_start for worst-placement display
        rows = gt_by_tid2.get(sc.recording_id)
        if rows:
            g = min(rows, key=lambda r: abs(float(r["set_start_s"]) - s["set_start_s"]))
            place_errs[-1] = (
                sc.place_err_s,
                sc.slot,
                s["set_start_s"],
                float(g["set_start_s"]),
                s["name"][:36],
            )
            if sc.ref_err_s is None:
                # loop/segment or out-of-ratio → count as loops_hit
                if g.get("is_loop") or g.get("ref_segments"):
                    loops_hit += 1
                else:
                    ratio = float(g.get("tempo_ratio") or 1.0)
                    if not (0.9 <= ratio <= 1.15):
                        loops_hit += 1
            else:
                # straight clip — reconstruct display tuple
                ratio = float(g.get("tempo_ratio") or 1.0)
                expected = (
                    float(g["ref_start_s"])
                    + (s["set_start_s"] - float(g["set_start_s"])) * ratio
                )
                ref_rows.append(
                    (
                        sc.ref_err_s,
                        sc.slot,
                        sc.stem,
                        s["ref_start_s"],
                        expected,
                        s["name"][:36],
                    )
                )

    # Trajectory list: (span_class, stem, strict, facc, density) for spans with GT
    traj = [
        (sc.span_class, sc.stem, sc.strict, sc.fiber, sc.density)
        for sc in span_scores
        if sc.strict is not None
    ]

    # Decompose pass (separate from scoring; uses raw span + GT data)
    decomp: list[tuple] = []
    if args.decompose:
        for sc, s in zip(span_scores, spans):
            if sc.strict is None:
                continue
            rows = gt_by_tid2.get(sc.recording_id)
            if not rows:
                continue
            g = min(rows, key=lambda r: abs(float(r["set_start_s"]) - s["set_start_s"]))
            nt, no, nk = _decompose_span(
                _pred_segs_from_span(s, anchor_s=float(g["set_start_s"])), g
            )
            decomp.append((sc.stem, nt, no, nk, sc.strict))

    # --- Print section (identical to old main) ---
    n = len(spans)
    print(f"=== end-to-end pipeline vs GT ({args.set_id}, {n} predicted spans) ===")
    nid = id_ok + len(id_bad)
    # HONEST identity headline: appearance-recall over the ADJUDICABLE (bound) GT.
    # The de-poisoned GT abstains on a large fraction of appearances (no track_id);
    # the per-span number below is prediction-centric and DEFLATED by that abstain
    # (a correct prediction on an abstained appearance scores as a miss). Recall
    # over bound appearances excludes the unadjudicable ones instead of failing them.
    idr = identity_recall(gt_rows, spans)
    print(
        f"identity (HONEST — recall over adjudicable GT): "
        f"{idr['hits']}/{idr['bound']} ({100 * idr['recall']:.0f}%)  "
        f"| adjudicable {100 * idr['adjudicable_frac']:.0f}% "
        f"({idr['abstain']} appearances abstain — UNMEASURABLE, not scored)"
    )
    print(
        f"identity (per-span, DEFLATED by abstain — diagnostic only): "
        f"{id_ok}/{nid} ({100 * id_ok / max(nid, 1):.0f}%)  "
        f"[{no_gt} spans had no same-slot GT row]"
    )
    # Form-level coverage: GT rows no predicted span selects (nearest-set_start)
    # are unscored by every per-span metric below. A slot played across a stem
    # transition (instrumental->full, acappella over instrumental) has >1 GT row
    # per recording; the per-span scorer picks one, silently dropping the rest.
    gt_forms_tot = sum(1 for r in gt_rows if r.get("track_id"))
    unmatched_forms = unmatched_gt_forms(gt_rows, spans)
    if unmatched_forms:
        lost_s = sum(f["set_end_s"] - f["set_start_s"] for f in unmatched_forms)
        by_stem = Counter(f["claimed_stem"] or "regular" for f in unmatched_forms)
        breakdown = ", ".join(f"{k}×{v}" for k, v in sorted(by_stem.items()))
        print(
            f"GT form coverage: {gt_forms_tot - len(unmatched_forms)}/{gt_forms_tot} "
            f"forms scored; {len(unmatched_forms)} UNSCORED "
            f"({lost_s:.0f}s GT invisible to per-span metrics): {breakdown}"
        )
    pe = np.array([r[0] for r in place_errs])
    if pe.size:
        print(
            f"set placement |pred-gt|: median={np.median(pe):.1f}s  "
            f"<5s: {100 * (pe < 5).mean():.0f}%  "
            f"<15s: {100 * (pe < 15).mean():.0f}%  "
            f"p90={np.percentile(pe, 90):.1f}s  (n={len(pe)})"
        )
    else:
        print("set placement |pred-gt|: no accepted, GT-matchable placements")
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
        # overlay-density stratification: dense pileups (>=3 concurrent GT
        # layers — intro/finale medleys) are partly ill-posed to recreate;
        # report them separately so they can't silently dominate the headline.
        sparse = [r for r in traj if r[4] < 4]
        dense = [r for r in traj if r[4] >= 4]
        if dense:
            print(f"  density med<4 layers {_ta(sparse)}")
            print(f"  density med>=4       {_ta(dense)}  (sustained medley pileups)")
            nl_sp = [r for r in sparse if r[0] in ("multiseg", "loop")]
            if nl_sp:
                print(f"  HEADLINE (pileups excluded) {_ta(nl_sp)}")
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
        matched_tids = {s2["recording_id"] for s2 in spans}
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
        if getattr(args, "emit_never_matched", None):
            from alignment.never_matched import write_never_matched
            from pathlib import Path as _Path

            write_never_matched(
                args.set_id, gt_rows, spans, _Path(args.emit_never_matched)
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
