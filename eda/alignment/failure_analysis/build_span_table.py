#!/usr/bin/env python3
"""Materialize a per-span failure table for the alignment aligner.

eda is a read-only consumer: this imports the scoring helpers from the
alignment_prototype harness (the SAME matching logic score_timeline_vs_gt uses)
and, instead of printing aggregates, emits ONE ROW PER matched GT span with its
error decomposition, so downstream analysis can ask "what characterizes a
failing span". No harness edits; no pi-storage; no stale manifest.

Sources per span (all local, all reproducible on CPU):
  - GT fields          labeling/fixtures/<set>_ground_truth.yaml
  - predictions        alignment/out/<set>_predicted_timeline_lt.json
                       (the looptrace variant — carries ref_segments, so trajectory
                        is scorable; base timeline is scalar-only)
  - id namespace map   labeling/fixtures/id_maps/<set>.json   (bridge, when present)
  - instance ambiguity alignment/looptrace/out/audit_<set>.json
                       (frac_distinct = fraction of GT seconds that are distinct-take
                        repeats — unwinnable on phonetics alone)

Usage:
    venvs/audio/bin/python -m eda.alignment.failure_analysis.build_span_table
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from core.contracts import join_guard, load_timeline

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from alignment.path_decode import (
    _span_class,
    audible_seconds,
    gap_hallucination_frac,
    gt_placement_onset,
    trajectory_acc,
)
from alignment.never_matched import assign_spans_to_forms
from alignment.score_timeline_vs_gt import (
    _pred_segs_from_span,
    norm_slot,
)

OUT_DIR = Path(__file__).resolve().parent / "out"
_ALN = _REPO / "alignment"

# (set_id, human label, GT yaml)
SETS = [
    ("1fsnxchk", "BB12", _REPO / "labeling/fixtures/bb12_ground_truth.yaml"),
    ("2nvzlh2k", "BB11", _REPO / "labeling/fixtures/bb11_ground_truth.yaml"),
]

FIELDS = [
    "set",
    "set_id",
    "slot",
    "recording_id",
    "name",
    "gt_stem",  # authoritative axis: matched GT row's claimed_stem (None->regular)
    "pred_stem",  # what the pipeline ROUTED on (materialized set_track_slots — buggy)
    "stem_mismatch",  # 1 when routing != truth (stale-materialize mis-route)
    "span_class",
    # GT features
    "gt_set_start_s",
    "gt_ref_start_s",
    "gt_duration_s",  # envelope: set_end - set_start (spans the silent gaps)
    "gt_audible_s",  # played mix-seconds (honest recall/loss denominator)
    "tempo_ratio",
    "is_loop",
    "pitch_shift_semi",
    "n_ref_segments",
    "audible_frac",
    # instance ambiguity (from audit)
    "frac_clone",
    "frac_distinct",
    "frac_unique",
    # predictions
    "pred_set_start_s",
    "pred_ref_start_s",
    "pred_n_segments",
    # errors  (the targets of the analysis)
    "identity_hit",
    "set_start_err_s",
    "ref_offset_err_s",
    "traj_strict",
    "gap_halluc",  # frac of silent-gap seconds the prediction fills (precision)
    # ref-offset scorability (straight non-odd clips only, mirrors harness)
    "ref_offset_scored",
    # RT1: 1 = synthetic row for a GT form no predicted span was assigned to
    # (form-centric recall loss; traj_strict=0, no prediction to score)
    "coverage_miss",
]


def _load_gt(gt_path: Path, set_id: str) -> tuple[list[dict], dict]:
    """GT rows (mix dropped) with tlp* ids normalized to canonical recording_id,
    plus a {recording_id -> [rows]} index. Mirrors score_timeline_vs_gt exactly."""
    gt_doc = yaml.safe_load(gt_path.read_text())
    join_guard(gt_doc.get("set_id") or "", set_id, context="GT yaml vs set_id")
    rows = [r for r in gt_doc["tracks"] if str(r.get("slot_label")) != "mix"]
    id_map_path = _REPO / "labeling" / "fixtures" / "id_maps" / f"{set_id}.json"
    id_map = json.loads(id_map_path.read_text()) if id_map_path.exists() else {}
    for r in rows:
        tid = str(r.get("track_id") or "")
        if tid in id_map and id_map[tid] != tid:
            r["track_id"] = id_map[tid]
    by_tid: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("track_id"):
            by_tid.setdefault(str(r["track_id"]), []).append(r)
    return rows, by_tid


def _load_audit(set_id: str) -> dict[str, dict]:
    """{norm_slot -> audit span dict} (frac_clone/frac_distinct/frac_unique).

    Keyed by BOTH the normalized slot and the audit row's track_id: BB12's GT
    uses plain numeric slot labels ('012') where the timeline spine has
    w-layers ('12w1'), so a slot-only join comes back empty (measured 0/83);
    the track_id key survives the namespace mismatch."""
    p = _ALN / "looptrace" / "out" / f"audit_{set_id}.json"
    if not p.is_file():
        return {}
    au = json.loads(p.read_text())
    out: dict[str, dict] = {}
    for a in au.get("spans", []):
        out[norm_slot(a["slot_label"])] = a
        if a.get("track_id"):
            out.setdefault(str(a["track_id"]), a)
    return out


def _rows_for_set(set_id: str, label: str, gt_path: Path, suffix: str) -> list[dict]:
    tl_path = _ALN / "out" / f"{set_id}_predicted_timeline{suffix}.json"
    join_guard(load_timeline(tl_path).sid, set_id, context="timeline vs set_id")
    timeline = json.loads(tl_path.read_text())

    # Freshness: shout if this timeline was produced against different code / GT /
    # id_map than the current state (the "is this file stale?" fog). Local-only
    # here — the spine (set_track_slots) needs pi, so it is not recomputed in the
    # scorecard; run `make align-state` for the full check.
    from alignment import provenance

    fresh, drift = provenance.check(timeline, set_id, gt_paths=[gt_path])
    if not fresh:
        print(
            f"  ⚠️  [{label}] timeline {tl_path.name} is STALE — drifted vs current: "
            f"{', '.join(drift)}. Scores may not reflect current code/GT/id_map. "
            f"Re-run infer."
        )

    gt_rows, gt_by_tid = _load_gt(gt_path, set_id)
    audit = _load_audit(set_id)

    # RT1: injective per-recording span->form assignment (no reuse), so a reprise
    # scores each span against its own form and unmatched forms surface as recall
    # loss instead of vanishing under the old nearest-only collapse.
    assigned, unmatched_forms = assign_spans_to_forms(gt_rows, timeline["spans"])

    out: list[dict] = []
    for s in timeline["spans"]:
        slot = norm_slot(s["slot_label"])
        # pred_stem is what the pipeline ROUTED on (materialized set_track_slots,
        # corrupted by the row-text (Instrumental)/(Acappella) drop bug). The
        # authoritative AXIS is the matched GT row's claimed_stem (set below).
        pred_stem = s.get("claimed_stem") or "regular"

        # identity: does any GT row overlapping the predicted span (±5s) share
        # the predicted recording_id?  (harness convention)
        overlapping = [
            r
            for r in gt_rows
            if r.get("track_id")
            and float(r["set_start_s"]) < s["set_end_s"] + 5
            and float(r["set_end_s"]) > s["set_start_s"] - 5
        ]
        id_hit = ""  # blank = no GT overlap to judge against
        if overlapping:
            id_hit = int(
                any(str(r["track_id"]) == s["recording_id"] for r in overlapping)
            )

        # placement + ref + trajectory: the injectively-assigned GT form (RT1).
        # No form -> either the recording is absent from GT, or this span lost the
        # per-recording assignment to a nearer sibling span (a spurious extra
        # prediction). Both are no-GT-anchor rows.
        g = assigned.get(id(s))
        if g is None:
            out.append(
                {
                    "set": label,
                    "set_id": set_id,
                    "slot": slot,
                    "recording_id": s["recording_id"],
                    "name": s.get("name", "")[:60],
                    "gt_stem": "",  # no GT anchor -> truth unknown for this span
                    "pred_stem": pred_stem,
                    "stem_mismatch": "",
                    "span_class": "",
                    "identity_hit": id_hit,
                    "ref_offset_scored": 0,
                    "coverage_miss": 0,
                }
            )
            continue
        gset = float(g["set_start_s"])
        gdur = float(g["set_end_s"]) - gset
        span_cls = _span_class(g)
        segs = _pred_segs_from_span(s, anchor_s=gset)
        strict, _npred, _facc = trajectory_acc(segs, g, fiber=None)

        # ref-offset MAE: straight clips only, tempo in [0.9,1.15] (harness rule)
        ratio = float(g.get("tempo_ratio") or 1.0)
        scored = not (g.get("is_loop") or g.get("ref_segments")) and (
            0.9 <= ratio <= 1.15
        )
        ref_err = ""
        if scored:
            expected = float(g["ref_start_s"]) + (s["set_start_s"] - gset) * ratio
            ref_err = abs(float(s["ref_start_s"]) - expected)

        a = audit.get(slot) or audit.get(str(s["recording_id"]), {})
        gt_segs = g.get("ref_segments")
        gt_stem = g.get("claimed_stem") or "regular"  # None == host/regular track
        out.append(
            {
                "set": label,
                "set_id": set_id,
                "slot": slot,
                "recording_id": s["recording_id"],
                "name": s.get("name", "")[:60],
                "gt_stem": gt_stem,
                "pred_stem": pred_stem,
                "stem_mismatch": int(gt_stem != pred_stem),
                "span_class": span_cls,
                "gt_set_start_s": round(gset, 2),
                "gt_ref_start_s": round(float(g["ref_start_s"]), 2),
                "gt_duration_s": round(gdur, 2),
                "gt_audible_s": round(audible_seconds(g), 2),
                "tempo_ratio": round(ratio, 4),
                "is_loop": int(bool(g.get("is_loop"))),
                "pitch_shift_semi": g.get("pitch_shift_semi") or 0,
                "n_ref_segments": len(gt_segs) if gt_segs else 1,
                "audible_frac": g.get("audible_frac"),
                "frac_clone": a.get("frac_clone"),
                "frac_distinct": a.get("frac_distinct"),
                # audit rows carry clone/distinct/unique SECONDS + frac_clone/
                # frac_distinct; frac_unique is the remainder
                "frac_unique": (
                    round(1.0 - a["frac_clone"] - a["frac_distinct"], 4)
                    if a.get("frac_clone") is not None
                    and a.get("frac_distinct") is not None
                    else None
                ),
                "pred_set_start_s": round(float(s["set_start_s"]), 2),
                "pred_ref_start_s": round(float(s["ref_start_s"]), 2),
                "pred_n_segments": len(s["ref_segments"])
                if s.get("ref_segments")
                else 1,
                "identity_hit": id_hit,
                # placement vs the AUDIBLE entry (audible_start_s), not clip extent
                "set_start_err_s": round(
                    abs(gt_placement_onset(g) - float(s["set_start_s"])), 2
                ),
                "ref_offset_err_s": round(ref_err, 2) if scored else "",
                "traj_strict": round(float(strict), 4),
                "gap_halluc": round(gap_hallucination_frac(segs, g), 4),
                "ref_offset_scored": int(scored),
                "coverage_miss": 0,
            }
        )

    # RT1 coverage: emit a synthetic row per GT FORM no span was assigned to, so
    # its audible seconds re-enter the impact-weighted loss denominator as full
    # loss (traj_strict=0). This is form-level, not the old recording-level check
    # (a recording matched by SOME span could still lose a form to the collapse).
    cov_secs = 0.0
    for g in unmatched_forms:
        span_cls = _span_class(g)
        gt_segs = g.get("ref_segments")
        gset = float(g["set_start_s"])
        aud = round(audible_seconds(g), 2)
        cov_secs += aud
        out.append(
            {
                "set": label,
                "set_id": set_id,
                "slot": norm_slot(g.get("slot_label") or ""),
                "recording_id": str(g["track_id"]),
                "name": str(g.get("track", ""))[:60],
                "gt_stem": g.get("claimed_stem") or "regular",
                "pred_stem": "",  # no prediction was assigned to this form
                "stem_mismatch": "",
                "span_class": span_cls,  # in analyze's matched set -> counts loss
                "gt_set_start_s": round(gset, 2),
                "gt_ref_start_s": round(float(g.get("ref_start_s") or 0.0), 2),
                "gt_duration_s": round(float(g["set_end_s"]) - gset, 2),
                "gt_audible_s": aud,
                "tempo_ratio": round(float(g.get("tempo_ratio") or 1.0), 4),
                "is_loop": int(bool(g.get("is_loop"))),
                "pitch_shift_semi": g.get("pitch_shift_semi") or 0,
                "n_ref_segments": len(gt_segs) if gt_segs else 1,
                "audible_frac": g.get("audible_frac"),
                "identity_hit": "",  # no prediction -> nothing to judge identity on
                "set_start_err_s": "",  # no prediction -> excluded from median
                "ref_offset_err_s": "",
                "traj_strict": 0.0,  # unrecovered appearance -> full loss
                "ref_offset_scored": 0,
                "coverage_miss": 1,
            }
        )
    print(
        f"[{label}] spans={len(timeline['spans'])}  "
        f"rows_emitted={len(out)}  "
        f"unmatched GT forms (recall loss)={len(unmatched_forms)} ({cov_secs:.0f}s audible)"
    )
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--suffix",
        default="_lt",
        help="timeline variant suffix (default _lt; use _gtstem_lt for the "
        "GT-routing re-measure)",
    )
    ap.add_argument("--csv", default="span_table.csv", help="output CSV name")
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(exist_ok=True)
    all_rows: list[dict] = []
    for set_id, label, gt_path in SETS:
        all_rows.extend(_rows_for_set(set_id, label, gt_path, args.suffix))

    csv_path = OUT_DIR / args.csv
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"\nwrote {len(all_rows)} rows -> {csv_path}")

    # faithfulness cross-check: aggregate identity / placement vs the harness's
    # printed numbers so we know the per-span table reproduces score_timeline_vs_gt.
    for label in ("BB12", "BB11"):
        ids = [
            r["identity_hit"]
            for r in all_rows
            if r["set"] == label and r["identity_hit"] != ""
        ]
        # median over rows WITH a prediction (matched, non-coverage); the <15s
        # RATE counts unmatched forms as fails (RT1 signed-off metric rule).
        placed = [
            r
            for r in all_rows
            if r["set"] == label and r.get("span_class") and r.get("coverage_miss") != 1
        ]
        n_cov = sum(
            1 for r in all_rows if r["set"] == label and r.get("coverage_miss") == 1
        )
        pe = np.array([r["set_start_err_s"] for r in placed], dtype=float)
        lt15 = int((pe < 15).sum())
        rate_den = len(pe) + n_cov
        print(
            f"[{label}] xcheck: identity {sum(ids)}/{len(ids)} "
            f"({100 * sum(ids) / max(1, len(ids)):.0f}%)  "
            f"set_start median={np.median(pe):.1f}s (n={len(pe)})  "
            f"<15s={100 * lt15 / max(1, rate_den):.0f}% "
            f"(of {rate_den}, incl {n_cov} unrecovered forms)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
