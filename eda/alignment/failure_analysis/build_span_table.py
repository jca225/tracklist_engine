#!/usr/bin/env python3
"""Materialize a per-span failure table for the alignment aligner.

eda is a read-only consumer: this imports the scoring helpers from the
alignment_prototype harness (the SAME matching logic score_timeline_vs_gt uses)
and, instead of printing aggregates, emits ONE ROW PER matched GT span with its
error decomposition, so downstream analysis can ask "what characterizes a
failing span". No harness edits; no pi-storage; no stale manifest.

Sources per span (all local, all reproducible on CPU):
  - GT fields          labeling/fixtures/<set>_ground_truth.yaml
  - predictions        workspaces/alignment_prototype/out/<set>_predicted_timeline_lt.json
                       (the looptrace variant — carries ref_segments, so trajectory
                        is scorable; base timeline is scalar-only)
  - id namespace map   labeling/fixtures/id_maps/<set>.json   (bridge, when present)
  - instance ambiguity workspaces/alignment_prototype/looptrace/out/audit_<set>.json
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

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.alignment_prototype.path_decode import _span_class, trajectory_acc
from workspaces.alignment_prototype.score_timeline_vs_gt import (
    _pred_segs_from_span,
    norm_slot,
)

OUT_DIR = Path(__file__).resolve().parent / "out"
_ALN = _REPO / "workspaces" / "alignment_prototype"

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
    "gt_duration_s",
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
    # ref-offset scorability (straight non-odd clips only, mirrors harness)
    "ref_offset_scored",
]


def _load_gt(gt_path: Path, set_id: str) -> tuple[list[dict], dict]:
    """GT rows (mix dropped) with tlp* ids normalized to canonical recording_id,
    plus a {recording_id -> [rows]} index. Mirrors score_timeline_vs_gt exactly."""
    rows = [
        r
        for r in yaml.safe_load(gt_path.read_text())["tracks"]
        if str(r.get("slot_label")) != "mix"
    ]
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
    """{norm_slot -> audit span dict} (frac_clone/frac_distinct/frac_unique)."""
    p = _ALN / "looptrace" / "out" / f"audit_{set_id}.json"
    if not p.is_file():
        return {}
    au = json.loads(p.read_text())
    out: dict[str, dict] = {}
    for a in au.get("spans", []):
        out[norm_slot(a["slot_label"])] = a
    return out


def _rows_for_set(set_id: str, label: str, gt_path: Path, suffix: str) -> list[dict]:
    tl_path = _ALN / "out" / f"{set_id}_predicted_timeline{suffix}.json"
    timeline = json.loads(tl_path.read_text())
    gt_rows, gt_by_tid = _load_gt(gt_path, set_id)
    audit = _load_audit(set_id)

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

        # placement + ref + trajectory: nearest same-recording GT row
        rows = gt_by_tid.get(s["recording_id"])
        if not rows:
            # predicted a recording GT never places at this time -> no GT anchor.
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
                }
            )
            continue
        g = min(rows, key=lambda r: abs(float(r["set_start_s"]) - s["set_start_s"]))
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

        a = audit.get(slot, {})
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
                "tempo_ratio": round(ratio, 4),
                "is_loop": int(bool(g.get("is_loop"))),
                "pitch_shift_semi": g.get("pitch_shift_semi") or 0,
                "n_ref_segments": len(gt_segs) if gt_segs else 1,
                "audible_frac": g.get("audible_frac"),
                "frac_clone": a.get("frac_clone"),
                "frac_distinct": a.get("frac_distinct"),
                "frac_unique": a.get("frac_unique"),
                "pred_set_start_s": round(float(s["set_start_s"]), 2),
                "pred_ref_start_s": round(float(s["ref_start_s"]), 2),
                "pred_n_segments": len(s["ref_segments"])
                if s.get("ref_segments")
                else 1,
                "identity_hit": id_hit,
                "set_start_err_s": round(abs(gset - float(s["set_start_s"])), 2),
                "ref_offset_err_s": round(ref_err, 2) if scored else "",
                "traj_strict": round(float(strict), 4),
                "ref_offset_scored": int(scored),
            }
        )

    # coverage: GT recordings NEVER matched by any predicted span (upstream loss)
    matched = {s["recording_id"] for s in timeline["spans"]}
    never = [
        r for r in gt_rows if r.get("track_id") and str(r["track_id"]) not in matched
    ]
    print(
        f"[{label}] spans={len(timeline['spans'])}  "
        f"rows_emitted={len(out)}  "
        f"GT recordings never matched (upstream/coverage loss)={len({str(r['track_id']) for r in never})}"
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
        rs = [r for r in all_rows if r["set"] == label and r.get("span_class")]
        ids = [
            r["identity_hit"]
            for r in all_rows
            if r["set"] == label and r["identity_hit"] != ""
        ]
        pe = np.array([r["set_start_err_s"] for r in rs])
        print(
            f"[{label}] xcheck: identity {sum(ids)}/{len(ids)} "
            f"({100 * sum(ids) / max(1, len(ids)):.0f}%)  "
            f"set_start median={np.median(pe):.1f}s <15s={100 * (pe < 15).mean():.0f}%"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
