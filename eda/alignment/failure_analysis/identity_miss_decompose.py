#!/usr/bin/env python3
"""Decompose identity misses on BB11/BB12 by *mechanism*, not just bucket.

Motivation: the headline "identity 83/84%" is read as a discrimination-model /
data-supply problem. This script shows it is mostly neither. For every predicted
span it asks:

  - Is the assigned recording one of the GT recordings actually playing in the
    predicted span's window?  (identity correct iff yes)
  - How many GT tracks overlap the predicted window?  (layering / segmentation)
  - How long is the predicted span?  (micro-fragment vs real)

and classifies each miss into one of four mechanisms:

  1. micro-fragment  (<5 s)                         -> boundary/window artifact
  2. SEGMENTATION    (>=5 GT overlap AND >=90 s)     -> ONE span labelled a single
        recording over a region where 5-12 GT tracks play. This is a PLACEMENT /
        span-extent failure counted as an identity miss; a better ID model does
        nothing for it.
  3. LAYER/transition (2-4 GT overlap)               -> dense mashup/transition
        moment; a single-label-per-span output structurally cannot be 100% here.
  4. TRUE discrimination (1 GT overlap)              -> clean solo region, model
        picked an unrelated song. The ONLY bucket a better identity model fixes.

It reads the `_lt` (looptrace) predicted timeline (the scorecard's own source of
truth) and the GT fixtures. No audio, no DB -> fast and deterministic.

Companion write-up: IDENTITY_MISS_DECOMPOSITION.md (same dir). The candidate-
availability half of the census (was the correct recording downloaded +
fingerprinted?) is a canonical-DB query documented there, not re-run here.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "workspaces" / "alignment_prototype" / "out"
FIX = REPO / "labeling" / "fixtures"
SETS = {"2nvzlh2k": "bb11", "1fsnxchk": "bb12"}
MICRO_S = 5.0
SEG_MIN_OVERLAP = 5
SEG_MIN_DUR_S = 90.0


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def misses_for(set_id: str, nm: str) -> list[dict]:
    tl = json.loads((OUT / f"{set_id}_predicted_timeline_lt.json").read_text())
    spans = tl if isinstance(tl, list) else tl["spans"]
    gt = yaml.safe_load((FIX / f"{nm}_ground_truth.yaml").read_text())["tracks"]
    out = []
    for s in spans:
        s0, s1 = float(s["set_start_s"]), float(s["set_end_s"])
        rec = str(s["recording_id"])
        ov = [
            (r, _overlap(s0, s1, float(r["set_start_s"]), float(r["set_end_s"])))
            for r in gt
            if r.get("track_id")
        ]
        ov = [(r, o) for r, o in ov if o > 0]
        if (
            not ov
        ):  # no overlapping GT row WITH a track_id -> id_correct is None (scorer rule)
            continue
        tids = {str(r["track_id"]) for r, _ in ov}
        if rec in tids:  # identity correct
            continue
        best = max(ov, key=lambda x: x[1])[0]
        out.append(
            dict(
                set=nm,
                slot=str(s.get("slot_label") or ""),
                dur=round(s1 - s0, 1),
                n_gt_overlap=len(ov),
                gt_rec=str(best["track_id"]),
                gt_stem=str(best.get("claimed_stem") or "?"),
                gt_track=str(best.get("track") or "")[:40],
                pred_rec=rec,
                pred_name=(s.get("name") or "")[:40],
            )
        )
    return out


def mechanism(m: dict) -> str:
    if m["dur"] < MICRO_S:
        return "1. micro-fragment (<5s)"
    if m["n_gt_overlap"] >= SEG_MIN_OVERLAP and m["dur"] >= SEG_MIN_DUR_S:
        return (
            "2. SEGMENTATION (giant span over 5-12 GT tracks; placement, not identity)"
        )
    if m["n_gt_overlap"] >= 2:
        return (
            "3. LAYER/transition (2-4 co-present tracks; single-label cannot be 100%)"
        )
    return "4. TRUE single-track discrimination miss"


def main() -> int:
    rows = [m for sid, nm in SETS.items() for m in misses_for(sid, nm)]
    cnt: Counter[str] = Counter()
    sec: dict[str, float] = defaultdict(float)
    for m in rows:
        k = mechanism(m)
        cnt[k] += 1
        sec[k] += m["dur"]
    tot = sum(m["dur"] for m in rows)
    print(
        f"\n=== identity-miss decomposition (BB11+BB12, n={len(rows)}, {tot:.0f} span-s) ==="
    )
    for k in sorted(cnt):
        print(
            f"  {cnt[k]:>2} spans | {sec[k]:>6.0f}s | {100 * sec[k] / max(tot, 1):>3.0f}% | {k}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
