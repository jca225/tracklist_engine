#!/usr/bin/env python3
"""GT completeness & coverage: what is labeled, what is left as 'original mix'.

Answers two questions for the abstention design:
  1. Are there plays with no findable audio (missing track_id / ref_source)?
  2. What fraction of the mix timeline has NO annotation — the regions left as
     the original mix because they were too hard to label? Those are exactly
     where the aligner should abstain rather than predict.

Usage:
    venvs/audio/bin/python -m workspaces.section_hsmm.gt_coverage [--set-id 1fsnxchk]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", default="1fsnxchk")
    p.add_argument("--gt", type=Path,
                   default=_REPO / "labeling/fixtures/bb12_ground_truth.yaml")
    p.add_argument("--min-gap", type=float, default=5.0,
                   help="report uncovered regions longer than this (s)")
    args = p.parse_args(argv)

    all_rows = yaml.safe_load(args.gt.read_text())["tracks"]
    rows = [r for r in all_rows if str(r.get("slot_label")) != "mix"]

    # mix duration from the aligning manifest
    from workspaces.alignment_prototype.refine_ref_offsets import find_aligning_dir
    manifest = json.loads((find_aligning_dir(args.set_id) / "manifest.json").read_text())
    mix_dur = float(manifest.get("mix_duration_s") or 0.0)

    # --- completeness ---
    no_tid = [r for r in rows if not r.get("track_id")]
    unalign = [r for r in rows if r.get("unalignable")]
    skip = [r for r in rows if r.get("skip_training")]
    notes = [r for r in rows if r.get("source_note")]
    src: dict[str, int] = {}
    for r in rows:
        src[r.get("ref_source") or "(none)"] = src.get(r.get("ref_source") or "(none)", 0) + 1

    print(f"=== GT completeness ({args.set_id}, {len(rows)} plays, mix {mix_dur:.0f}s) ===")
    print(f"plays with NO track_id (unfindable): {len(no_tid)}")
    print(f"flagged unalignable: {len(unalign)}  | skip_training: {len(skip)}  "
          f"| with source_note: {len(notes)}")
    print(f"ref_source: {dict(sorted(src.items(), key=lambda x: -x[1]))}")
    if no_tid:
        for r in no_tid[:10]:
            print(f"   no-tid: slot={r.get('slot_label')} {str(r.get('track',''))[:50]}")
    if notes:
        for r in notes[:8]:
            print(f"   note[{r.get('slot_label')}]: {str(r.get('source_note'))[:80]}")

    # --- timeline coverage (union of labeled spans) ---
    spans = sorted((float(r["set_start_s"]), float(r["set_end_s"])) for r in rows
                   if r.get("set_start_s") is not None and r.get("set_end_s") is not None)
    merged: list[list[float]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1] + 0.01:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    covered = sum(b - a for a, b in merged)
    gaps = []
    prev = 0.0
    for a, b in merged:
        if a - prev >= args.min_gap:
            gaps.append((prev, a, a - prev))
        prev = b
    if mix_dur - prev >= args.min_gap:
        gaps.append((prev, mix_dur, mix_dur - prev))

    print(f"\n=== timeline coverage ===")
    print(f"labeled (union of spans): {covered:.0f}s of {mix_dur:.0f}s "
          f"= {100*covered/max(mix_dur,1):.0f}%")
    gap_tot = sum(g[2] for g in gaps)
    print(f"UNLABELED regions > {args.min_gap:.0f}s: {len(gaps)}  "
          f"totaling {gap_tot:.0f}s ({100*gap_tot/max(mix_dur,1):.0f}% of mix) "
          f"-> abstain targets")
    for a, b, d in sorted(gaps, key=lambda g: -g[2])[:12]:
        print(f"   gap {d:6.0f}s   [{a:7.0f} - {b:7.0f}]  ({a/60:.1f}-{b/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
