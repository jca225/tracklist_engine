#!/usr/bin/env python3
"""Reference sheet from a predicted timeline — the crash-proof pre-seed fallback.

Reads <set_id>_predicted_timeline.json (preseed.py output) and writes a clean,
mix-time-sorted markdown sheet the annotator keeps open while labeling: what the
aligner thinks plays when, on which layer, from which point in the reference.
Two columns of context (bed vs overlay) so concurrent layers read naturally.

Usage:
    venvs/audio/bin/python -m workspaces.section_hsmm.preseed_sheet --set-id 2nvzlh2k
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
SEED_OUT = _REPO / "workspaces/alignment_prototype/out"


def _mmss(s: float) -> str:
    s = int(round(s))
    return f"{s // 60}:{s % 60:02d}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    tl_path = SEED_OUT / f"{args.set_id}_predicted_timeline.json"
    if not tl_path.is_file():
        sys.exit(f"no predicted timeline at {tl_path} — run preseed.py first")
    tl = json.loads(tl_path.read_text())
    spans = sorted(tl["spans"], key=lambda s: s["set_start_s"])

    lines = [
        f"# BB pre-seed reference — {args.set_id}",
        "",
        f"Aligner predictions ({len(spans)} spans). **Everything needs verifying** "
        "(blind pre-seed, no tracklist cue). `bed` = harmonic/instrumental layer, "
        "`overlay` = acappella layer. `@ref` = where in the reference track the "
        "clip starts. `[jump]` = predicted within-song section jump.",
        "",
        "| mix time | layer | track | @ref | stem | flag |",
        "|---|---|---|---|---|---|",
    ]
    for s in spans:
        flag = "section-jump" if s.get("had_section_jump") else ""
        lines.append(
            f"| {_mmss(s['set_start_s'])}–{_mmss(s['set_end_s'])} "
            f"| {s['channel']} | {s['name']} | {_mmss(s['ref_start_s'])} "
            f"| {s.get('claimed_stem','regular')} | {flag} |"
        )
    # quick per-layer tally
    bed = sum(1 for s in spans if s["channel"] == "bed")
    lines += ["", f"_Totals: {bed} bed spans, {len(spans)-bed} overlay spans._"]

    out = args.out or (SEED_OUT / f"{args.set_id}_preseed_sheet.md")
    out.write_text("\n".join(lines))
    print(f"wrote {out}  ({len(spans)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
