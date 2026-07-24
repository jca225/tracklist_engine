#!/usr/bin/env python3
"""`make align-state SET=<id>` — one command that answers "where is the aligner
for this set, and can I trust its timeline?".

Prints, for each timeline variant on disk: its provenance stamp, and whether it
is FRESH vs the CURRENT state (aligner code / GT / id_map / the pi `set_track_slots`
spine). Kills the "is this July-6 file stale?" fog that cost a whole session
(state-of-record #18). Pair with `make scorecard` for the numbers themselves.

Usage:
    make align-state SET=1fsnxchk          # full check (queries pi for the spine)
    make align-state SET=1fsnxchk NOPI=1   # offline (skip the pi spine check)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from workspaces.alignment_prototype import provenance  # noqa: E402

OUT = REPO / "workspaces" / "alignment_prototype" / "out"
GT = {
    "1fsnxchk": REPO / "labeling" / "fixtures" / "bb12_ground_truth.yaml",
    "2nvzlh2k": REPO / "labeling" / "fixtures" / "bb11_ground_truth.yaml",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set-id", required=True)
    ap.add_argument(
        "--no-pi", action="store_true", help="skip the pi spine check (offline)"
    )
    args = ap.parse_args(argv)
    sid = args.set_id

    rows = None
    if not args.no_pi:
        try:
            from workspaces.alignment_prototype.infer import fetch_slot_rows

            rows = fetch_slot_rows(sid)
        except Exception as e:  # pi unreachable — degrade to local-only check
            print(f"(pi spine check skipped: {e})")

    gt_paths = [GT[sid]] if sid in GT else []
    print(f"═══ align-state: {sid} ═══")
    found = False
    for suffix in ("", "_lt"):
        p = OUT / f"{sid}_predicted_timeline{suffix}.json"
        tag = suffix or "base"
        if not p.exists():
            print(f"[{tag}] (no timeline on disk)")
            continue
        found = True
        tl = json.loads(p.read_text())
        prov = tl.get("provenance") or {}
        fresh, drift = provenance.check(tl, sid, rows=rows, gt_paths=gt_paths)
        age = datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")
        print(f"[{tag}] {p.name}")
        print(f"    written_at : {prov.get('written_at', '—')}   (file mtime {age})")
        print(f"    code_sha   : {prov.get('code_sha', '—')}")
        print(f"    flags      : {json.dumps(prov.get('flags', {}))}")
        if fresh:
            print("    status     : ✅ FRESH — safe to score")
        else:
            checked = "code/GT/id_map" + ("/spine" if rows is not None else "")
            print(
                f"    status     : ⚠️  STALE ({checked}) — drifted: {', '.join(drift)}"
            )
            print("                 → re-run infer before trusting scores")
    if not found:
        print("(no timelines — run infer first)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
