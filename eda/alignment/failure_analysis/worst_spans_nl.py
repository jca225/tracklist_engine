#!/usr/bin/env python3
"""Render the worst aligner spans as plain-English debugging notes.

Reads the failure-analysis span_table.csv (one row per GT span, with the
aligner's prediction joined in) and writes a human-readable report: for each
of the worst spans (ranked by GT-seconds lost), a sentence describing where it
should play, what the aligner got right/wrong, and a first-guess cause — so a
human can open the mix at that timestamp and determine why by ear.

    venvs/audio/bin/python -m eda.alignment.failure_analysis.worst_spans_nl [--top 12]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
SPAN_TABLE = _REPO / "eda/alignment/failure_analysis/out/span_table.csv"
OUT = Path.home() / "aligning" / "WORST_SPANS.md"


def _f(x: str) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def mmss(s: float | None) -> str:
    if s is None:
        return "??:??"
    s = int(round(s))
    return f"{s // 60}:{s % 60:02d}"


def sec_lost(r: dict) -> float:
    # Played seconds (gt_audible_s) not the gap-inflated envelope; fall back to
    # gt_duration_s for older span-tables without the column.
    dur = _f(r.get("gt_audible_s")) if r.get("gt_audible_s") not in (None, "") else None
    if dur is None:
        dur = _f(r["gt_duration_s"])
    tj = _f(r["traj_strict"])
    if dur is None or tj is None:
        return -1.0
    return dur * (1 - tj)


def diagnose(r: dict) -> str:
    idhit = r.get("identity_hit") == "1"
    sserr = _f(r.get("set_start_err_s"))
    tj = _f(r.get("traj_strict")) or 0.0
    stem_mm = r.get("stem_mismatch") == "1"
    if not idhit:
        return (
            "WRONG/NO IDENTITY — the aligner did not match this to the right "
            "recording. Decoding can't help until identity is fixed. "
            "Listen: is the song even audible here, or buried under a louder layer?"
        )
    if stem_mm:
        return (
            f"STEM AXIS WRONG — GT says {r['gt_stem']}, aligner said "
            f"{r['pred_stem']}. Right song, wrong vocal/instrumental form."
        )
    if sserr is not None and sserr > 10:
        return (
            f"PLACEMENT — found the song but put its start {mmss(sserr)} away "
            "from where it actually enters. Wrong spot in the mix."
        )
    if tj < 0.3:
        loopnote = " It's a LOOP." if r.get("is_loop") == "1" else ""
        return (
            "TRAJECTORY — found the song AND roughly the right entry, but "
            "couldn't track how it moves through the mix (jumps / re-pitch / "
            f"tempo-warp / segment order).{loopnote} This is decode, not placement."
        )
    return "PARTIAL — mostly right; small residual error."


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=12)
    args = p.parse_args(argv)

    rows = [r for r in csv.DictReader(SPAN_TABLE.open()) if sec_lost(r) >= 0]
    by_set: dict[str, list[dict]] = {}
    for r in rows:
        by_set.setdefault(r["set"], []).append(r)

    lines = ["# Worst aligner spans — manual debug notes", ""]
    lines.append(
        "Ranked by GT-seconds lost (`gt_duration x (1 - traj_strict)`). "
        "Open the mix at the **GT time** and listen for the cause.\n"
    )
    for setname in sorted(by_set):
        spans = sorted(by_set[setname], key=sec_lost, reverse=True)[: args.top]
        lines.append(f"## {setname}\n")
        for i, r in enumerate(spans, 1):
            gt0 = _f(r["gt_set_start_s"])
            dur = _f(r["gt_duration_s"])
            tj = (_f(r["traj_strict"]) or 0.0) * 100
            span_cls = r.get("span_class", "?")
            lines.append(f"**{i}. {r['name']}** ({r['gt_stem']}) — slot {r['slot']}")
            lines.append(
                f"- Should play: **{mmss(gt0)}** for **{mmss(dur)}** "
                f"(lost {sec_lost(r):.0f}s; we scored {tj:.0f}% of it; shape={span_cls})"
            )
            lines.append(f"- Why: {diagnose(r)}")
            lines.append("")
    OUT.write_text("\n".join(lines))
    print(
        f"wrote {OUT}  ({sum(len(v) for v in by_set.values())} spans, top {args.top}/set)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
