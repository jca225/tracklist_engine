#!/usr/bin/env python3
"""Sweep a music recognizer (ACRCloud) across a span of a DJ mix and report
*where* songs sit — the multi-segment driver on top of recognize_segment.py.

Built for live sets whose tracklist runs out (e.g. the Murph Club Space gaps):
step a short window across [--start, --end], recognize each, then collapse
consecutive identical hits into spans so the output reads as a timeline of
"song X from t0 to t1". Recognition names the SONG, not the version — confirm by
hand (see docs/open_set_alignment_endstate.md).

Usage:
  venvs/audio/bin/python scripts/recognize_sweep.py --set-id pwgrrb1 --start 5400 --end 7200 --step 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.recognize_segment import (  # noqa: E402
    _aligning_dir, _load_env, acrcloud, extract_segment,
)


def _mmss(s: float) -> str:
    s = int(round(s))
    return f"{s // 60}:{s % 60:02d}"


def _best_hit(resp: dict) -> dict | None:
    if resp.get("status", {}).get("code") != 0:
        return None
    music = resp.get("metadata", {}).get("music") or []
    if not music:
        return None
    m = max(music, key=lambda x: x.get("score", 0))
    return {
        "artist": ", ".join(a.get("name", "") for a in m.get("artists", [])),
        "title": m.get("title", ""),
        "album": (m.get("album") or {}).get("name", ""),
        "score": m.get("score", 0),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--set-id")
    src.add_argument("--audio", type=Path)
    p.add_argument("--stem", default="mix", help="mix | mix_instrumental | mix_vocals")
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--end", type=float, required=True)
    p.add_argument("--step", type=float, default=30.0, help="seconds between probes")
    p.add_argument("--dur", type=float, default=11.0, help="probe window length")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    _load_env(_REPO / ".env")
    host = os.getenv("ACRCLOUD_IDENTIFY_HOST") or os.getenv("ACRCLOUD_HOST")
    key = os.getenv("ACRCLOUD_ACCESS_KEY")
    sec = (os.getenv("ACRCLOUD_ACCESS_SECRET") or os.getenv("ACRCLOUD_SECRET_KEY")
           or os.getenv("ACRCLOUD_SECRET"))
    if not all((host, key, sec)):
        sys.exit("ACRCloud creds missing in .env (ACRCLOUD_IDENTIFY_HOST/ACCESS_KEY/SECRET_KEY)")

    if args.audio:
        audio = args.audio
    else:
        d = _aligning_dir(args.set_id)
        cands = sorted(d.glob(f"{args.stem}.*"))
        if not cands:
            sys.exit(f"no {args.stem}.* in {d}")
        audio = cands[0]
    print(f"sweep {audio.name}  {_mmss(args.start)}–{_mmss(args.end)}  "
          f"step {args.step:.0f}s win {args.dur:.0f}s", file=sys.stderr)

    probes: list[dict] = []
    t = args.start
    while t < args.end:
        fd, tmp = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        tmp = Path(tmp)
        try:
            extract_segment(audio, t, args.dur, tmp)
            hit = None
            for attempt in range(3):
                try:
                    hit = _best_hit(acrcloud(tmp.read_bytes(), host, key, sec))
                    break
                except Exception as e:  # transient network / rate-limit
                    if attempt == 2:
                        print(f"  {_mmss(t)} request failed: {e}", file=sys.stderr)
                    time.sleep(2)
        finally:
            tmp.unlink(missing_ok=True)
        label = f"{hit['artist']} - {hit['title']}" if hit else None
        probes.append({"t": t, "label": label, "score": hit["score"] if hit else 0,
                       "album": hit["album"] if hit else ""})
        mark = f"{label}  ({hit['score']})" if hit else "—"
        print(f"  {_mmss(t):>6}  {mark}")
        t += args.step

    # collapse consecutive identical labels into spans
    spans: list[dict] = []
    for pr in probes:
        if pr["label"] is None:
            continue
        if spans and spans[-1]["label"] == pr["label"] and pr["t"] - spans[-1]["_last"] <= args.step * 1.6:
            spans[-1]["_last"] = pr["t"]
            spans[-1]["end_s"] = pr["t"] + args.dur
            spans[-1]["max_score"] = max(spans[-1]["max_score"], pr["score"])
            spans[-1]["n"] += 1
        else:
            spans.append({"label": pr["label"], "start_s": pr["t"],
                          "end_s": pr["t"] + args.dur, "_last": pr["t"],
                          "max_score": pr["score"], "n": 1})
    for s in spans:
        s.pop("_last", None)

    print("\n=== recognized timeline ===")
    if not spans:
        print("  (no confident matches — likely SoundCloud/unreleased tail; "
              "see open_set_alignment_endstate.md SoundCloud tier)")
    for s in spans:
        flag = "" if s["n"] >= 2 else "  [single-probe, weak]"
        print(f"  {_mmss(s['start_s'])}–{_mmss(s['end_s'])}  {s['label']}  "
              f"(score {s['max_score']}, {s['n']} probes){flag}")

    out = args.out or (_REPO / "workspaces/alignment_prototype/out" /
                       f"{args.set_id or audio.stem}_recognized.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"audio": str(audio), "probes": probes, "spans": spans}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
