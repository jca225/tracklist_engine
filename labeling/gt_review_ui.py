#!/usr/bin/env python3
"""Build an interactive GT-review UI from a labeled set's .als.

Emits every field the aligner consumes per ground-truth clip — identity (from
the clip's audio FILE, the .als oracle), stem, mix span, ref span (where in the
source song), tempo ratio, pitch, loop/ref-segments, volume-aware audible
window, track_id, ref_source — into a single self-contained HTML page with:
  * a stem-coloured mix-timeline minimap,
  * a filterable/searchable clip list grouped by mashup section,
  * an inspector that A/B-plays the MIX segment vs the SOURCE segment so
    discrepancies are audible,
  * auto-flagged likely issues (mix self-reference, empty slot, no track_id,
    out-of-range tempo, skip_training),
  * per-clip discrepancy notes + flags (localStorage) and a JSON export.

Audio plays through a local server rooted at the set folder, so the page is
written THERE and served from there.

Usage:
    venvs/audio/bin/python -m labeling.gt_review_ui            # BB12 defaults
    venvs/audio/bin/python -m labeling.gt_review_ui --serve    # also launch server
    venvs/audio/bin/python -m labeling.gt_review_ui \\
        --als "<...>.als" --set-dir "~/aligning/<set>" [--serve --port 8777]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from labeling.export_als_to_gt import (
    DEFAULT_ALS,
    DEFAULT_SET_DIR,
    _placeholder_note,
    collect_kept_clip_rows,
)

_TEMPLATE = Path(__file__).resolve().parent / "gt_review" / "template.html"


def _rel_under(path: str, root: Path) -> tuple[str, bool]:
    """(path relative to set folder, is_under_root). Non-under files can't be
    served, so the UI shows their data but disables playback."""
    try:
        rel = os.path.relpath(path, root)
    except ValueError:
        return path, False
    return rel, not rel.startswith("..")


def _track_id_enricher(set_dir: Path):
    """Map a clip to its track_id the way the GT enrichment does — the raw .als
    export only resolves an EXACT manifest-path match (≈1/166), but the clip's
    SLOT maps to a manifest row (every manifest row has a track_id). Secondary:
    the already-enriched fixture (slot, mix_start) when present."""
    manifest = json.loads((set_dir / "manifest.json").read_text())
    set_id = str(manifest.get("set_id") or "").strip()
    by_slot = {str(t.get("label") or "").strip(): t.get("track_id")
               for t in manifest["tracks"] if t.get("label")}
    # find the enriched GT fixture by matching set_id (filenames aren't set-id-based)
    by_slotstart: dict[tuple, str] = {}
    import yaml
    for fix in sorted((_REPO / "labeling/fixtures").glob("*.y*ml")):
        try:
            doc = yaml.safe_load(fix.read_text()) or {}
        except Exception:
            continue
        if str(doc.get("set_id") or "").strip() != set_id:
            continue
        for r in doc.get("tracks", []):
            tid = r.get("track_id")
            if tid:
                by_slotstart[(str(r.get("slot_label")),
                              round(float(r.get("set_start_s") or -1), 1))] = tid
        break

    def enrich(row):
        if row.recording_id:
            return row.recording_id, "als-path"
        key = (row.slot_label or "", round(row.set_start_s, 1))
        if key in by_slotstart:
            return by_slotstart[key], "gt-fixture"
        tid = by_slot.get(row.slot_label or "")
        return (tid, "slot→manifest") if tid else (None, None)

    return enrich


def build_payload(als: Path, set_dir: Path) -> dict:
    set_id, rows, _ = collect_kept_clip_rows(als, set_dir)
    rows = sorted(rows, key=lambda r: r.set_start_s)
    enrich_tid = _track_id_enricher(set_dir)

    manifest = json.loads((set_dir / "manifest.json").read_text())
    mix_dur = float(manifest.get("mix_duration_s") or 0.0)
    mix_file = "mix.m4a" if (set_dir / "mix.m4a").is_file() else (
        "mix.flac" if (set_dir / "mix.flac").is_file() else "mix.m4a")

    clips = []
    for i, r in enumerate(rows):
        rel, ok = _rel_under(r.clip.path, set_dir)
        fname = Path(r.clip.path).name
        note = _placeholder_note(r.clip.path, r.clip.group_name)
        flags = ["unalignable"] if note else []
        track_id, tid_src = enrich_tid(r)
        clips.append({
            "idx": i,
            "display": r.display,
            "file": rel,
            "audio_ok": ok and (set_dir / rel).is_file(),
            "group": r.clip.group_name,
            "lane": r.clip.track_name,
            "slot": r.slot_label or "",
            "stem": r.claimed_stem,
            "ref_source": r.ref_source,
            "track_id": track_id,
            "track_id_src": tid_src,
            "mix_start_s": round(r.set_start_s, 3),
            "mix_end_s": round(r.set_end_s, 3),
            "ref_start_s": round(r.ref_start_s, 3),
            "ref_end_s": round(r.ref_end_s, 3),
            "tempo_ratio": (round(r.tempo_ratio, 5) if r.tempo_ratio is not None else None),
            "pitch_shift_semi": r.pitch_shift_semi,
            "is_loop": r.is_loop,
            "ref_segments": [
                {"mix_start_s": round(s.mix_start_s, 3),
                 "ref_start_s": round(s.ref_start_s, 3),
                 "ref_end_s": round(s.ref_end_s, 3)}
                for s in (r.ref_segments or ())
            ],
            "audible_frac": r.audible_frac,
            "audible_start_s": r.audible_start_s,
            "audible_end_s": r.audible_end_s,
            "skip_training": r.skip_training,
            "unalignable": note is not None,
            "source_note": note,
            "flags": flags,
        })

    return {
        "set_id": set_id,
        "title": set_dir.name,
        "mix_file": mix_file,
        "mix_duration_s": mix_dur,
        "clips": clips,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--als", type=Path, default=DEFAULT_ALS)
    p.add_argument("--set-dir", type=Path, default=DEFAULT_SET_DIR)
    p.add_argument("--out", type=Path, default=None,
                   help="output html (default: <set-dir>/gt_review.html)")
    p.add_argument("--serve", action="store_true", help="launch a local server + open browser")
    p.add_argument("--port", type=int, default=8777)
    args = p.parse_args(argv)

    als = args.als.expanduser()
    set_dir = args.set_dir.expanduser()
    if not als.is_file():
        print(f"not found: {als}", file=sys.stderr); return 2
    if not set_dir.is_dir():
        print(f"not found: {set_dir}", file=sys.stderr); return 2

    payload = build_payload(als, set_dir)
    html = _TEMPLATE.read_text()
    html = html.replace("__TITLE__", payload["title"]).replace(
        "__GT_DATA__", json.dumps(payload))
    out = args.out or (set_dir / "gt_review.html")
    out.write_text(html)
    n = len(payload["clips"])
    flagged = sum(1 for c in payload["clips"] if c["flags"])
    print(f"wrote {out}  ({n} GT clips, {flagged} mix-self-ref auto-flagged)")

    if not args.serve:
        print("\nopen it with audio enabled (served from the set folder):")
        print(f"  cd {json.dumps(str(set_dir))} && python3 -m http.server {args.port}")
        print(f"  then open http://localhost:{args.port}/{out.name}")
        return 0

    import http.server, socketserver, threading, webbrowser, functools
    os.chdir(set_dir)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(set_dir))
    httpd = socketserver.TCPServer(("127.0.0.1", args.port), handler)
    url = f"http://localhost:{args.port}/{out.name}"
    print(f"serving {set_dir} at {url}  (Ctrl-C to stop)")
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
