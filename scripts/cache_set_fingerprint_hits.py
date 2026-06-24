#!/usr/bin/env python3
"""Scan a set mix for landmark fingerprint hits and persist ``set_fingerprint_hits``.

For each tracklist slot with a ``recording_id``, slide a window in a band
around the scraped cue (or coarse timeline start) and write peaked matches to:

  * local JSON cache: ``workspaces/alignment_prototype/.cache/set_fp_hits/<set_id>.json``
  * canonical DB ``set_fingerprint_hits`` (unless ``--no-push-pi``)

Requires reference fingerprints in ``fp_index`` cache (run
``scripts/backfill_track_fingerprints.py`` first) and an aligning folder on
the Mac (``~/aligning/<set_id>__*`` with mix + manifest).

    venvs/audio/bin/python scripts/cache_set_fingerprint_hits.py --set-id 1fsnxchk
    venvs/audio/bin/python scripts/cache_set_fingerprint_hits.py --set-id 1fsnxchk --dry-run
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from workspaces.alignment_prototype.fp_index import DEFAULT_CACHE_DIR, FpKey, load
from workspaces.alignment_prototype.fp_placement_refine import find_aligning_dir
from workspaces.alignment_prototype.mix_fp_hits import (
    MixFpHit,
    load_mix_mono,
    scan_band,
)

PI_HOST = "pi-storage"
CANONICAL_DB = "/mnt/storage/data/db/music_database.db"
DEFAULT_HITS_CACHE = (
    REPO / "workspaces" / "alignment_prototype" / ".cache" / "set_fp_hits"
)

_WIN_S = 12.0
_STEP_S = 2.0
_DEFAULT_BAND_S = 90.0


def ssh_sql(sql: str) -> str:
    cmd = f'sqlite3 -separator "|" {CANONICAL_DB} "{sql}"'
    r = subprocess.run(
        ["ssh", PI_HOST, cmd], capture_output=True, text=True, check=True
    )
    return r.stdout.strip()


def fetch_slots(set_id: str) -> tuple[dict, ...]:
    sql = (
        "SELECT COALESCE(recording_id, track_id), COALESCE(claimed_stem,'regular'), "
        "COALESCE(cue_seconds, cue_time_seconds, ''), row_index "
        f"FROM set_track_slots WHERE set_id='{set_id}' ORDER BY row_index"
    )
    rows: list[dict] = []
    for ln in ssh_sql(sql).splitlines():
        if not ln.strip():
            continue
        rid, stem, cue, _idx = ln.split("|", 3)
        if not rid:
            continue
        cue_s = float(cue) if cue else None
        rows.append({"recording_id": rid, "claimed_stem": stem, "cue_s": cue_s})
    return tuple(rows)


def push_hits_to_pi(set_id: str, hits: tuple[MixFpHit, ...]) -> None:
    if not hits:
        return
    payload = [
        {
            "mix_start_s": h.mix_start_s,
            "mix_end_s": h.mix_end_s,
            "matched_track_id": h.recording_id,
            "matched_stem": h.stem,
            "score": h.score,
        }
        for h in hits
    ]
    b64 = base64.b64encode(json.dumps(payload).encode()).decode("ascii")
    py = f"""
import base64, json, sqlite3
set_id = {set_id!r}
hits = json.loads(base64.b64decode({b64!r}).decode())
conn = sqlite3.connect({CANONICAL_DB!r})
conn.execute('PRAGMA foreign_keys=ON')
conn.execute('DELETE FROM set_fingerprint_hits WHERE set_id=?', (set_id,))
for h in hits:
    conn.execute(
        '''INSERT INTO set_fingerprint_hits
           (set_id, mix_start_s, mix_end_s, matched_track_id, matched_stem, score)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(set_id, mix_start_s, matched_track_id, matched_stem) DO UPDATE SET
             mix_end_s=excluded.mix_end_s,
             score=excluded.score,
             detected_at=CURRENT_TIMESTAMP''',
        (set_id, h['mix_start_s'], h['mix_end_s'], h['matched_track_id'],
         h['matched_stem'], h['score']),
    )
conn.commit()
print(len(hits))
"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(py)
        tmp = fh.name
    remote = f"/tmp/cache_set_fp_{set_id[:12]}.py"
    subprocess.run(["scp", tmp, f"{PI_HOST}:{remote}"], check=True)
    subprocess.run(
        ["ssh", PI_HOST, f"~/tracklist_engine/venvs/audio/bin/python {remote}"],
        check=True,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", required=True)
    p.add_argument("--band-s", type=float, default=_DEFAULT_BAND_S)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--hits-cache", type=Path, default=DEFAULT_HITS_CACHE)
    p.add_argument("--no-push-pi", action="store_true")
    args = p.parse_args(argv)

    align = find_aligning_dir(args.set_id)
    if align is None:
        print(f"no aligning folder for {args.set_id}", file=sys.stderr)
        return 1
    mix_path = None
    for name in ("mix.m4a", "mix_instrumental.flac", "mix.flac"):
        cand = align / name
        if cand.is_file():
            mix_path = cand
            break
    if mix_path is None:
        print(f"no mix audio in {align}", file=sys.stderr)
        return 1

    slots = fetch_slots(args.set_id)
    if not slots:
        print(f"no slots for {args.set_id}", file=sys.stderr)
        return 1

    print(
        f"set={args.set_id} slots={len(slots)} mix={mix_path.name} band=±{args.band_s:.0f}s"
    )
    if args.dry_run:
        for s in slots[:8]:
            anchor = s["cue_s"]
            print(f"  {s['recording_id']}/{s['claimed_stem']} cue={anchor}")
        if len(slots) > 8:
            print(f"  ... +{len(slots) - 8} more")
        return 0

    mix_y = load_mix_mono(mix_path)
    all_hits: list[MixFpHit] = []
    seen: set[tuple[str, str]] = set()
    for slot in slots:
        rid = slot["recording_id"]
        stem = slot["claimed_stem"]
        key = (rid, stem)
        if key in seen:
            continue
        seen.add(key)
        ref_fp = load(FpKey(rid, stem), cache_dir=args.cache_dir)
        if ref_fp is None:
            continue
        anchor = slot["cue_s"]
        if anchor is None:
            continue
        lo = anchor - args.band_s
        hi = anchor + args.band_s
        hits = scan_band(
            mix_y,
            ref_fp=ref_fp,
            ref_y=None,
            lo_s=lo,
            hi_s=hi,
            win_s=_WIN_S,
            step_s=_STEP_S,
            recording_id=rid,
            stem=stem,
        )
        all_hits.extend(hits)

    args.hits_cache.mkdir(parents=True, exist_ok=True)
    out_path = args.hits_cache / f"{args.set_id}.json"
    out_path.write_text(json.dumps([asdict(h) for h in all_hits], indent=2))
    print(f"wrote {len(all_hits)} hits -> {out_path}")
    if not args.no_push_pi and all_hits:
        push_hits_to_pi(args.set_id, tuple(all_hits))
        print(f"pushed {len(all_hits)} rows to pi-storage set_fingerprint_hits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
