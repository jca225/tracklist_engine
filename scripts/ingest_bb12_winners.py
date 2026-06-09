#!/usr/bin/env python3
"""Batch-ingest BB12 Ableton candidate winners into pi-storage (P1.5).

Reads the live `.als`, finds every clip whose OriginalFileRef points at
``stems/.../candidates/...``, resolves ``recording_id`` from manifest +
set_track_slots, and runs ``ingest_stem_url.py --file`` for each winner.

Skips:
  - paths already ingested (same recording_id + stem + player_id prefix)
  - Rvmor ``tlp*`` phantom recording_ids (no ``recording`` row yet)

Usage:
  venvs/audio/bin/python scripts/ingest_bb12_winners.py --dry-run
  venvs/audio/bin/python scripts/ingest_bb12_winners.py
  venvs/audio/bin/python scripts/ingest_bb12_winners.py --limit 5
"""
from __future__ import annotations

import argparse
import gzip
import html
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from lxml import etree

REPO = Path(__file__).resolve().parent.parent
PYTHON = REPO / "venvs" / "audio" / "bin" / "python"
INGEST = REPO / "scripts" / "ingest_stem_url.py"
DEFAULT_ALS = Path.home() / "Desktop/big bootie 12 labeling Project/big bootie 12 labeling_fast.als"
DEFAULT_SET = Path.home() / "aligning/1fsnxchk__Two Friends - Big Bootie Mix Volume 12"
SET_ID = "1fsnxchk"
PI_HOST = "pi-storage"
PI_DB = "/mnt/storage/data/db/music_database.db"

# Annotator folder 049 holds When You Were Young; canonical slot is 014w1.
_SLOT_OVERRIDES: dict[str, str] = {"049": "1q8nc02p"}


@dataclass(frozen=True)
class Winner:
    path: Path
    slot: str
    recording_id: str
    role: str


def _ssh_sql(sql: str) -> str:
    r = subprocess.run(
        ["ssh", PI_HOST, f"sqlite3 {PI_DB} {sql!r}"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or "ssh sqlite failed")
    return r.stdout


def _load_winners(als_path: Path, set_dir: Path) -> tuple[list[Winner], list[str]]:
    manifest = json.loads((set_dir / "manifest.json").read_text())
    slot2row: dict[str, dict] = {}
    for row in manifest.get("tracks") or []:
        m = re.search(r"/(\d{3}(?:w\d+)?)__", str(row.get("local_path") or ""))
        if m:
            slot2row[m.group(1)] = row

    slot2tid: dict[str, str] = {}
    for line in _ssh_sql(
        f"SELECT slot_label, recording_id FROM set_track_slots "
        f"WHERE set_id='{SET_ID}' AND slot_label IS NOT NULL"
    ).splitlines():
        if "|" in line:
            slot, tid = line.split("|", 1)
            slot2tid[slot] = tid

    valid_ids = set(_ssh_sql("SELECT recording_id FROM recording").split())

    root = etree.fromstring(gzip.decompress(als_path.read_bytes()))
    raw: dict[str, Winner] = {}
    skipped: list[str] = []

    for clip in root.xpath(".//AudioClip"):
        ps = clip.xpath(".//SourceContext//OriginalFileRef//Path")
        if not ps:
            continue
        p = html.unescape(ps[0].get("Value") or "")
        if "/candidates/" not in p:
            continue
        path = Path(p)
        if not path.is_file():
            skipped.append(f"missing file: {p}")
            continue
        m = re.search(r"/(\d{3}(?:w\d+)?)__", p)
        slot = m.group(1) if m else ""
        role = "acappella" if "/vocals/" in p or "acap" in p.lower() else "instrumental"
        row = slot2row.get(slot, {})
        tid = _SLOT_OVERRIDES.get(slot) or row.get("track_id") or slot2tid.get(slot)
        if slot and "w" in slot and not tid:
            base = slot.split("w", 1)[0]
            tid = slot2row.get(base, {}).get("track_id") or slot2tid.get(slot)
        if not tid:
            skipped.append(f"no recording_id for slot {slot}: {path.name}")
            continue
        if tid not in valid_ids:
            skipped.append(f"phantom recording_id {tid} slot {slot}: {path.name}")
            continue
        raw[str(path.resolve())] = Winner(path=path, slot=slot, recording_id=tid, role=role)

    return sorted(raw.values(), key=lambda w: (w.slot, w.role, w.path.name)), skipped


def _existing_ingested() -> set[tuple[str, str, str]]:
    """(recording_id, stem, match_key) already on pi."""
    out: set[tuple[str, str, str]] = set()
    for line in _ssh_sql(
        "SELECT recording_id, stem, path FROM track_audio "
        "WHERE stem IN ('acappella','instrumental') AND platform='manual'"
    ).splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        rid, stem, path = parts[0], parts[1], parts[2]
        base = Path(path).name
        out.add((rid, stem, base))
        # Also match ingest_stem_* canonical names back to candidate files.
        if "ingest_stem_" in base:
            tail = base.split("ingest_stem_", 1)[-1]
            out.add((rid, stem, tail))
    return out


def _candidate_key(path: Path) -> str:
    return path.name


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--als", type=Path, default=DEFAULT_ALS)
    p.add_argument("--set-dir", type=Path, default=DEFAULT_SET)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip-preflight", action="store_true")
    args = p.parse_args(argv)

    if not args.als.is_file():
        print(f"not found: {args.als}", file=sys.stderr)
        return 2
    if not (args.set_dir / "manifest.json").is_file():
        print(f"not found: {args.set_dir}/manifest.json", file=sys.stderr)
        return 2

    winners, skipped = _load_winners(args.als, args.set_dir)
    existing = _existing_ingested()

    todo: list[Winner] = []
    for w in winners:
        cand = _candidate_key(w.path)
        if (w.recording_id, w.role, cand) in existing:
            continue
        if (w.recording_id, w.role, w.path.stem) in existing:
            continue
        todo.append(w)

    if args.limit:
        todo = todo[: args.limit]

    print(f"winners={len(winners)} skipped={len(skipped)} todo={len(todo)}")
    for msg in skipped:
        print(f"  skip: {msg}")

    ok = fail = 0
    for w in todo:
        reason = (
            f"source:labeling_bb12|quality:human_winner|ref:online_candidate|"
            f"slot:{w.slot}|file:{w.path.name[:60]}"
        )
        cmd = [
            str(PYTHON), str(INGEST),
            "--file", str(w.path),
            "--track-id", w.recording_id,
            "--role", w.role,
            "--set-id", SET_ID,
            "--reason", reason,
        ]
        if args.skip_preflight:
            cmd.append("--skip-preflight")
        print(f"\n[{ok + fail + 1}/{len(todo)}] slot={w.slot} {w.role} {w.recording_id}")
        if args.dry_run:
            print(" ", " ".join(cmd))
            continue
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        if r.stdout:
            print(r.stdout, end="")
        if r.stderr:
            print(r.stderr, end="", file=sys.stderr)
        if r.returncode == 0 or "inserted new track_audio row" in (r.stdout + r.stderr):
            ok += 1
        else:
            fail += 1
            print(f"FAILED rc={r.returncode}", file=sys.stderr)

    print(f"\ndone: ok={ok} fail={fail}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
