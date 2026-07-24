#!/usr/bin/env python3
"""Emit the scrape-id -> canonical recording_id BRIDGE map from canonical pi.

This is the map read at ``labeling/fixtures/id_maps/<set_id>.json`` by the
scorer (``eda/alignment/failure_analysis/build_span_table.py``,
``workspaces/alignment_prototype/score_timeline_vs_gt.py``) and by inference
(``workspaces/alignment_prototype/infer.py`` ``_manifest_by_tid``). Its job is to
put predictions and GT into ONE id-namespace: a predicted span / manifest row
may be keyed by any scrape-namespace id (``set_track_slots.tlp_id`` — a bare
number, or ``track_id`` — a hash or ``tlp<n>`` for un-resolved sided rows), while
predictions carry the canonical ``recording_id``. Without a correct bridge,
ref-stem/lyrics resolution silently no-ops (BB11 2026-07-02: 0/67 refs resolved)
and identity is scored across mismatched namespaces.

Distinct from ``<set_id>_work.json`` (recording -> same-work siblings, fiber
pooling) — that is a different map with a different generator.

Why this script exists: the bridge map was previously hand-committed (BB11,
5250ab4) with NO generator, so it silently rotted — BB12's was never built and
BB11's went stale (2026-07-02) while the spine changed. See state-of-record
decision #18. Regenerating from the canonical spine is the fix; run it whenever
``set_track_slots`` for a GT set changes.

The DB is queried over SSH to pi-storage (the local ``data/db/music_database.db``
is a stale dev copy — and on this machine no longer exists). See
``.claude/skills/pi-storage-query/SKILL.md``.

Usage:
    venvs/audio/bin/python scripts/build_bridge_id_map.py --set-id 1fsnxchk
    venvs/audio/bin/python scripts/build_bridge_id_map.py --set-id 2nvzlh2k --check
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "labeling" / "fixtures" / "id_maps"
PI_HOST = "pi-storage"
PI_DB = "/mnt/storage/data/db/music_database.db"


def _spine_rows(set_id: str) -> list[tuple[str, str, str]]:
    """(tlp_id, track_id, recording_id) per slot from canonical set_track_slots."""
    sql = (
        "SELECT COALESCE(tlp_id,''), COALESCE(track_id,''), "
        f"COALESCE(recording_id,'') FROM set_track_slots WHERE set_id='{set_id}' "
        "ORDER BY row_index;"
    )
    r = subprocess.run(
        ["ssh", PI_HOST, f'sqlite3 -separator "|" {PI_DB} "{sql}"'],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    rows: list[tuple[str, str, str]] = []
    for line in r.stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) == 3:
            rows.append((parts[0], parts[1], parts[2]))
    return rows


def build_map(set_id: str) -> dict[str, str]:
    """Every scrape-namespace id form -> its slot's canonical recording_id.

    Keys = {tlp_id, track_id, recording_id} (whichever are non-empty); value =
    recording_id (as-is — for un-resolved sided rows that IS a ``tlp<n>`` value,
    which is exactly what predictions carry). Self-maps are kept (harmless: the
    scorer skips ``k == v``; ``_manifest_by_tid`` setdefault is a no-op) so the
    map is a complete id -> canonical lookup regardless of which id the manifest
    happens to key on.
    """
    rows = _spine_rows(set_id)
    if not rows:
        sys.exit(f"no set_track_slots rows for set_id={set_id} on canonical pi")
    m: dict[str, str] = {}
    for tlp, tid, rid in rows:
        if not rid:
            continue
        for key in (tlp, tid, rid):
            if key:
                m[key] = rid
    return m


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set-id", required=True)
    ap.add_argument(
        "--check",
        action="store_true",
        help="compare against the existing map on disk (if any) and report "
        "shared-key agreement / drift; do not write.",
    )
    args = ap.parse_args(argv)

    new = build_map(args.set_id)
    out_path = OUT_DIR / f"{args.set_id}.json"

    # Sanity: every value must be a recording_id that appears in the spine.
    valid_targets = set(new.values())
    bad = [k for k, v in new.items() if v not in valid_targets]
    assert not bad, f"internal: values not self-consistent: {bad[:5]}"

    if out_path.exists():
        old = json.loads(out_path.read_text())
        shared = set(old) & set(new)
        agree = sum(1 for k in shared if old[k] == new[k])
        disagree = [(k, old[k], new[k]) for k in shared if old[k] != new[k]]
        print(
            f"[{args.set_id}] existing={len(old)} new={len(new)} "
            f"shared={len(shared)} agree={agree} disagree={len(disagree)} "
            f"old-only={len(set(old) - set(new))} new-only={len(set(new) - set(old))}"
        )
        for d in disagree[:8]:
            print("  DISAGREE:", d)
    else:
        print(f"[{args.set_id}] new map: {len(new)} entries (no prior on disk)")

    if args.check:
        print("(--check: not writing)")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(new, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
