#!/usr/bin/env python3
"""Audit GT fixture recording_ids against the canonical DB: does each fixture
``track_id`` resolve in the canonical DB to a recording whose name matches the
fixture's ``track`` name?

Why this exists: the scorer's "same song" pool is decided CANONICALLY by
``work_id`` via ``labeling/fixtures/id_maps/<set>_work.json`` (built by
``scripts/build_work_map.py`` from the fixture's ``track_id``s). That map is
only correct if the fixture's ``track_id``s still match the canonical DB. We
found cases where a fixture row "Two Friends - Emily (Remix)" carries
``track_id = 2uq9800f``, but ``2uq9800f`` in the canonical DB is "Pacific Coast
Highway (Acappella)" — a different song entirely. The work map then links the
wrong siblings, and cross-recording credits silently go to the wrong song.

This audit surfaces that drift: for each fixture track, it queries the
canonical DB for the recording's ``full_name`` and compares (token overlap)
to the fixture's ``track`` name. Mismatches are printed as a to-fix list.

This is a REPORT only — it does not mutate anything. Fix the fixture's
``track_id`` (or the id_map bridge) and re-run ``scripts/build_work_map.py``.

Usage:
    venvs/audio/bin/python scripts/audit_gt_recording_ids.py --set-id 1fsnxchk
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "labeling" / "fixtures"
DB_PATH = "/mnt/storage/data/db/music_database.db"

# Version/stem/variant qualifiers and credit words that describe the *edit*, not
# the underlying song identity. Stripped before comparison so "Emily (Remix)"
# compares by its core title+artist, not by the word "remix".
_QUALIFIERS = {
    "remix",
    "rework",
    "edit",
    "mix",
    "extended",
    "original",
    "acappella",
    "instrumental",
    "bootleg",
    "mashup",
    "altversion",
    "vip",
    "radio",
    "club",
    "version",
    "feat",
    "featuring",
    "with",
    "the",
    "and",
    "vs",
}


def _tokens(name: str) -> set[str]:
    """Normalized identity tokens: fold the acapella/acappella spelling variant,
    drop version/credit qualifiers, keep >=2-char tokens (so "You", "AJR" survive)."""
    s = (name or "").lower().replace("acapella", "acappella").replace("ft.", " ")
    return {t for t in re.findall(r"[a-z0-9]{2,}", s) if t not in _QUALIFIERS}


def _name_match(fx_name: str, db_name: str) -> bool:
    """Same underlying song? Jaccard OR small-side containment — containment
    catches truncations / added "ft." credits; the union threshold rejects a
    genuinely different title even when one long token happens to overlap."""
    a = _tokens(fx_name)
    b = _tokens(db_name)
    if not a or not b:
        return False
    inter = len(a & b)
    if inter == 0:
        return False
    jaccard = inter / len(a | b)
    containment = inter / min(len(a), len(b))
    # Recall-biased screen: a poison detector must prefer false positives (a
    # human/id-binding stage confirms flags) over false negatives (the Type-II
    # miss this whole operation exists to kill). Thresholds chosen so genuine
    # cross-song swaps (near-zero core-title overlap) fail while same-song rows
    # with dirty fixture text ("Outside Official Acapella", "Ke$ha", truncations)
    # still match. NOT an authoritative certifier — that is the track_audio_id
    # binding (master-plan Phase-1 step 3b); this only surfaces candidates.
    return jaccard >= 0.34 or containment >= 0.4


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", required=True)
    args = p.parse_args(argv)

    fx = [
        f
        for f in sorted(FIXTURES.glob("*_ground_truth.yaml"))
        if yaml.safe_load(f.read_text()).get("set_id") == args.set_id
    ]
    if len(fx) != 1:
        sys.exit(f"no single GT fixture for {args.set_id}")
    tracks = [
        t
        for t in yaml.safe_load(fx[0].read_text()).get("tracks", [])
        if str(t.get("slot_label")) != "mix" and t.get("track_id")
    ]

    rids = sorted({str(t["track_id"]) for t in tracks})
    rid_list = ",".join(f"'{r}'" for r in rids)
    sql = (
        f"SELECT recording_id, full_name FROM recording "
        f"WHERE recording_id IN ({rid_list});"
    )
    r = subprocess.run(
        ["ssh", "pi-storage", f"sqlite3 -csv {DB_PATH} <<'SQL'\n{sql}\nSQL"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    if r.returncode != 0:
        sys.exit(f"pi-storage query failed: {r.stderr.strip() or r.stdout.strip()}")
    # sqlite3 -csv emits NO header row here (no -header flag), so parse every
    # row. csv.reader handles names containing commas/quotes correctly.
    db_name: dict[str, str] = {}
    for row in csv.reader(io.StringIO(r.stdout)):
        if len(row) >= 2:
            db_name[row[0]] = row[1]

    print(
        f"=== GT fixture recording_id audit ({args.set_id}, {len(tracks)} tracks) ==="
    )
    # Three distinct failure classes, deduped by recording_id (a rid can appear
    # on several slots — e.g. a repeated track — but is one identity fact):
    #   poison    — rid resolves to a *different song* (the D1 GT-poisoning class)
    #   not_in_db — rid returns no row at all (genuinely absent / bad id)
    #   blank     — rid exists but has an empty full_name (can't verify; tlp*
    #               sided-row placeholders live here — a data gap, not poison)
    poison: dict[str, tuple] = {}
    not_in_db: dict[str, tuple] = {}
    blank: dict[str, tuple] = {}
    for t in tracks:
        rid = str(t["track_id"])
        slot = t.get("slot_label")
        fx_name = str(t.get("track") or "")
        if rid not in db_name:
            not_in_db.setdefault(rid, (slot, rid, fx_name))
        elif not db_name[rid].strip():
            blank.setdefault(rid, (slot, rid, fx_name))
        elif not _name_match(fx_name, db_name[rid]):
            poison.setdefault(rid, (slot, rid, fx_name, db_name[rid]))

    if poison:
        print(
            f"  POISON — fixture track_id resolves to a DIFFERENT song "
            f"({len(poison)} unique rid):"
        )
        for slot, rid, fxn, dbn in poison.values():
            print(f"    slot {slot}  rid={rid}")
            print(f"      fixture: {fxn[:60]}")
            print(f"      DB:      {dbn[:60]}")
    if not_in_db:
        print(f"  NOT IN canonical DB ({len(not_in_db)} unique rid):")
        for slot, rid, name in not_in_db.values():
            print(f"    slot {slot}  rid={rid}  fixture={name[:50]}")
    if blank:
        print(
            f"  UNVERIFIABLE — rid present but blank full_name "
            f"({len(blank)} unique rid; data gap, not counted as poison):"
        )
        for slot, rid, name in blank.values():
            print(f"    slot {slot}  rid={rid}  fixture={name[:50]}")
    if not (poison or not_in_db or blank):
        print("  (all fixture track_ids resolve to matching names in the canonical DB)")
    # Fail on poison (the class this audit exists to catch) and on genuinely
    # absent ids. Blank-name rows are reported but do not fail the gate.
    return 0 if not (poison or not_in_db) else 1


if __name__ == "__main__":
    sys.exit(main())
