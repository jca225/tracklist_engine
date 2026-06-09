"""Generate a prioritized BB12 GT fidelity spot-check list."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from labeling.ground_truth.schema import load
from core.result import Err, Ok

PI_HOST = "pi-storage"
PI_DB = "/mnt/storage/data/db/music_database.db"
DEFAULT_YAML = _REPO / "labeling/fixtures/bb12_ground_truth.yaml"
OUT_DEFAULT = _REPO / "labeling/fixtures/bb12_gt_spotcheck.json"

# P1.5 winner-ingest + retry tail (track_audio_id block from June ingest).
WINNER_TRACK_IDS = frozenset({
    "12m8zb3x", "1wvvqwsf", "14jvfw6x", "1n9qvhhp", "zxuqpkf", "13n64sf5",
    "1v0zv1s5", "nrtk415", "j47669p", "sh4csf5", "2j9zlwuf", "1t4gvsx5", "1q89rul5",
})

RISK_VERSIONS = frozenset({"remix", "rework", "edit", "bootleg", "mashup", "altversion"})
RISK_STEMS = frozenset({"acappella", "instrumental"})


def _ssh_sql(query: str) -> list[list[str]]:
    cmd = ["ssh", PI_HOST, f"sqlite3 -separator '|' -header {PI_DB} {query!r}"]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    if not lines:
        return []
    header = lines[0].split("|")
    return [dict(zip(header, ln.split("|"), strict=False)) for ln in lines[1:]]


def _risk_score(row: dict, *, gt_track_id: str | None) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    version = (row.get("claimed_version") or "").lower()
    stem = (row.get("claimed_stem") or "regular").lower()
    tid = row.get("track_id") or gt_track_id or ""

    if not tid:
        score += 3
        reasons.append("null_recording_id")
    if version in RISK_VERSIONS:
        score += 2
        reasons.append(f"version={version}")
    if stem in RISK_STEMS:
        score += 2
        reasons.append(f"stem={stem}")
    if tid in WINNER_TRACK_IDS:
        score += 3
        reasons.append("p15_winner_ingest")
    if row.get("is_manual") == "1":
        score += 1
        reasons.append("manual_ingest")
    return score, reasons


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--yaml", type=Path, default=DEFAULT_YAML)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--top", type=int, default=30, help="Max rows in checklist")
    args = ap.parse_args(argv)

    match load(args.yaml):
        case Err(e):
            print(e.detail, file=sys.stderr)
            return 1
        case Ok(gt):
            pass

    gt_by_slot = {t.slot_label or t.label: t for t in gt.tracks}
    pi_rows = _ssh_sql(
        "SELECT slot_label, track_id, claimed_version, claimed_stem, title "
        "FROM set_track_slots WHERE set_id='1fsnxchk' ORDER BY row_index"
    )
    manual_rows = _ssh_sql(
        "SELECT DISTINCT ta.track_id, 1 AS is_manual FROM track_audio ta "
        "WHERE ta.platform='manual' AND ta.track_id IN "
        "(SELECT DISTINCT track_id FROM set_track_slots WHERE set_id='1fsnxchk')"
    )
    manual_ids = {r["track_id"] for r in manual_rows}

    items: list[dict] = []
    for row in pi_rows:
        slot = row["slot_label"]
        gt_row = gt_by_slot.get(slot)
        gt_tid = gt_row.track_id if gt_row else None
        merged = {**row, "is_manual": "1" if (row.get("track_id") in manual_ids) else "0"}
        score, reasons = _risk_score(merged, gt_track_id=gt_tid)
        if score == 0:
            continue
        items.append({
            "slot": slot,
            "track_id": row.get("track_id") or gt_tid,
            "title": row.get("title") or (gt_row.label if gt_row else ""),
            "claimed_version": row.get("claimed_version"),
            "claimed_stem": row.get("claimed_stem"),
            "risk_score": score,
            "reasons": reasons,
            "gt_set_span": (
                [gt_row.set_start_s, gt_row.set_end_s] if gt_row else None
            ),
        })

    items.sort(key=lambda x: (-x["risk_score"], x["slot"]))
    items = items[: args.top]
    payload = {
        "set_id": gt.set_id,
        "n_gt_rows": len(gt.tracks),
        "checklist": items,
        "instructions": (
            "In Ableton: for each row, audition ref audio vs mix slice at gt_set_span. "
            "Re-export YAML only where audio clearly diverges from annotation."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {len(items)} spot-check rows -> {args.out}")
    for it in items[:10]:
        print(f"  [{it['risk_score']}] {it['slot']:6} {it['title'][:50]}  ({', '.join(it['reasons'])})")
    if len(items) > 10:
        print(f"  ... +{len(items) - 10} more in JSON")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
