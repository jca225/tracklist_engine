"""GT-driven inventory reconcile — diff labeling truth vs canonical pi-storage."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core.result import Err, Ok
from core.slot_inventory import InventoryAction, suggest_actions
from labeling.ground_truth.schema import load as load_gt
from labeling.inventory_check import evaluate_set_inventory

PI_HOST = "pi-storage"
PI_DB = "/mnt/storage/data/db/music_database.db"


def _ssh_sqlite(sql: str) -> list[dict]:
    proc = subprocess.run(
        ["ssh", PI_HOST, f"sqlite3 -json {PI_DB} {json.dumps(sql)}"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ssh sqlite failed")
    if not proc.stdout.strip():
        return []
    return json.loads(proc.stdout)


def reconcile_gt(yaml_path: Path) -> list[InventoryAction]:
    match load_gt(yaml_path):
        case Err(e):
            raise RuntimeError(f"GT load: {e.detail}")
        case Ok(gt):
            pass

    actions: list[InventoryAction] = []
    inventory = {
        s.claim.slot_label: s for s in evaluate_set_inventory(gt.set_id, _ssh_sqlite)
    }

    for track in gt.tracks:
        label = track.slot_label or track.label
        if label == "mix":
            continue
        inv = inventory.get(label)
        if inv is None:
            continue
        actions.extend(suggest_actions(inv))
        if track.media_links.any():
            from core.slot_inventory import ActionKind

            actions.append(
                InventoryAction(
                    kind=ActionKind.REVIEW_MANUAL,
                    set_id=gt.set_id,
                    slot_label=label,
                    recording_id=track.track_id or "",
                    detail="upsert media_links from GT",
                )
            )
    return actions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--yaml", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("reconcile_actions.csv"))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    try:
        actions = reconcile_gt(args.yaml)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["kind", "set_id", "slot_label", "recording_id", "detail"]
        )
        w.writeheader()
        for a in actions:
            w.writerow(asdict(a))

    print(f"Wrote {len(actions)} actions -> {args.out}")
    if args.apply:
        print("Review CSV then run ingest_stem_url / replace_track_audio per row")
    return 0


if __name__ == "__main__":
    sys.exit(main())
