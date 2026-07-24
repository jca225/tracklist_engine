"""Quarantine non-manifest slot-prefix orphans under ~/aligning.

Moves untagged ``tracks|stems/NNN__*`` entries that are not referenced by the
manifest into ``_orphan_slot_collisions/``. Annotator-tagged ``[NNNbpm KK]`` /
``[no-features]`` names are never touched.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from labeling.acquire.reconcile_aligning_manifest import (  # noqa: E402
    _ORPHAN_QUARANTINE_DIR,
    _find_set_dir,
    scan_slot_collisions,
)


@dataclass(frozen=True)
class QuarantineReport:
    moved: tuple[Path, ...]


def quarantine_orphans(set_dir: Path, *, dry_run: bool = True) -> QuarantineReport:
    set_dir = Path(set_dir)
    scan = scan_slot_collisions(set_dir)
    dest_root = set_dir / _ORPHAN_QUARANTINE_DIR
    moved: list[Path] = []
    for entry in scan.quarantine_candidates:
        rel_parent = "tracks" if entry.kind == "track" else "stems"
        dest = dest_root / rel_parent / entry.path.name
        if dest.exists():
            raise SystemExit(f"quarantine destination already exists: {dest}")
        if dry_run:
            print(
                f"would move {entry.path.relative_to(set_dir)} "
                f"-> {dest.relative_to(set_dir)}"
            )
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(entry.path), str(dest))
            print(
                f"moved {entry.path.relative_to(set_dir)} "
                f"-> {dest.relative_to(set_dir)}"
            )
        moved.append(entry.path)
    return QuarantineReport(moved=tuple(moved))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Quarantine untagged slot orphans not referenced by manifest"
    )
    parser.add_argument("set_id", help="Set id (e.g. 1fsnxchk)")
    parser.add_argument(
        "--dest",
        default="~/aligning",
        help="Aligning root (default: ~/aligning)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Move files (default: dry-run only)",
    )
    args = parser.parse_args(argv)

    set_dir = _find_set_dir(Path(args.dest).expanduser(), args.set_id)
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"quarantine {set_dir.name} [{mode}]")
    report = quarantine_orphans(set_dir, dry_run=not args.apply)
    print(f"moved: {len(report.moved)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
