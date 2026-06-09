"""Audit ALS clip paths for GT export readiness.

Re-runs the export clip pipeline and flags rows that still need manual QA
(missing local files, or ``tracks/`` refs with no exact manifest path match).
Identity labels come from the ALS path; manifest is pull inventory only.

Usage::

    venvs/audio/bin/python -m labeling.als_path_audit \\
        --out labeling/fixtures/bb12_path_audit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from labeling.als_io import build_manifest_index, display_from_path, match_manifest_for_path, slot_from_path
from labeling.export_als_to_gt import DEFAULT_ALS, DEFAULT_SET_DIR, ClipRow, collect_kept_clip_rows


@dataclass(frozen=True)
class AuditRow:
    status: str  # auto_cleared | needs_ears
    gt_label: str
    path_label: str
    als_path: str
    path_slot: str
    gt_slot: str
    track_id: str | None
    claimed_stem: str
    ref_source: str
    set_start_s: float
    set_end_s: float
    file_exists: bool
    reasons: tuple[str, ...]


def audit_row(row: ClipRow, manifest) -> AuditRow:
    path = row.clip.path
    path_label = display_from_path(path)
    gt_label = row.display
    path_slot = slot_from_path(path) or ""
    reasons: list[str] = []

    exists = Path(path).is_file()
    if not exists:
        reasons.append("missing_file")

    matched = match_manifest_for_path(path, manifest)
    if matched is None and "/tracks/" in path.replace("\\", "/"):
        reasons.append("unresolved_manifest")

    status = "needs_ears" if reasons else "auto_cleared"
    return AuditRow(
        status=status,
        gt_label=gt_label,
        path_label=path_label,
        als_path=path,
        path_slot=path_slot,
        gt_slot=row.slot_label,
        track_id=row.recording_id,
        claimed_stem=row.claimed_stem,
        ref_source=row.ref_source,
        set_start_s=row.set_start_s,
        set_end_s=row.set_end_s,
        file_exists=exists,
        reasons=tuple(reasons),
    )


def run_audit(
    als_path: Path,
    set_dir: Path,
    *,
    include_all: bool = False,
) -> tuple[str, list[AuditRow]]:
    manifest = build_manifest_index(set_dir / "manifest.json")
    set_id, rows, _review = collect_kept_clip_rows(
        als_path, set_dir, include_all=include_all,
    )
    return set_id, [audit_row(r, manifest) for r in rows]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--als", type=Path, default=DEFAULT_ALS)
    ap.add_argument("--set-dir", type=Path, default=DEFAULT_SET_DIR)
    ap.add_argument("--out", type=Path, default=_REPO / "labeling/fixtures/bb12_path_audit.json")
    ap.add_argument("--include-all-clips", action="store_true")
    args = ap.parse_args(argv)

    if not args.als.is_file():
        print(f"not found: {args.als}", file=sys.stderr)
        return 2
    if not args.set_dir.is_dir():
        print(f"not found: {args.set_dir}", file=sys.stderr)
        return 2

    try:
        set_id, rows = run_audit(
            args.als, args.set_dir, include_all=args.include_all_clips,
        )
    except (OSError, ValueError) as e:
        print(f"audit failed: {e}", file=sys.stderr)
        return 1

    cleared = [r for r in rows if r.status == "auto_cleared"]
    ears = [r for r in rows if r.status == "needs_ears"]

    payload = {
        "set_id": set_id,
        "n_rows": len(rows),
        "auto_cleared": len(cleared),
        "needs_ears": len(ears),
        "note": (
            "ALS is canonical for identity (path → label/stem/slot). "
            "manifest.json is pull inventory; used only for set_id, mix duration, "
            "and exact-path track_id attachment."
        ),
        "rows": [asdict(r) for r in rows],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    print(f"audit: {len(cleared)} auto_cleared, {len(ears)} needs_ears -> {args.out}")
    for row in ears[:12]:
        print(
            f"  [{','.join(row.reasons)}] {row.set_start_s:.0f}-{row.set_end_s:.0f}s "
            f"GT={row.gt_label[:35]!r} path={row.path_label[:35]!r}"
        )
    if len(ears) > 12:
        print(f"  ... +{len(ears) - 12} more in JSON")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
