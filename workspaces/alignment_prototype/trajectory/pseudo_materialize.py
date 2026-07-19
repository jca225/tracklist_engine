"""Materialize audited pseudo-GT YAML from an agentic timeline (E1 §7.2).

Consumes a validated agentic PredictedTimeline + the pulled-set Manifest,
keeps only AUTO_COMMIT spans that survive G0–G4, resolves ``track_id`` for
audio lookup, and writes an atomic YAML artifact with per-gate drop counts.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.contracts import load_manifest, load_timeline
from core.contracts.ids import normalize_slot_label
from core.contracts.manifest import ManifestRow
from workspaces.alignment_prototype.drivers.base import load_timeline_dict
from workspaces.alignment_prototype.trajectory.pseudo_labels import (
    AUTO_COMMIT_MODE,
    pseudo_gt_row,
)

_GATED = frozenset({"g2_independence", "g3_fiber"})


@dataclass(frozen=True)
class MaterializeReport:
    total: int
    accepted: int
    dropped: dict[str, int]
    output: Path


def resolve_track_id(
    span: dict,
    tracks: tuple[ManifestRow, ...],
) -> str | None:
    """Resolve a span's audio ``track_id`` from the typed manifest rows.

    Order: existing span ``track_id``, then ``recording_id``, then normalized
    slot label. Ambiguous matches are rejected (return None) rather than
    silently picking the first candidate.
    """
    by_tid: dict[str, list[ManifestRow]] = {}
    by_rid: dict[str, list[ManifestRow]] = {}
    by_slot: dict[str, list[ManifestRow]] = {}
    for row in tracks:
        by_tid.setdefault(row.track_id, []).append(row)
        if row.recording_id:
            by_rid.setdefault(str(row.recording_id), []).append(row)
        slot = normalize_slot_label(row.slot_label or row.label)
        if slot is not None:
            by_slot.setdefault(str(slot), []).append(row)

    def _unique(hits: list[ManifestRow]) -> str | None:
        tids = {h.track_id for h in hits}
        if len(tids) == 1:
            return next(iter(tids))
        return None

    existing = span.get("track_id")
    if existing not in (None, ""):
        hits = by_tid.get(str(existing), [])
        if hits:
            return _unique(hits)
        # span already carries a track_id; trust it if present even when the
        # manifest index is empty for that key (synthetic fixtures).
        return str(existing)

    rid = span.get("recording_id")
    if rid not in (None, ""):
        hits = by_rid.get(str(rid), [])
        if hits:
            tid = _unique(hits)
            if tid is not None:
                return tid

    slot = normalize_slot_label(str(span.get("slot_label") or ""))
    if slot is not None:
        hits = by_slot.get(str(slot), [])
        if hits:
            return _unique(hits)
    return None


def _drop_reason(span: dict) -> str | None:
    """Return a gate key when the span must be dropped, else None (candidate)."""
    mode = str(span.get("driver_mode") or "")
    rejection = span.get("pseudo_gate_rejection")
    if mode != AUTO_COMMIT_MODE:
        if rejection in _GATED:
            return str(rejection)
        return "g0_mode"
    return None


def materialize_pseudo_gt(
    timeline_path: Path,
    manifest_path: Path,
    output_path: Path,
) -> MaterializeReport:
    timeline_path = Path(timeline_path)
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)

    load_timeline(timeline_path)  # raises on schema faults
    raw = load_timeline_dict(timeline_path)
    manifest = load_manifest(manifest_path)
    source_sha = hashlib.sha256(timeline_path.read_bytes()).hexdigest()

    drops: Counter[str] = Counter()
    tracks: list[dict[str, Any]] = []
    spans = list(raw.get("spans") or ())
    for span in spans:
        reason = _drop_reason(span)
        if reason is not None:
            drops[reason] += 1
            continue
        row = pseudo_gt_row(span)
        if row is None:
            drops["g4_structure"] += 1
            continue
        tid = resolve_track_id(span, manifest.tracks)
        if tid is None:
            drops["audio_identity"] += 1
            continue
        row["track_id"] = tid
        tracks.append(row)

    provenance = {
        "source_timeline": str(timeline_path),
        "sha256": source_sha,
        "pseudo_safety": dict(raw.get("pseudo_safety") or {}),
    }
    doc = {
        "set_id": str(raw["set_id"]),
        "provenance": provenance,
        "gate_counts": {
            "total": len(spans),
            "accepted": len(tracks),
            "dropped": dict(sorted(drops.items())),
        },
        "tracks": tracks,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    text = yaml.safe_dump(doc, sort_keys=True, default_flow_style=False)
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, output_path)

    return MaterializeReport(
        total=len(spans),
        accepted=len(tracks),
        dropped=dict(sorted(drops.items())),
        output=output_path,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timeline", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)

    report = materialize_pseudo_gt(args.timeline, args.manifest, args.output)
    print(f"accepted {report.accepted}/{report.total} -> {report.output}")
    if report.dropped:
        print(
            "dropped: "
            + " ".join(f"{k}={v}" for k, v in sorted(report.dropped.items()))
        )
    return 2 if report.accepted == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
