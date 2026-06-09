"""Attach ``track_id`` to exported ground truth without touching ALS labels.

GT labels/timing come from the ``.als`` clip path. This pass only fills
``track_id`` for rows that lack one:

1. Exact manifest path match (clip path ↔ pull inventory)
2. Unique ``set_track_slots`` title match on pi-storage (label overlap + stem)

Never uses scrape slot prefix alone.

Usage::

    venvs/audio/bin/python -m labeling.enrich_gt_track_ids \\
        --yaml labeling/fixtures/bb12_ground_truth.yaml \\
        --als "$HOME/Desktop/big bootie 12 labeling Project/big bootie 12 labeling_fast.als" \\
        --set-dir "$HOME/aligning/1fsnxchk__Two Friends - Big Bootie Mix Volume 12"
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from labeling.als_io import (
    build_manifest_index,
    display_from_path,
    labels_overlap,
    match_manifest_for_path,
)
from labeling.export_als_to_gt import DEFAULT_ALS, DEFAULT_SET_DIR, ClipRow, collect_kept_clip_rows
from labeling.ground_truth.schema import GroundTruthSet, GroundTruthTrack, load, save
from core.result import Err, Ok

PI_HOST = "pi-storage"
PI_DB = "/mnt/storage/data/db/music_database.db"
DEFAULT_YAML = _REPO / "labeling/fixtures/bb12_ground_truth.yaml"
SPAN_TOL_S = 0.05


@dataclass(frozen=True)
class SlotRow:
    track_id: str | None
    claimed_stem: str
    display: str


@dataclass(frozen=True)
class EnrichResult:
    track: GroundTruthTrack
    track_id: str | None
    source: str  # kept | manifest_path | db_label | unresolved


def _ssh_sql(query: str, *, host: str = PI_HOST, db: str = PI_DB) -> list[dict[str, str]]:
    cmd = ["ssh", "-o", "ConnectTimeout=15", host, f"sqlite3 -separator '|' -header {db} {query!r}"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        return []
    header = lines[0].split("|")
    return [dict(zip(header, ln.split("|"), strict=False)) for ln in lines[1:]]


def _slot_display(row: dict[str, str]) -> str:
    full = (row.get("full_name") or "").strip()
    if full:
        return full
    title = (row.get("title") or "").strip()
    artists = (row.get("artists_json") or "").strip()
    if artists and artists.startswith("["):
        try:
            names = json.loads(artists)
            if names:
                return f"{names[0]} - {title}" if title else str(names[0])
        except json.JSONDecodeError:
            pass
    return title


def fetch_set_slots(set_id: str, *, host: str = PI_HOST, db: str = PI_DB) -> tuple[SlotRow, ...]:
    rows = _ssh_sql(
        "SELECT track_id, claimed_stem, full_name, title, artists_json "
        f"FROM set_track_slots WHERE set_id='{set_id}' ORDER BY row_index",
        host=host,
        db=db,
    )
    out: list[SlotRow] = []
    for row in rows:
        tid = (row.get("track_id") or "").strip() or None
        stem = (row.get("claimed_stem") or "regular").strip().lower()
        out.append(SlotRow(track_id=tid, claimed_stem=stem, display=_slot_display(row)))
    return tuple(out)


def _norm_slot(slot: str) -> str:
    s = str(slot or "").strip()
    if not s:
        return ""
    if s.isdigit() and len(s) <= 3:
        return str(int(s)).zfill(3)
    if "w" in s:
        base, _, suffix = s.partition("w")
        if base.isdigit():
            return f"{int(base):03d}w{suffix}"
    return s


def _track_key(t: GroundTruthTrack) -> tuple[str, str, float]:
    stem = (t.claimed_stem or "regular").lower()
    slot = _norm_slot(t.slot_label or t.label)
    return slot, stem, round(t.set_start_s, 1)


def _clip_key(row: ClipRow) -> tuple[str, str, float]:
    stem = (row.claimed_stem or "regular").lower()
    slot = _norm_slot(row.slot_label or "")
    return slot, stem, round(row.set_start_s, 1)


def associate_clips(
    gt: GroundTruthSet,
    clip_rows: tuple[ClipRow, ...],
) -> dict[tuple[str, str, float], ClipRow]:
    by_key: dict[tuple[str, str, float], ClipRow] = {}
    for row in clip_rows:
        by_key.setdefault(_clip_key(row), row)

    out: dict[tuple[str, str, float], ClipRow] = {}
    for t in gt.tracks:
        key = _track_key(t)
        if key in by_key:
            out[key] = by_key[key]
            continue
        stem = (t.claimed_stem or "regular").lower()
        candidates = [
            r for r in clip_rows
            if (r.claimed_stem or "regular").lower() == stem
            and abs(r.set_start_s - t.set_start_s) <= SPAN_TOL_S
        ]
        if len(candidates) == 1:
            out[key] = candidates[0]
        elif candidates:
            out[key] = min(
                candidates,
                key=lambda r: abs(r.set_end_s - t.set_end_s),
            )
    return out


def _overlap_score(left: str, right: str) -> int:
    def _tokens(label: str) -> set[str]:
        import re
        cleaned = re.sub(r"[^\w\s]", " ", label.lower())
        return {w for w in cleaned.split() if len(w) > 2}

    return len(_tokens(left) & _tokens(right))


def lookup_db_label(
    label: str,
    claimed_stem: str,
    slots: tuple[SlotRow, ...],
    *,
    require_stem: bool = True,
    min_tokens: int = 2,
) -> str | None:
    stem = (claimed_stem or "regular").lower()
    hits: list[tuple[int, SlotRow]] = []
    for row in slots:
        if require_stem and row.claimed_stem != stem:
            continue
        score = _overlap_score(label, row.display)
        if score < min_tokens:
            continue
        if not labels_overlap(label, row.display, min_tokens=min_tokens):
            continue
        hits.append((score, row))
    if not hits:
        return None
    hits.sort(key=lambda x: (-x[0], x[1].display))
    best_score = hits[0][0]
    top = [row for score, row in hits if score == best_score]
    if len(top) != 1:
        return None
    return top[0].track_id


def enrich_track(
    track: GroundTruthTrack,
    *,
    clip: ClipRow | None,
    manifest,
    slots: tuple[SlotRow, ...],
) -> EnrichResult:
    if track.track_id:
        return EnrichResult(track=track, track_id=track.track_id, source="kept")

    track_id: str | None = None
    source = "unresolved"

    if clip is not None:
        matched = match_manifest_for_path(clip.clip.path, manifest)
        if matched is not None and matched.track_id:
            track_id = matched.track_id
            source = "manifest_path"

    if track_id is None:
        label = clip.display if clip is not None else track.label
        path_label = display_from_path(clip.clip.path) if clip is not None else track.label
        for candidate in (path_label, track.label, label):
            track_id = lookup_db_label(candidate, track.claimed_stem, slots, require_stem=True)
            if track_id:
                source = "db_label"
                break
        if track_id is None and track.ref_source in ("online_candidate", "demucs", "phase_cancel"):
            for candidate in (path_label, track.label, label):
                track_id = lookup_db_label(
                    candidate, track.claimed_stem, slots,
                    require_stem=False, min_tokens=3,
                )
                if track_id:
                    source = "db_label_relaxed"
                    break

    if track_id is None:
        return EnrichResult(track=track, track_id=None, source="unresolved")

    enriched = replace(track, track_id=track_id)
    return EnrichResult(track=enriched, track_id=track_id, source=source)


def enrich_gt(
    gt: GroundTruthSet,
    *,
    clip_rows: tuple[ClipRow, ...],
    manifest,
    slots: tuple[SlotRow, ...],
) -> tuple[GroundTruthSet, list[EnrichResult]]:
    assoc = associate_clips(gt, clip_rows)
    results: list[EnrichResult] = []
    tracks: list[GroundTruthTrack] = []
    for t in gt.tracks:
        clip = assoc.get(_track_key(t))
        res = enrich_track(t, clip=clip, manifest=manifest, slots=slots)
        results.append(res)
        tracks.append(res.track)
    return replace(gt, tracks=tuple(tracks)), results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yaml", type=Path, default=DEFAULT_YAML)
    ap.add_argument("--als", type=Path, default=DEFAULT_ALS)
    ap.add_argument("--set-dir", type=Path, default=DEFAULT_SET_DIR)
    ap.add_argument("--out", type=Path, default=None, help="defaults to --yaml")
    ap.add_argument("--pi-host", default=PI_HOST)
    ap.add_argument("--pi-db", default=PI_DB)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    match load(args.yaml):
        case Err(e):
            print(e.detail, file=sys.stderr)
            return 1
        case Ok(gt):
            pass

    if not args.als.is_file():
        print(f"not found: {args.als}", file=sys.stderr)
        return 2
    if not args.set_dir.is_dir():
        print(f"not found: {args.set_dir}", file=sys.stderr)
        return 2

    try:
        _set_id, clip_rows, _ = collect_kept_clip_rows(args.als, args.set_dir)
        manifest = build_manifest_index(args.set_dir / "manifest.json")
        slots = fetch_set_slots(gt.set_id, host=args.pi_host, db=args.pi_db)
    except (OSError, ValueError, subprocess.CalledProcessError) as e:
        print(f"enrich failed: {e}", file=sys.stderr)
        return 1

    enriched, results = enrich_gt(gt, clip_rows=tuple(clip_rows), manifest=manifest, slots=slots)
    counts: dict[str, int] = {}
    for r in results:
        counts[r.source] = counts.get(r.source, 0) + 1

    print(
        f"enrich: kept={counts.get('kept', 0)} "
        f"manifest_path={counts.get('manifest_path', 0)} "
        f"db_label={counts.get('db_label', 0)} "
        f"db_label_relaxed={counts.get('db_label_relaxed', 0)} "
        f"unresolved={counts.get('unresolved', 0)}"
    )
    for r in results:
        if r.source == "unresolved":
            print(f"  unresolved: {r.track.set_start_s:.0f}s {r.track.label[:50]}")

    if args.dry_run:
        return 0

    out = args.out or args.yaml
    title = args.set_dir.name
    match save(enriched, out, title=title):
        case Err(e):
            print(e.detail, file=sys.stderr)
            return 1
        case Ok(path):
            print(f"wrote {len(enriched.tracks)} tracks -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
