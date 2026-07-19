"""Idempotent Mac-side manifest repair from pi slots + on-disk aligning tree.

Reconciles the aligning pull manifest against canonical ``set_track_slots``
without a destructive full re-pull. Wires ``local_path`` / ``stems`` only when
disk files match the slot label *and* pass title-token disambiguation (fail
closed on colliding orphans). Known proxy homes (e.g. Lux → ``mix_instrumental.flac``)
live in ``PROXY_SLOT_AUDIO``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import msgspec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.contracts import MANIFEST_FILENAME, load_manifest  # noqa: E402

from labeling.inventory_check import (  # noqa: E402
    evaluate_set_inventory,
    fetch_slot_rows,
    satisfaction_to_manifest_fields,
)
from labeling.pull_set_for_alignment import fetch_tracks, ssh_sqlite  # noqa: E402

# (set_id, slot_label) -> filename under set_dir (not under tracks/)
PROXY_SLOT_AUDIO: dict[tuple[str, str], str] = {
    ("1fsnxchk", "032"): "mix_instrumental.flac",  # Lux Holm - Omega — no release
}

_AUDIO_EXT = frozenset({".m4a", ".mp3", ".flac", ".wav", ".ogg", ".opus", ".aac"})
_STEM_NAMES = ("vocals", "instrumental")
_TITLE_STOP = frozenset(
    {
        "a",
        "an",
        "and",
        "by",
        "feat",
        "featuring",
        "ft",
        "in",
        "of",
        "the",
        "vs",
        "with",
    }
)


@dataclass(frozen=True)
class ReconcileReport:
    added: list[str]
    stem_rewrites: list[tuple[str, str, str]]
    wired: list[str]
    unresolved: list[str]
    backed_up: Path | None
    warnings: tuple[str, ...] = ()


def _slot_variants(slot_label: str) -> list[str]:
    s = str(slot_label)
    out = [s]
    m = re.match(r"^0*(\d+)(.*)$", s)
    if m:
        for cand in (m.group(1) + m.group(2), m.group(1).zfill(3) + m.group(2)):
            if cand not in out:
                out.append(cand)
    return out


def _slot_display_name(row: dict[str, Any]) -> str:
    return (
        str(row.get("full_name") or row.get("name") or row.get("slot_title") or "")
    ).strip()


def _split_artist_title(name: str) -> tuple[str, str]:
    if " - " in name:
        artist, title = name.split(" - ", 1)
        return artist.strip(), title.strip()
    return "", name.strip()


def _title_part(name: str) -> str:
    if " - " in name:
        return name.split(" - ", 1)[1].strip()
    return name.strip()


def _strong_title_tokens(name: str) -> set[str]:
    title = _title_part(name)
    toks = re.split(r"[\W_]+", title.lower())
    return {t for t in toks if len(t) >= 3 and t not in _TITLE_STOP}


def _name_score_ok(slot_name: str, filename: str) -> bool:
    """Require ≥1 strong title token in the filename stem (fail closed)."""
    want = _strong_title_tokens(slot_name)
    if not want:
        return False
    stem = Path(filename).stem
    m = re.match(r"^\d{3}(?:w\d+)?__", stem)
    if m:
        stem = stem[m.end() :]
    hay = set(re.split(r"[\W_]+", stem.lower()))
    return bool(want & hay)


def _local_path_is_valid(
    set_dir: Path,
    set_id: str,
    slot_label: str,
    slot_name: str,
    local_path: Path,
) -> bool:
    """True when an existing manifest path is allowed for this slot."""
    if not local_path.is_file():
        return False
    proxy_rel = PROXY_SLOT_AUDIO.get((set_id, slot_label))
    if (
        proxy_rel is not None
        and local_path.resolve() == (set_dir / proxy_rel).resolve()
    ):
        return True
    return _name_score_ok(slot_name, local_path.name)


def _track_file_candidates(set_dir: Path, slot_label: str) -> list[Path]:
    tracks_dir = set_dir / "tracks"
    if not tracks_dir.is_dir():
        return []
    by_res: dict[Path, Path] = {}
    for slot in _slot_variants(slot_label):
        for hit in sorted(tracks_dir.glob(f"{slot}__*")):
            if hit.suffix.lower() in _AUDIO_EXT and hit.is_file():
                by_res[hit.resolve()] = hit
    return list(by_res.values())


def _pick_track_file(
    set_dir: Path, set_id: str, slot_label: str, slot_name: str
) -> Path | None:
    proxy = PROXY_SLOT_AUDIO.get((set_id, slot_label))
    if proxy is not None:
        path = set_dir / proxy
        if path.is_file():
            return path
    matched = [
        p
        for p in _track_file_candidates(set_dir, slot_label)
        if _name_score_ok(slot_name, p.name)
    ]
    if len(matched) == 1:
        return matched[0]
    return None


def _stem_dirs(set_dir: Path, slot_label: str, slot_name: str) -> list[Path]:
    stems_root = set_dir / "stems"
    if not stems_root.is_dir():
        return []
    out: list[Path] = []
    seen: set[Path] = set()
    for slot in _slot_variants(slot_label):
        for directory in sorted(stems_root.glob(f"{slot}__*")):
            if not directory.is_dir():
                continue
            if not _name_score_ok(slot_name, directory.name):
                continue
            resolved = directory.resolve()
            if resolved not in seen:
                seen.add(resolved)
                out.append(directory)
    return out


def _wire_stem_paths(stem_dirs: list[Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for stem_name in _STEM_NAMES:
        for directory in stem_dirs:
            for ext in (".flac", ".m4a", ".wav", ".mp3"):
                candidate = directory / f"{stem_name}{ext}"
                if candidate.is_file():
                    out[stem_name] = str(candidate)
                    break
            if stem_name in out:
                break
    return out


def _proxy_manifest_fields(
    set_id: str,
    slot_label: str,
    slot_name: str,
    local_path: Path | None,
    set_dir: Path,
) -> dict[str, Any] | None:
    """Manifest gap/stem/satisfaction when a slot uses ``PROXY_SLOT_AUDIO``."""
    proxy_rel = PROXY_SLOT_AUDIO.get((set_id, slot_label))
    if proxy_rel is None or local_path is None:
        return None
    if local_path.resolve() != (set_dir / proxy_rel).resolve():
        return None
    proxy_stem = Path(proxy_rel).stem
    return {
        "stem": "instrumental",
        "satisfaction": "fallback",
        "gap": f"proxy:{proxy_stem} — original unavailable ({slot_name})",
    }


def _effective_stem(
    claimed_stem: str, stems: dict[str, str], local_path: Path | None
) -> str:
    if claimed_stem == "acappella" and "vocals" in stems:
        return "acappella"
    if claimed_stem == "instrumental" and "instrumental" in stems:
        return "instrumental"
    if local_path is not None and claimed_stem == "regular":
        return "regular"
    if local_path is not None and claimed_stem in ("acappella", "instrumental"):
        if claimed_stem == "acappella" and "vocals" not in stems:
            return claimed_stem
        if claimed_stem == "instrumental" and "instrumental" not in stems:
            return claimed_stem
    return claimed_stem


def _base_row(slot: dict[str, Any], pi_track: Any | None = None) -> dict[str, Any]:
    label = str(slot["slot_label"])
    rid = str(slot["recording_id"])
    name = _slot_display_name(slot)
    artist, title = _split_artist_title(name)
    claimed_stem = slot.get("claimed_stem") or "regular"
    claimed_variant = slot.get("claimed_variant") or "regular"
    row: dict[str, Any] = {
        "track_id": rid,
        "track_audio_id": None,
        "label": label,
        "slot_label": label,
        "artist": artist,
        "title": title or name,
        "version": None,
        "stem": claimed_stem,
        "variant": claimed_variant,
        "axes_key": f"original__{claimed_stem}__{claimed_variant}",
        "local_path": None,
        "pi_path": None,
        "duration_s": None,
        "stems": {},
        "satisfaction": "unresolved",
        "gap": "reconcile: no resolvable local audio",
    }
    if pi_track is not None:
        row["track_audio_id"] = pi_track.track_audio_id
        row["artist"] = pi_track.artist
        row["title"] = pi_track.title
        row["version"] = pi_track.version
        row["stem"] = pi_track.stem
        row["variant"] = pi_track.variant
        row["axes_key"] = (
            f"{pi_track.version or 'original'}__{pi_track.stem}__{pi_track.variant}"
        )
        row["pi_path"] = pi_track.pi_path
        row["duration_s"] = pi_track.duration_s
    return row


def _load_satisfaction_by_label(
    set_id: str,
) -> tuple[dict[str, Any], list[str]]:
    """Pi inventory enrichment; SSH/parse failures become visible warnings."""
    try:
        return {
            s.claim.slot_label: s for s in evaluate_set_inventory(set_id, ssh_sqlite)
        }, []
    except (subprocess.CalledProcessError, json.JSONDecodeError, OSError) as exc:
        return {}, [f"inventory enrichment skipped: {exc}"]


def reconcile_manifest(set_dir: Path, *, dry_run: bool = True) -> ReconcileReport:
    set_dir = Path(set_dir)
    manifest_path = set_dir / MANIFEST_FILENAME
    loaded = load_manifest(manifest_path)
    manifest: dict[str, Any] = msgspec.to_builtins(loaded)
    set_id = str(manifest["set_id"])

    slot_rows = fetch_slot_rows(set_id, ssh_sqlite)
    pi_tracks = fetch_tracks(set_id)
    pi_by_label = {t.label: t for t in pi_tracks}
    satisfaction_by_label, report_warnings = _load_satisfaction_by_label(set_id)

    existing = list(manifest.get("tracks") or [])
    by_label = {
        str(t.get("slot_label") or t.get("label")): dict(t)
        for t in existing
        if t.get("slot_label") or t.get("label")
    }
    by_id = {str(t.get("track_id")): dict(t) for t in existing if t.get("track_id")}

    added: list[str] = []
    stem_rewrites: list[tuple[str, str, str]] = []
    wired: list[str] = []
    unresolved: list[str] = []

    reconciled: list[dict[str, Any]] = []
    seen_labels: set[str] = set()

    for slot in slot_rows:
        label = str(slot["slot_label"])
        rid = str(slot["recording_id"])
        slot_name = _slot_display_name(slot)
        seen_labels.add(label)

        row = by_label.get(label) or by_id.get(rid)
        if row is None:
            row = _base_row(slot, pi_by_label.get(label))
            added.append(label)
        else:
            row = dict(row)
            row.setdefault("slot_label", label)
            row.setdefault("label", label)
            row["track_id"] = rid

        old_stem = str(row.get("stem") or "regular")
        claimed_stem = slot.get("claimed_stem") or "regular"
        claimed_variant = slot.get("claimed_variant") or "regular"
        row["variant"] = claimed_variant

        existing_local = row.get("local_path")
        if existing_local:
            path_obj = Path(str(existing_local))
            if not _local_path_is_valid(set_dir, set_id, label, slot_name, path_obj):
                row["local_path"] = None

        track_path = _pick_track_file(set_dir, set_id, label, slot_name)
        stem_dirs = _stem_dirs(set_dir, label, slot_name)
        stems = _wire_stem_paths(stem_dirs)

        if track_path is not None:
            row["local_path"] = str(track_path)
            wired.append(label)
        elif row.get("local_path") and Path(str(row["local_path"])).is_file():
            wired.append(label)
        else:
            row["local_path"] = None
            unresolved.append(label)

        row["stems"] = stems
        local_path_obj = Path(row["local_path"]) if row.get("local_path") else None
        proxy_fields = _proxy_manifest_fields(
            set_id, label, slot_name, local_path_obj, set_dir
        )
        if proxy_fields is not None:
            new_stem = str(proxy_fields["stem"])
        else:
            new_stem = _effective_stem(claimed_stem, stems, local_path_obj)
        if new_stem != old_stem:
            stem_rewrites.append((label, old_stem, new_stem))
        row["stem"] = new_stem
        row["axes_key"] = (
            f"{row.get('version') or 'original'}__{new_stem}__{claimed_variant}"
        )

        sat = satisfaction_by_label.get(label)
        if sat is not None:
            row.update(satisfaction_to_manifest_fields(sat))
        elif row.get("local_path") is None:
            row["satisfaction"] = "unresolved"
            row["gap"] = row.get("gap") or "reconcile: no resolvable local audio"

        if proxy_fields is not None:
            row.update(proxy_fields)

        reconciled.append(row)

    for track in existing:
        label = str(track.get("slot_label") or track.get("label") or "")
        if label and label not in seen_labels:
            reconciled.append(dict(track))

    backed_up: Path | None = None
    if not dry_run:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_name = f"{MANIFEST_FILENAME}.bak_reconcile_{ts}"
        backed_up = manifest_path.with_name(backup_name)
        backed_up.write_text(manifest_path.read_text())
        manifest["tracks"] = reconciled
        manifest_path.write_text(json.dumps(manifest, indent=2))

    return ReconcileReport(
        added=added,
        stem_rewrites=stem_rewrites,
        wired=wired,
        unresolved=unresolved,
        backed_up=backed_up,
        warnings=tuple(report_warnings),
    )


def _find_set_dir(dest: Path, set_id: str) -> Path:
    hits = sorted(dest.glob(f"{set_id}__*"))
    if not hits:
        raise SystemExit(f"no aligning folder matching {set_id}__* under {dest}")
    if len(hits) > 1:
        print(
            f"warning: multiple folders for {set_id}, using {hits[0]}", file=sys.stderr
        )
    return hits[0]


def _print_report(report: ReconcileReport) -> None:
    print(f"added:       {', '.join(report.added) or '(none)'}")
    print(f"wired:       {', '.join(report.wired) or '(none)'}")
    print(f"unresolved:  {', '.join(report.unresolved) or '(none)'}")
    if report.warnings:
        print("warnings:")
        for msg in report.warnings:
            print(f"  {msg}")
    if report.stem_rewrites:
        print("stem_rewrites:")
        for label, old, new in report.stem_rewrites:
            print(f"  {label}: {old} -> {new}")
    if report.backed_up is not None:
        print(f"backup:      {report.backed_up}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconcile aligning pull manifest from pi slots + local disk"
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
        help="Write manifest after backup (default: dry-run report only)",
    )
    args = parser.parse_args(argv)

    set_dir = _find_set_dir(Path(args.dest).expanduser(), args.set_id)
    report = reconcile_manifest(set_dir, dry_run=not args.apply)
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"reconcile {set_dir.name} [{mode}]")
    _print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
