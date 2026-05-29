"""Rename files in existing ~/aligning/<set>/ folders to match the new
section-number + w-suffix naming emitted by pull_set_for_alignment.py.

Reads the folder's manifest.json (old labels) and queries pi-storage for
the new labels per track_id, then renames the m4a, its .asd, and the
matching stems subdir. Preserves any user-tag suffixes
(`[NNNbpm KK]`, `[no-features]`) inside the filename.

Skips folders explicitly excluded (e.g. BB12, which has unsaved Ableton
work). Dry-run by default; pass --apply to commit renames.

Usage:
    python labeling/migrate_aligning_naming.py                 # dry-run all
    python labeling/migrate_aligning_naming.py --apply         # apply all
    python labeling/migrate_aligning_naming.py --apply 1fsnxchk  # just one
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pull_set_for_alignment import fetch_tracks  # noqa: E402

ALIGN_ROOT = Path.home() / "aligning"

# BB12 is excluded because the user has unsaved Ableton work tied to the
# old filenames. Remove from this set when it's safe to touch.
EXCLUDED_SETS = {"1fsnxchk"}


def find_folder(set_id: str) -> Path | None:
    """Match the folder name prefix (set_id is the first token before __)."""
    if not ALIGN_ROOT.exists():
        return None
    for sub in ALIGN_ROOT.iterdir():
        if sub.is_dir() and sub.name.startswith(f"{set_id}__"):
            return sub
    return None


def discover_folders() -> dict[str, Path]:
    """Map set_id -> folder for every ~/aligning/<set_id>__... directory."""
    out: dict[str, Path] = {}
    for sub in ALIGN_ROOT.iterdir():
        if not sub.is_dir():
            continue
        if "__" not in sub.name:
            continue
        set_id = sub.name.split("__", 1)[0]
        out[set_id] = sub
    return out


def plan_renames(folder: Path, set_id: str) -> list[tuple[Path, Path]]:
    """Return ordered list of (src, dst) renames for this folder.
    Walks tracks/, tracks/*.asd, and stems/<subdir>/."""
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        print(f"  SKIP {set_id}: no manifest.json")
        return []
    manifest = json.loads(manifest_path.read_text())

    print(f"  querying pi-storage for new labels...")
    new_tracks = fetch_tracks(set_id)
    new_label_by_tid = {t.track_id: t.label for t in new_tracks}

    renames: list[tuple[Path, Path]] = []
    tracks_dir = folder / "tracks"
    stems_dir = folder / "stems"

    for entry in manifest["tracks"]:
        tid = entry["track_id"]
        old_local_path = Path(entry["local_path"])
        old_basename = old_local_path.name
        if "__" not in old_basename:
            continue
        old_prefix, _ = old_basename.split("__", 1)
        new_label = new_label_by_tid.get(tid)
        if new_label is None:
            print(f"  WARN: track_id={tid} not in new fetch — leaving alone")
            continue
        if old_prefix == new_label:
            continue  # already matches

        # Match files by the full old basename stem (no extension), so
        # that user-renamed variants with a `[NNNbpm KK]` or
        # `[no-features]` tag still get caught (the tag sits between the
        # artist-title and the extension). The prefix-only match used
        # previously was too coarse — when a partial re-pull deposited
        # other tracks under the same `NNN__` prefix, we'd rename them
        # too. Matching the full track-identifying stem avoids that.
        old_stem = old_local_path.stem  # e.g. "076__Thomas Gold - Saints & Sinners (Remix)"
        old_after_prefix = old_stem.split("__", 1)[1]  # "Thomas Gold - Saints & Sinners (Remix)"

        if tracks_dir.exists():
            for f in tracks_dir.iterdir():
                if not f.name.startswith(f"{old_prefix}__"):
                    continue
                # Strip prefix + ext to get the user-visible tail.
                if not f.is_file():
                    continue
                rest = f.name.split("__", 1)[1]
                # rest looks like "Artist - Title (Suffix).m4a" or
                # "Artist - Title (Suffix) [tag].m4a" or
                # "Artist - Title (Suffix).m4a.asd"
                if not (rest.startswith(old_after_prefix)
                        or rest.startswith(old_after_prefix + " ")):
                    continue
                new_name = f"{new_label}__{rest}"
                renames.append((f, tracks_dir / new_name))

        # Stems subdir — same matching rules.
        if stems_dir.exists():
            for sub in stems_dir.iterdir():
                if not sub.is_dir():
                    continue
                if not sub.name.startswith(f"{old_prefix}__"):
                    continue
                rest = sub.name.split("__", 1)[1]
                if not (rest == old_after_prefix
                        or rest.startswith(old_after_prefix + " ")):
                    continue
                new_sub_name = f"{new_label}__{rest}"
                renames.append((sub, stems_dir / new_sub_name))

    return renames


def _size_recursive(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return 0


def rewrite_manifest(folder: Path, set_id: str, dry_run: bool) -> bool:
    """Update local_path and stems paths in manifest.json to reflect the
    new label scheme. Reads fresh labels from pi-storage via fetch_tracks,
    then rewrites the prefix in each path. Files are NOT moved here —
    that's plan_renames + apply_renames. This just keeps the manifest
    consistent with disk."""
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    new_tracks = fetch_tracks(set_id)
    new_label_by_tid = {t.track_id: t.label for t in new_tracks}

    changed = 0
    for entry in manifest["tracks"]:
        tid = entry["track_id"]
        new_label = new_label_by_tid.get(tid)
        if new_label is None:
            continue
        for key in ("local_path",):
            old = entry.get(key)
            if not old:
                continue
            p = Path(old)
            if "__" not in p.name:
                continue
            prefix, rest = p.name.split("__", 1)
            if prefix == new_label:
                continue
            new_name = f"{new_label}__{rest}"
            entry[key] = str(p.with_name(new_name))
            changed += 1
        stems = entry.get("stems") or {}
        for sname, spath in list(stems.items()):
            if not spath:
                continue
            sp = Path(spath)
            # stem path is .../<subdir>/<stem_name>.<ext> where subdir
            # has the NN__Artist - Title format
            parts = list(sp.parts)
            # Find the subdir part (its name starts with "<old>__")
            for i, part in enumerate(parts):
                if "__" not in part:
                    continue
                prefix = part.split("__", 1)[0]
                if not prefix.isdigit() and "w" not in prefix:
                    continue
                # Only rewrite if this is the stems subdir; identify by
                # looking like NN[wK]__Artist - Title.
                # Simpler: rewrite any "NN__"/"NNwK__" prefix to new label.
                rest_of_part = part.split("__", 1)[1]
                # We only want to rewrite the per-track stem subdir,
                # which is parent of the file. Heuristic: rewrite only
                # if the next part is a filename with stem extension.
                if i == len(parts) - 2 and "." in parts[-1]:
                    new_part = f"{new_label}__{rest_of_part}"
                    if new_part != part:
                        parts[i] = new_part
                        entry["stems"][sname] = str(Path(*parts))
                        changed += 1
                    break

    if changed == 0:
        return False
    marker = " (dry-run)" if dry_run else ""
    print(f"  manifest: {changed} path(s) updated{marker}")
    if not dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2))
    return True


def apply_renames(pairs: list[tuple[Path, Path]], dry_run: bool) -> int:
    """Apply src->dst renames.

    When dst already exists, the killed re-pull placed a new-named copy
    next to the old-named src. We resolve: keep the larger (the rsync
    used --inplace, so a partial transfer is shorter than the complete
    file from the original pull). The smaller of the two is removed."""
    n = 0
    for src, dst in pairs:
        if dst.exists():
            src_size = _size_recursive(src)
            dst_size = _size_recursive(dst)
            marker = " (dry-run)" if dry_run else ""
            if src_size > dst_size:
                # src is the more-complete copy; overwrite dst.
                print(f"  conflict: {src.name} ({src_size}) > {dst.name} ({dst_size}); overwrite{marker}")
                if not dry_run:
                    if dst.is_dir():
                        import shutil
                        shutil.rmtree(dst)
                    else:
                        dst.unlink()
                    src.rename(dst)
                n += 1
            elif dst_size > src_size or src_size == dst_size:
                # dst is at least as good as src; remove src.
                print(f"  conflict: drop {src.name} ({src_size}), keep {dst.name} ({dst_size}){marker}")
                if not dry_run:
                    if src.is_dir():
                        import shutil
                        shutil.rmtree(src)
                    else:
                        src.unlink()
                n += 1
            continue
        marker = " (dry-run)" if dry_run else ""
        print(f"  {src.name}  ->  {dst.name}{marker}")
        if not dry_run:
            src.rename(dst)
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("set_ids", nargs="*",
                    help="Specific set_ids to migrate (default: all in ~/aligning/)")
    ap.add_argument("--apply", action="store_true",
                    help="Actually rename (default: dry-run)")
    args = ap.parse_args()

    if args.set_ids:
        folders = {sid: find_folder(sid) for sid in args.set_ids}
        folders = {k: v for k, v in folders.items() if v is not None}
    else:
        folders = discover_folders()

    if not folders:
        print("No folders found in ~/aligning/")
        return 1

    total = 0
    for set_id, folder in folders.items():
        if set_id in EXCLUDED_SETS:
            print(f"\n=== {set_id}  ({folder.name}) — EXCLUDED, skipping ===")
            continue
        print(f"\n=== {set_id}  ({folder.name}) ===")
        renames = plan_renames(folder, set_id)
        if not renames:
            print("  (nothing to rename)")
        else:
            total += apply_renames(renames, dry_run=not args.apply)
        rewrite_manifest(folder, set_id, dry_run=not args.apply)

    verb = "would rename" if not args.apply else "renamed"
    print(f"\n{verb} {total} entries.")
    if not args.apply and total:
        print("Re-run with --apply to commit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
