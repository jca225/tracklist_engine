"""Pull a single DJ set's mix recording + reference tracks into a local
working folder, ready to drag into Ableton for alignment.

Runs on the Mac. Queries pi-storage's canonical DB over SSH, then
rsyncs the audio files locally. The folder is intended as ephemeral —
delete it once alignment data has been written back to the DB.

The local folder is a **read-replica of pi-storage**: only this script
writes to it (rsync is the only producer), and re-running it on an
existing folder is a delta refresh (rsync archive mode only transfers
changed files). With `--prune`, the script additionally deletes local
audio files under `tracks/` and `stems/` that pi-storage no longer
considers part of the set (re-resolved track ids, regenerated stems,
replaced audio). Without `--prune`, those orphans accumulate silently.

Usage:
    python scripts/pull_set_for_alignment.py <set_id> [--dest ~/aligning]
    python scripts/pull_set_for_alignment.py <set_id> --dry-run
    python scripts/pull_set_for_alignment.py <set_id> --prune           # refresh + delete orphans
    python scripts/pull_set_for_alignment.py <set_id> --prune --dry-run # check mode
    python scripts/pull_set_for_alignment.py --list-recent              # browse candidates

Output layout:
    ~/aligning/<set_id>__<sanitized-title>/
        mix.<ext>                    # the DJ set recording
        tracks/
            001__Artist - Title.<ext>
            002__Artist - Title.<ext>
            ...
        manifest.json                # set_id, track ids, paths, durations

Prune safety:
  - Pruning is scoped to (a) files directly inside `tracks/`, and
    (b) files inside stem subdirs the pull plan itself created.
  - Stem subdirs whose name does NOT match the current plan (e.g.
    user-renamed `stems/146__... [126bpm 10B]/`, ad-hoc experiment
    folders) are treated as user territory and skipped entirely.
  - Files whose name carries the human-annotator rename tag
    (`[NNNbpm KK]` for tempo+key, `[no-features]` for skip-flag) are
    skipped — those renames are an intentional one-sided Mac-side
    mutation that exposes tempo/key inline for Ableton's clip browser.
    The rename never propagates back to pi-storage.
  - Within scope, only files with audio extensions
    (.m4a/.mp3/.flac/.ogg/.wav/.opus/.aac) are deletion candidates.
    Ableton `.asd`, `.als`, `manifest.json`, and hand-placed files at
    the folder root are never touched.
  - The keep-set is built from the *plan* (pi-storage's current view),
    not from `succeeded`, so a transient rsync failure does not cause
    its file to be misclassified as an orphan.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

PI_HOST = "pi-storage"
PI_DB = "/mnt/storage/data/db/music_database.db"

_BAD = re.compile(r'[\x00-\x1f/\\:*?"<>|]')


def sanitize(s: str, max_len: int = 120) -> str:
    s = unicodedata.normalize("NFC", s or "")
    s = _BAD.sub("-", s)
    s = re.sub(r"\s+", " ", s).strip(" .")
    return (s[:max_len].rstrip(" .") or "_")


@dataclass(frozen=True)
class StemRow:
    stem_name: str  # 'vocals' | 'instrumental' | ...
    pi_path: str


@dataclass(frozen=True)
class TrackRow:
    track_id: str
    track_audio_id: int
    artist: str
    title: str
    version_tag: str | None
    pi_path: str
    codec: str | None
    duration_s: float | None
    stems: tuple[StemRow, ...]


@dataclass(frozen=True)
class MixRow:
    set_id: str
    title: str
    pi_path: str
    codec: str | None
    duration_s: float | None


def ssh_sqlite(query: str) -> list[dict]:
    """Run a sqlite3 query on pi-storage and return parsed JSON rows.
    `.mode json` emits a JSON array; '' for no results."""
    script = f".mode json\n{query.strip()}\n"
    cmd = ["ssh", PI_HOST, f"sqlite3 {PI_DB}"]
    out = subprocess.run(
        cmd, input=script, capture_output=True, text=True, check=True,
    )
    body = out.stdout.strip()
    if not body:
        return []
    return json.loads(body)


def fetch_mix(set_id: str) -> MixRow | None:
    rows = ssh_sqlite(f"""
        SELECT s.set_id, s.title, sa.path, sa.codec, sa.duration_s
        FROM dj_sets s
        JOIN set_audio sa ON sa.set_id = s.set_id
        WHERE s.set_id = '{set_id}'
        ORDER BY sa.is_reference DESC, sa.set_audio_id ASC
        LIMIT 1;
    """)
    if not rows:
        return None
    r = rows[0]
    return MixRow(
        set_id=r["set_id"],
        title=r.get("title") or set_id,
        pi_path=r["path"],
        codec=r.get("codec"),
        duration_s=r.get("duration_s"),
    )


def fetch_tracks(set_id: str) -> list[TrackRow]:
    # is_reference is rarely set, so pick the "best" track_audio row per
    # track_id: prefer is_reference=1, then youtube_music (cleanest source),
    # then spotify, then any other platform, breaking ties by most recent.
    rows = ssh_sqlite(f"""
        WITH ranked AS (
            SELECT
                ta.track_id,
                ta.track_audio_id,
                ta.path,
                ta.codec,
                ta.duration_s,
                tm.artists_json,
                tm.title           AS meta_title,
                tm.version_tag,
                MIN(l.track_media_id) OVER (PARTITION BY ta.track_id) AS first_seen,
                ROW_NUMBER() OVER (
                    PARTITION BY ta.track_id
                    ORDER BY ta.is_reference DESC,
                             CASE ta.platform
                                 WHEN 'youtube_music' THEN 0
                                 WHEN 'spotify' THEN 1
                                 WHEN 'soundcloud' THEN 2
                                 WHEN 'youtube' THEN 3
                                 ELSE 4
                             END,
                             ta.downloaded_at DESC
                ) AS pick
            FROM dj_set_track_media_links l
            JOIN track_audio ta ON ta.track_id = l.track_id
            LEFT JOIN track_metadata tm ON tm.track_id = ta.track_id
            WHERE l.set_id = '{set_id}'
        )
        SELECT track_id, track_audio_id, artists_json, meta_title AS title,
               version_tag, path, codec, duration_s
        FROM ranked
        WHERE pick = 1
        ORDER BY first_seen;
    """)
    if not rows:
        return []

    # Bulk-fetch stems for all picked track_audio_ids in one query.
    audio_ids = [r["track_audio_id"] for r in rows]
    id_list = ",".join(str(i) for i in audio_ids)
    stem_rows = ssh_sqlite(f"""
        SELECT track_audio_id, stem_name, path
        FROM track_stems
        WHERE track_audio_id IN ({id_list});
    """)
    stems_by_audio: dict[int, list[StemRow]] = {}
    for s in stem_rows:
        stems_by_audio.setdefault(s["track_audio_id"], []).append(
            StemRow(stem_name=s["stem_name"], pi_path=s["path"])
        )

    out: list[TrackRow] = []
    for r in rows:
        try:
            artists = json.loads(r["artists_json"]) if r.get("artists_json") else []
        except (json.JSONDecodeError, TypeError):
            artists = []
        artist = " & ".join(a for a in artists if a) or "Unknown"
        out.append(TrackRow(
            track_id=r["track_id"],
            track_audio_id=r["track_audio_id"],
            artist=artist,
            title=r.get("title") or "Unknown",
            version_tag=r.get("version_tag"),
            pi_path=r["path"],
            codec=r.get("codec"),
            duration_s=r.get("duration_s"),
            stems=tuple(stems_by_audio.get(r["track_audio_id"], [])),
        ))
    return out


def list_recent(limit: int = 20) -> None:
    rows = ssh_sqlite(f"""
        SELECT s.set_id, s.title,
               COUNT(DISTINCT ta.track_id) AS n_tracks,
               (SELECT 1 FROM set_audio sa WHERE sa.set_id = s.set_id LIMIT 1) AS has_mix
        FROM dj_sets s
        JOIN dj_set_track_media_links l ON l.set_id = s.set_id
        JOIN track_audio ta ON ta.track_id = l.track_id
        GROUP BY s.set_id
        HAVING has_mix = 1 AND n_tracks >= 10
        ORDER BY n_tracks DESC
        LIMIT {limit};
    """)
    print(f"{'set_id':<14} {'tracks':>6}  title")
    print("-" * 78)
    for r in rows:
        print(f"{r['set_id']:<14} {r['n_tracks']:>6}  {(r.get('title') or '')[:55]}")


# Files we may delete during --prune. Other extensions (.asd, .als,
# .json, etc.) are never pruning candidates.
_AUDIO_EXTS = frozenset({".m4a", ".mp3", ".flac", ".ogg", ".wav", ".opus", ".aac"})

# Patterns emitted by the human annotator's rename workflow. The
# annotator renames track files and stem subdirs to expose tempo + key
# inline so Ableton's clip browser shows them without opening each clip
# (and `[no-features]` flags tracks that have no pi-storage features
# yet, so the annotator knows to skip them). Prune treats any name
# containing one of these patterns as user-modified and leaves it
# alone. The rename is one-sided: it never propagates back to
# pi-storage, so re-pulls produce fresh un-tagged copies alongside the
# user's tagged ones — that's expected, and the annotator either
# re-tags them or ignores them.
#
# Known annotator tags:
#   `[NNNbpm KK]`   — tempo + Camelot key, e.g. [126bpm 8B], [84bpm 6B]
#   `[no-features]` — no Essentia row in pi-storage for this track
_USER_TAG_PATTERN = re.compile(
    r"\[\s*\d+\s*bpm\b|\[no-features\]",
    re.IGNORECASE,
)


def _user_tagged(name: str) -> bool:
    return bool(_USER_TAG_PATTERN.search(name))


def prune_orphans(
    tracks_dir: Path,
    stems_dir: Path | None,
    keep: set[Path],
    dry_run: bool,
) -> tuple[int, int]:
    """Delete audio files under tracks/ and stems/ that aren't in `keep`.

    Scope:
      - Only files directly inside `tracks_dir`.
      - Only files inside stem subdirs the pull plan itself created.
      - Subdirs whose name doesn't match the plan (e.g. user-renamed
        `stems/146__... [126bpm 10B]/`, ad-hoc experiment folders) are
        treated as user territory and skipped entirely.
      - Files whose name carries the human-annotator BPM/key tag (see
        `_USER_TAG_PATTERN`) are likewise skipped — those are user
        renames, not orphans.

    Returns (n_files_pruned, n_dirs_pruned)."""
    keep_resolved = {p.resolve() for p in keep}
    # plan_subdirs: parent dirs we own. Anything else in stems/ is
    # treated as user territory.
    plan_subdirs = {p.parent.resolve() for p in keep}
    pruned_files = 0

    def _consider(f: Path) -> bool:
        if not f.is_file():
            return False
        if f.suffix.lower() not in _AUDIO_EXTS:
            return False
        if f.parent.resolve() not in plan_subdirs:
            return False
        if f.resolve() in keep_resolved:
            return False
        if _user_tagged(f.name):
            return False
        return True

    # tracks/ is flat — one directory, all files in plan_subdirs.
    if tracks_dir.exists():
        for f in tracks_dir.iterdir():
            if not _consider(f):
                continue
            print(f"  - prune: {f.relative_to(tracks_dir.parent)}"
                  + (" (dry-run)" if dry_run else ""))
            if not dry_run:
                f.unlink()
            pruned_files += 1

    # stems/<subdir>/<file> — only descend into subdirs that the plan
    # owns. Other subdirs (renamed by user, experiment folders) are
    # skipped entirely.
    if stems_dir is not None and stems_dir.exists():
        for sub in stems_dir.iterdir():
            if not sub.is_dir():
                continue
            if sub.resolve() not in plan_subdirs:
                continue
            for f in sub.iterdir():
                if not _consider(f):
                    continue
                print(f"  - prune: {f.relative_to(tracks_dir.parent)}"
                      + (" (dry-run)" if dry_run else ""))
                if not dry_run:
                    f.unlink()
                pruned_files += 1

    # Reap empty plan-owned stem subdirectories.
    pruned_dirs = 0
    if stems_dir is not None and stems_dir.exists():
        for sub in stems_dir.iterdir():
            if not sub.is_dir():
                continue
            if sub.resolve() not in plan_subdirs:
                continue
            if any(sub.iterdir()):
                continue
            print(f"  - prune empty dir: {sub.relative_to(tracks_dir.parent)}"
                  + (" (dry-run)" if dry_run else ""))
            if not dry_run:
                sub.rmdir()
            pruned_dirs += 1
    return pruned_files, pruned_dirs


def rsync(src_remote: str, dst_local: Path, dry_run: bool) -> bool:
    """rsync one file from pi-storage to a local path. Includes a 60s idle
    timeout so a dropped Tailscale connection doesn't wedge the process."""
    dst_local.parent.mkdir(parents=True, exist_ok=True)
    flags = ["-aL", "--partial", "--inplace", "--timeout=60"]
    if dry_run:
        flags.append("--dry-run")
    cmd = ["rsync", *flags, f"{PI_HOST}:{src_remote}", str(dst_local)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ! rsync failed: {result.stderr.strip()}", file=sys.stderr)
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("set_id", nargs="?", help="dj_sets.set_id (e.g. 2vpur281)")
    ap.add_argument("--dest", default="~/aligning",
                    help="local working dir root (default: ~/aligning)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would happen, no rsync")
    ap.add_argument("--list-recent", action="store_true",
                    help="show sets with downloaded mix + ≥10 tracks")
    ap.add_argument("--no-stems", action="store_true",
                    help="skip pulling vocals/instrumental stems")
    ap.add_argument("--prune", action="store_true",
                    help="after pulling, delete local audio files under "
                         "tracks/ and stems/ that pi-storage no longer "
                         "considers part of the set. Combine with --dry-run "
                         "to preview deletions without touching anything.")
    args = ap.parse_args()

    if args.list_recent:
        list_recent()
        return 0

    if not args.set_id:
        ap.error("set_id required (or use --list-recent to browse)")

    if not shutil.which("rsync"):
        print("ERROR: rsync not on PATH", file=sys.stderr)
        return 2

    print(f"Fetching set metadata for {args.set_id}...")
    mix = fetch_mix(args.set_id)
    if mix is None:
        print(f"ERROR: no set_audio row for {args.set_id}", file=sys.stderr)
        return 1
    tracks = fetch_tracks(args.set_id)
    if not tracks:
        print(f"ERROR: no reference tracks linked to set {args.set_id}", file=sys.stderr)
        return 1

    folder_name = f"{args.set_id}__{sanitize(mix.title)}"
    dest_root = Path(args.dest).expanduser() / folder_name
    tracks_dir = dest_root / "tracks"
    stems_dir = dest_root / "stems"

    n_with_stems = sum(1 for t in tracks if t.stems)
    print(f"\nSet: {mix.title}")
    print(f"  Mix:    {Path(mix.pi_path).name}  ({mix.duration_s and f'{mix.duration_s/60:.1f} min'})")
    print(f"  Tracks: {len(tracks)}")
    if not args.no_stems:
        print(f"  Stems:  {n_with_stems}/{len(tracks)} tracks have stems available")
    print(f"  Dest:   {dest_root}\n")

    mix_ext = Path(mix.pi_path).suffix or ".m4a"
    mix_dst = dest_root / f"mix{mix_ext}"
    print(f"[mix] {mix.pi_path} -> {mix_dst}")
    if not args.dry_run:
        if not rsync(mix.pi_path, mix_dst, args.dry_run):
            return 1

    # Build the "what should exist locally" keep-set as we plan the pull.
    # `--prune` uses it to delete orphans afterward. We populate it from
    # the plan (pi-storage's current view), not from `succeeded` (which
    # would let a transient rsync failure incorrectly mark files as
    # orphans).
    keep_paths: set[Path] = {mix_dst}

    succeeded: list[dict] = []
    for i, t in enumerate(tracks, start=1):
        ext = Path(t.pi_path).suffix or ".m4a"
        suffix = f" ({t.version_tag})" if t.version_tag else ""
        name = f"{i:03d}__{sanitize(t.artist)} - {sanitize(t.title)}{suffix}{ext}"
        dst = tracks_dir / name
        keep_paths.add(dst)
        n_stems = len(t.stems) if not args.no_stems else 0
        stems_note = f"  +{n_stems} stems" if n_stems else ""
        print(f"[{i:03d}/{len(tracks)}] {t.artist} - {t.title}{stems_note}")

        track_entry: dict = {
            "track_id": t.track_id,
            "track_audio_id": t.track_audio_id,
            "artist": t.artist,
            "title": t.title,
            "version_tag": t.version_tag,
            "local_path": str(dst),
            "pi_path": t.pi_path,
            "duration_s": t.duration_s,
            "stems": {},
        }

        # Stem subdir mirrors the track filename minus extension, so the
        # human-readable name pairs visibly with its track.
        stem_subdir_name = Path(name).stem  # e.g. "001__Artist - Title"

        if args.dry_run:
            print(f"    -> {dst}")
            for s in (t.stems if not args.no_stems else ()):
                stem_dst = stems_dir / stem_subdir_name / f"{s.stem_name}{Path(s.pi_path).suffix or '.m4a'}"
                keep_paths.add(stem_dst)
                print(f"    -> {stem_dst}")
                track_entry["stems"][s.stem_name] = str(stem_dst)
            succeeded.append(track_entry)
            continue

        if not rsync(t.pi_path, dst, dry_run=False):
            continue

        if not args.no_stems:
            for s in t.stems:
                stem_ext = Path(s.pi_path).suffix or ".m4a"
                stem_dst = stems_dir / stem_subdir_name / f"{s.stem_name}{stem_ext}"
                keep_paths.add(stem_dst)
                if rsync(s.pi_path, stem_dst, dry_run=False):
                    track_entry["stems"][s.stem_name] = str(stem_dst)

        succeeded.append(track_entry)

    manifest = {
        "set_id": args.set_id,
        "title": mix.title,
        "mix_local_path": str(mix_dst),
        "mix_pi_path": mix.pi_path,
        "mix_duration_s": mix.duration_s,
        "tracks": succeeded,
    }
    if not args.dry_run:
        (dest_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if args.prune:
        print("\nPruning orphaned local audio files...")
        n_files, n_dirs = prune_orphans(
            tracks_dir=tracks_dir,
            # With --no-stems we don't reconcile stems this run: a
            # previously-pulled stem might still be wanted, and we
            # didn't query pi-storage's current stem list. Leave them
            # alone.
            stems_dir=None if args.no_stems else stems_dir,
            keep=keep_paths,
            dry_run=args.dry_run,
        )
        if n_files == 0 and n_dirs == 0:
            print("  (nothing to prune)")
        else:
            verb = "would delete" if args.dry_run else "deleted"
            print(f"  {verb} {n_files} file(s), {n_dirs} empty dir(s)")

    print(f"\nDone. {len(succeeded)}/{len(tracks)} tracks pulled.")
    print(f"Drag {dest_root} into Ableton.")
    print(f"When finished: rm -rf {dest_root!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
