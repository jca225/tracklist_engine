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

When the same track_id appears at more than one slot in the set (e.g.
a vocal looped under two different beats), each slot gets its own
filename (`001w2__Artist - Title.m4a`, `023w1__Artist - Title.m4a`)
but only the first occurrence triggers an rsync — subsequent slots
are hard-linked to the cached file, so the duplicates cost zero extra
disk and edits to one (re-tagging, warp markers) propagate to the
others. Ableton sees each linked filename as a distinct clip.

Usage:
    python labeling/pull_set_for_alignment.py <set_id> [--dest ~/aligning]
    python labeling/pull_set_for_alignment.py <set_id> --dry-run
    python labeling/pull_set_for_alignment.py <set_id> --prune           # refresh + delete orphans
    python labeling/pull_set_for_alignment.py <set_id> --prune --dry-run # check mode
    python labeling/pull_set_for_alignment.py --list-recent              # browse candidates

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
    """One *appearance* of a track in the set. The same track_id can
    appear in multiple TrackRows when Two Friends play the same track at
    more than one slot (e.g. a vocal looped under two different beats).
    Each appearance gets its own label and local filename; in the pull
    loop, the first appearance triggers an rsync and subsequent
    appearances are hard-linked to the cached file."""
    track_id: str
    track_audio_id: int
    artist: str
    title: str
    version: str | None          # original | remix | rework | … (track_metadata)
    stem: str                    # regular | acappella | instrumental (track_audio)
    variant: str                 # regular | extended (track_audio)
    # Raw "Artist - Title (Qualifier)" from track_metadata. Carries the full
    # remixer qualifier ("(Syn Cole Remix)") that the version column
    # collapses to "remix". Used to build the filename suffix.
    full_name: str | None
    pi_path: str
    codec: str | None
    duration_s: float | None
    stems: tuple[StemRow, ...]
    row_index: int  # original 1001tracklists row order, used for sorting
    # Label derived from 1001tracklists row prefix: primary rows
    # ("NN ..." or "NN HH:MM ...") become "NNN"; "w/" rows layered
    # under a primary become "NNNw1", "NNNw2", ... The numeric part
    # follows the published section_no (or, if the set has no section
    # prefixes at all, a fall-back per-row counter).
    label: str


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


_PRIMARY_PREFIX_RE = re.compile(r"^\s*(\d+)\b")
_W_PREFIX_RE = re.compile(r"^\s*w/")

# Trailing parenthetical on a track_metadata.full_name like
# "Artist - Title (Syn Cole Remix)". We want the *last* such parens because
# titles occasionally embed inner ones, e.g. "Title (feat. X) (Syn Cole Remix)".
_TRAILING_PAREN_RE = re.compile(r"\(([^()]+)\)\s*$")


_VERSION_DISPLAY: dict[str, str] = {
    "remix": "Remix",
    "rework": "Rework",
    "altversion": "AltVersion",
    "edit": "Edit",
    "bootleg": "Bootleg",
    "mashup": "Mashup",
}


def _qualifier_suffix(
    full_name: str | None,
    version: str | None,
    *,
    stem: str = "regular",
    variant: str = "regular",
) -> str:
    """Build the version/remix suffix for a track filename.

    Prefers the trailing parenthetical of `full_name` ("(Syn Cole Remix)") so
    the annotator can tell which remix a slot is at a glance. Falls back to
    the version axis ("remix" → "(Remix)") only when full_name carries no
    parens. Appends stem/variant qualifiers when not regular."""
    if full_name:
        m = _TRAILING_PAREN_RE.search(full_name)
        if m:
            return f" ({m.group(1).strip()})"
    if version and version != "original":
        label = _VERSION_DISPLAY.get(version, version.title())
        return f" ({label})"
    if stem == "acappella":
        return " (Acappella)"
    if stem == "instrumental":
        return " (Instrumental)"
    if variant == "extended":
        return " (Extended Mix)"
    return ""


def _label_rows(
    rows: list[dict],
) -> list[tuple[str, int, str]]:
    """Given dj_set_rows in row_index order with text_excerpt + track_id,
    return one (label, row_index, track_id) tuple per row.

    Labels follow the 1001tracklists row prefix:
      - "01 ..." / "06 08:29 ..." → primary; emit "NNN" using the
        published section_no (or 1-up from the previous primary when
        section_no isn't parseable).
      - "w/ ..." → layered; emit "NNNwK" where NNN is the current
        primary's number and K is the per-section w counter.
      - Anything else (rare) is treated as primary.

    Tracks played at multiple slots in the set (same track_id appearing
    at multiple row_indexes) get a separate (label, row_index) entry per
    appearance — the caller is responsible for de-duplicating the rsync
    transfer if it wants to (e.g. by hard-linking siblings)."""
    labeled: list[tuple[str, int, str]] = []
    primary_seq = 0  # 1-up counter, used when section_no isn't parseable
    cur_primary_label: str | None = None
    w_counter = 0

    for r in rows:
        track_id = r["track_id"]
        text = r.get("text_excerpt") or ""
        if _W_PREFIX_RE.match(text) and cur_primary_label is not None:
            w_counter += 1
            label = f"{cur_primary_label}w{w_counter}"
        else:
            primary_seq += 1
            m = _PRIMARY_PREFIX_RE.match(text)
            section_no = int(m.group(1)) if m else primary_seq
            cur_primary_label = f"{section_no:03d}"
            w_counter = 0
            label = cur_primary_label

        labeled.append((label, r["row_index"], track_id))
    return labeled


def fetch_tracks(set_id: str) -> list[TrackRow]:
    # First pass: pull ordered dj_set_rows to compute per-track labels
    # that mirror 1001tracklists' published section / "w/" layering.
    row_rows = ssh_sqlite(f"""
        SELECT row_index,
               text_excerpt,
               COALESCE(
                   json_extract(data_attrs_json, '$."data-trackid"'),
                   CASE WHEN json_extract(data_attrs_json, '$."data-isided"') = 'true'
                             AND json_extract(data_attrs_json, '$."data-id"') IS NOT NULL
                        THEN 'tlp' || CAST(json_extract(data_attrs_json, '$."data-id"') AS TEXT)
                   END
               ) AS track_id
        FROM dj_set_rows
        WHERE set_id = '{set_id}'
        ORDER BY row_index;
    """)
    # Drop rows without a usable track_id (rare but possible).
    row_rows = [r for r in row_rows if r.get("track_id")]
    if not row_rows:
        return []
    labeled = _label_rows(row_rows)

    # Second pass: resolve each unique track_id to its best track_audio
    # row + metadata. Platform ordering prefers cleanest sources.
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
                tm.version,
                tm.full_name,
                ta.stem,
                ta.variant,
                ROW_NUMBER() OVER (
                    PARTITION BY ta.track_id
                    ORDER BY ta.is_reference DESC,
                             CASE ta.platform
                                 WHEN 'manual' THEN 0
                                 WHEN 'youtube_music' THEN 1
                                 WHEN 'spotify' THEN 2
                                 WHEN 'soundcloud' THEN 3
                                 WHEN 'youtube' THEN 4
                                 ELSE 5
                             END,
                             ta.downloaded_at DESC
                ) AS pick
            FROM dj_set_track_media_links l
            JOIN track_audio ta ON ta.track_id = l.track_id
            LEFT JOIN track_metadata tm ON tm.track_id = ta.track_id
            WHERE l.set_id = '{set_id}'
        )
        SELECT track_id, track_audio_id, artists_json, meta_title AS title,
               version, full_name, stem, variant, path, codec, duration_s
        FROM ranked
        WHERE pick = 1;
    """)
    if not rows:
        return []
    audio_by_tid: dict[str, dict] = {r["track_id"]: r for r in rows}

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
    for label, row_index, tid in labeled:
        r = audio_by_tid.get(tid)
        if r is None:
            continue  # in dj_set_rows but no track_audio yet; skip
        try:
            artists = json.loads(r["artists_json"]) if r.get("artists_json") else []
        except (json.JSONDecodeError, TypeError):
            artists = []
        artist = " & ".join(a for a in artists if a) or "Unknown"
        out.append(TrackRow(
            track_id=tid,
            track_audio_id=r["track_audio_id"],
            artist=artist,
            title=r.get("title") or "Unknown",
            version=r.get("version"),
            stem=r.get("stem") or "regular",
            variant=r.get("variant") or "regular",
            full_name=r.get("full_name"),
            pi_path=r["path"],
            codec=r.get("codec"),
            duration_s=r.get("duration_s"),
            stems=tuple(stems_by_audio.get(r["track_audio_id"], [])),
            row_index=row_index,
            label=label,
        ))
    # Preserve dj_set_rows order (the published 1001tl ordering).
    out.sort(key=lambda t: t.row_index)
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


def pull_or_link(
    pi_path: str,
    dst: Path,
    pulled: dict[str, Path],
    dry_run: bool,
) -> bool:
    """First time we see a pi_path in this run, rsync from pi-storage to
    `dst` and register it. On subsequent appearances (same audio used at
    multiple slots in the set), hard-link `dst` to the already-pulled
    sibling so both filenames share one inode — zero extra disk, and
    Ableton sees each as a distinct clip.

    Returns True on success. dry_run treats both paths as no-ops but
    still registers the planned dst so siblings know they have a parent."""
    sibling = pulled.get(pi_path)
    if sibling is None:
        if dry_run:
            pulled[pi_path] = dst
            return True
        ok = rsync(pi_path, dst, dry_run=False)
        if ok:
            pulled[pi_path] = dst
        return ok
    if dst == sibling:
        return True
    if dry_run:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists():
            if dst.stat().st_ino == sibling.stat().st_ino:
                return True  # already linked from a prior run
            dst.unlink()
        dst.hardlink_to(sibling)
        return True
    except OSError as e:
        print(f"  ! hardlink failed ({e}); falling back to rsync", file=sys.stderr)
        return rsync(pi_path, dst, dry_run=False)


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
    # First-pull cache so re-played tracks (and their stems) become hard
    # links into the first appearance's file instead of duplicate rsyncs.
    pulled_tracks: dict[str, Path] = {}
    pulled_stems: dict[str, Path] = {}
    for i, t in enumerate(tracks, start=1):
        ext = Path(t.pi_path).suffix or ".m4a"
        suffix = _qualifier_suffix(
            t.full_name, t.version, stem=t.stem, variant=t.variant,
        )
        name = f"{t.label}__{sanitize(t.artist)} - {sanitize(t.title)}{suffix}{ext}"
        dst = tracks_dir / name
        keep_paths.add(dst)
        n_stems = len(t.stems) if not args.no_stems else 0
        stems_note = f"  +{n_stems} stems" if n_stems else ""
        replay_note = "  (replay → hardlink)" if t.pi_path in pulled_tracks else ""
        print(f"[{i:03d}/{len(tracks)}] {t.label}  {t.artist} - {t.title}{stems_note}{replay_note}")

        track_entry: dict = {
            "track_id": t.track_id,
            "track_audio_id": t.track_audio_id,
            "label": t.label,
            "artist": t.artist,
            "title": t.title,
            "version": t.version,
            "stem": t.stem,
            "variant": t.variant,
            "axes_key": f"{t.version or 'original'}__{t.stem}__{t.variant}",
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
            pulled_tracks.setdefault(t.pi_path, dst)
            for s in (t.stems if not args.no_stems else ()):
                stem_dst = stems_dir / stem_subdir_name / f"{s.stem_name}{Path(s.pi_path).suffix or '.m4a'}"
                keep_paths.add(stem_dst)
                pulled_stems.setdefault(s.pi_path, stem_dst)
                print(f"    -> {stem_dst}")
                track_entry["stems"][s.stem_name] = str(stem_dst)
            succeeded.append(track_entry)
            continue

        if not pull_or_link(t.pi_path, dst, pulled_tracks, dry_run=False):
            continue

        if not args.no_stems:
            for s in t.stems:
                stem_ext = Path(s.pi_path).suffix or ".m4a"
                stem_dst = stems_dir / stem_subdir_name / f"{s.stem_name}{stem_ext}"
                keep_paths.add(stem_dst)
                if pull_or_link(s.pi_path, stem_dst, pulled_stems, dry_run=False):
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
