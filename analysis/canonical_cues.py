"""Download canonical original audio and compute shared cue points.

For each canonical track_id, ensure a `track_audio` row exists with
`variant_tag='original'`. If the DJ scraped an acapella or instrumental
variant (cue-detr performs poorly on those — trained on EDM with full
spectral content), we additionally fetch the full-song version so cue
points can be computed on dense audio. Those cue points are stored in
`canonical_track_cue_points` keyed by track_id and shared across all
variants of the same song.

Run:
    venvs/audio/bin/python -m audio_pipeline.analysis.canonical_cues \\
        --set-id 2nvzlh2k
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# Audio storage root. Default points at the canonical pi-storage path; override
# via TRACKLIST_AUDIO_ROOT for local-scratch runs (see CLAUDE.md → Storage).
_AUDIO_ROOT = Path(os.environ.get("TRACKLIST_AUDIO_ROOT", "/mnt/storage"))
_ORIGINALS_DIR = _AUDIO_ROOT / "objects" / "canonical_originals"
DEFAULT_DB_PATH = _REPO_ROOT / "data/db/music_database.db"

# Cue-detr sensitivity: the default (0.9) is tuned for strong EDM drops.
# Lowering it surfaces section-boundary cues in pop and other genres.
# Calibrated empirically on BB11 refs (see conversation log).
CUE_DETR_SENSITIVITY: float = 0.5


@dataclass(frozen=True)
class OriginalSpec:
    """A canonical track for which we need an 'original' variant."""
    track_id: str
    artist: str
    title: str


# Hand-curated for BB11's 5 GT refs. When expanding to other sets, this
# table should be derived automatically from tracklist metadata (strip
# version-tag suffixes like '(Acappella)' / '(Instrumental)').
BB11_ORIGINALS: tuple[OriginalSpec, ...] = (
    OriginalSpec("g8gtgdx",  "Bastille",           "Good Grief Don Diablo Remix"),
    OriginalSpec("26b4gz6f", "The Fray",           "How to Save a Life"),
    OriginalSpec("4gy6y1p",  "Carly Rae Jepsen",   "Call Me Maybe"),
    OriginalSpec("2m5wh0t5", "Gnash",              "I Hate U I Love U Olivia O'Brien"),
    # ntm7wqx (Antoine Delvig & Paul Vinx - Blondies) already has an original variant.
)


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


# ---- yt-dlp search + download --------------------------------------------

def _resolve_search(query: str) -> tuple[str, str] | None:
    """Returns (youtube_id, canonical_url) for the top match of `query`."""
    try:
        proc = subprocess.run(
            ["venvs/audio/bin/yt-dlp", "--print", "%(id)s\t%(webpage_url)s",
             "--no-download", f"ytsearch1:{query}"],
            cwd=_REPO_ROOT, check=True, capture_output=True, text=True, timeout=60,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"  ! search failed for {query!r}: {e}", file=sys.stderr)
        return None
    line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    if "\t" not in line:
        return None
    vid, url = line.split("\t", 1)
    return vid, url


def _download(query: str, out_path: Path) -> bool:
    """Download top ytsearch match of `query` to `out_path` as m4a.
    Returns True on success."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmpl = str(out_path.with_suffix(".%(ext)s"))
    try:
        subprocess.run(
            ["venvs/audio/bin/yt-dlp",
             "-f", "bestaudio[ext=m4a]/bestaudio",
             "-x", "--audio-format", "m4a",
             "-o", tmpl, "--no-warnings", "--quiet",
             f"ytsearch1:{query}"],
            cwd=_REPO_ROOT, check=True, timeout=300,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"  ! download failed: {e}", file=sys.stderr)
        return False
    return out_path.exists()


# ---- DB helpers -----------------------------------------------------------

def _find_original_audio(conn: sqlite3.Connection, track_id: str) -> tuple[int, str] | None:
    r = conn.execute(
        "SELECT track_audio_id, path FROM track_audio WHERE track_id=? AND variant_tag='original'",
        (track_id,),
    ).fetchone()
    return (int(r["track_audio_id"]), str(r["path"])) if r else None


def _insert_original_audio(
    conn: sqlite3.Connection, *, track_id: str, path: str, yt_id: str, yt_url: str,
) -> int:
    """Insert a new track_audio row with variant_tag='original'. Returns
    the track_audio_id. If the row already exists (same platform/player_id)
    returns the existing id."""
    conn.execute(
        """
        INSERT OR IGNORE INTO track_audio
            (track_id, platform, source_url, player_id, path, codec,
             is_reference, variant_tag)
        VALUES (?, 'youtube', ?, ?, ?, 'm4a', 0, 'original')
        """,
        (track_id, yt_url, yt_id, path),
    )
    conn.commit()
    r = conn.execute(
        "SELECT track_audio_id FROM track_audio WHERE track_id=? AND platform='youtube' AND player_id=?",
        (track_id, yt_id),
    ).fetchone()
    return int(r["track_audio_id"])


def _has_canonical_cues(conn: sqlite3.Connection, track_id: str) -> bool:
    r = conn.execute(
        "SELECT 1 FROM canonical_track_cue_points WHERE track_id=?", (track_id,)
    ).fetchone()
    return r is not None


def _write_canonical_cues(
    conn: sqlite3.Connection, *, track_id: str, cues: list[float],
    source_track_audio_id: int, source_variant_tag: str, sensitivity: float,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO canonical_track_cue_points
            (track_id, cue_points_json, source_track_audio_id,
             source_variant_tag, cue_detr_sensitivity)
        VALUES (?, ?, ?, ?, ?)
        """,
        (track_id, json.dumps(cues), source_track_audio_id, source_variant_tag, sensitivity),
    )
    conn.commit()


# ---- cue-detr -------------------------------------------------------------

def _run_cue_detr(audio_path: Path, sensitivity: float) -> list[float]:
    """Run cue-detr on a single audio file, return sorted cue points in seconds.

    Works by copying the file into a scratch directory since cue-detr's
    public API is directory-based (predict_cue_points_for_dir).
    """
    # Lazy import — cue-detr loads heavy models on first call.
    sys.path.insert(0, str(_REPO_ROOT / "cue-detr"))
    from cue_points import predict_cue_points_for_dir  # type: ignore[import-not-found]

    scratch = _REPO_ROOT / "data/cache/cue_detr_scratch"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    link = scratch / audio_path.name
    try:
        link.symlink_to(audio_path)
    except OSError:
        shutil.copy(audio_path, link)

    try:
        result = predict_cue_points_for_dir(
            tracks_dir=str(scratch),
            sensitivity=sensitivity,
            print_points=False,
            write_output=False,
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    for fname, points in result.items():
        if fname == link.name:
            return sorted(float(x) for x in points)
    return []


# ---- main flow ------------------------------------------------------------

def process(spec: OriginalSpec, conn: sqlite3.Connection) -> None:
    print(f"\n[{spec.track_id}] {spec.artist} — {spec.title}")

    existing = _find_original_audio(conn, spec.track_id)
    if existing is None:
        query = f"{spec.artist} {spec.title}"
        resolved = _resolve_search(query)
        if resolved is None:
            print("  SKIP: yt search returned nothing")
            return
        yt_id, yt_url = resolved
        out_path = _ORIGINALS_DIR / f"{spec.track_id}__youtube__{yt_id}.m4a"
        if not out_path.exists():
            print(f"  downloading {yt_url} → {out_path.name}")
            ok = _download(query, out_path)
            if not ok:
                print("  SKIP: download failed")
                return
        ta_id = _insert_original_audio(
            conn, track_id=spec.track_id, path=str(out_path),
            yt_id=yt_id, yt_url=yt_url,
        )
        print(f"  inserted track_audio_id={ta_id}")
    else:
        ta_id, audio_path_str = existing
        print(f"  original already present: track_audio_id={ta_id}")

    if _has_canonical_cues(conn, spec.track_id):
        print("  canonical cues already present — skipping cue-detr")
        return

    ta_path = conn.execute(
        "SELECT path FROM track_audio WHERE track_audio_id=?", (ta_id,),
    ).fetchone()["path"]
    print(f"  running cue-detr (sens={CUE_DETR_SENSITIVITY}) on {Path(ta_path).name}")
    cues = _run_cue_detr(Path(ta_path), CUE_DETR_SENSITIVITY)
    print(f"  → {len(cues)} cues: {[round(c, 1) for c in cues[:8]]}{'…' if len(cues) > 8 else ''}")
    _write_canonical_cues(
        conn, track_id=spec.track_id, cues=cues,
        source_track_audio_id=ta_id, source_variant_tag="original",
        sensitivity=CUE_DETR_SENSITIVITY,
    )
    print("  stored canonical_track_cue_points")


def backfill_existing_originals(conn: sqlite3.Connection) -> None:
    """Populate canonical_track_cue_points for tracks that already have an
    'original' variant (e.g. Antoine on BB11) — reuse existing
    track_analysis.cue_points_json at its original sensitivity rather than
    re-running cue-detr."""
    rows = conn.execute(
        """
        SELECT ta.track_id, ta.track_audio_id, ta.variant_tag,
               ta_analysis.cue_points_json
        FROM track_audio ta
        JOIN track_analysis ta_analysis ON ta_analysis.track_audio_id = ta.track_audio_id
        WHERE ta.variant_tag = 'original'
          AND NOT EXISTS (
              SELECT 1 FROM canonical_track_cue_points cp WHERE cp.track_id = ta.track_id
          )
        """
    ).fetchall()
    for r in rows:
        cues = json.loads(r["cue_points_json"]) if r["cue_points_json"] else []
        _write_canonical_cues(
            conn, track_id=r["track_id"], cues=cues,
            source_track_audio_id=r["track_audio_id"],
            source_variant_tag="original", sensitivity=0.9,
        )
        print(f"  backfilled canonical cues for {r['track_id']} ({len(cues)} cues)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set-id", default="2nvzlh2k")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    if args.set_id != "2nvzlh2k":
        print("Only BB11 is pre-curated right now. Add an OriginalSpec "
              "table for other sets or automate variant detection.")
        return 1

    conn = _connect(args.db)
    try:
        print("Backfilling canonical cues from existing 'original' variants…")
        backfill_existing_originals(conn)
        print("\nProcessing BB11 tracks needing an original variant…")
        for spec in BB11_ORIGINALS:
            process(spec, conn)
    finally:
        conn.close()
    print("\ndone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
