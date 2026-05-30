"""Reconcile disk ↔ DB drift: route orphan audio files into delete / register /
promote — never blindly.

An *orphan* is a file under {audio_root}/objects/** that no `track_audio.path`
references. They form when a download finalizes on disk but the row INSERT never
commits (the bug fixed forward by `core.db.insert_audio_or_reap`). This tool
cleans up the ones that already exist.

A blind reaper is WRONG: many orphans are the track's *only* audio, and some
coexist cases are the *real* full song while a short acappella holds the
registered reference row (the Daft Punk `9hp84x` case). So each orphan is sorted
into one of four dispositions:

  DELETE     — `.webm`/`.part` intermediates, and cross-folder byte-dup orphans
               (a single search hit wrongly matched to multiple tracks), but only
               when deletion cannot strand a track (every folder sharing the hash
               keeps registered audio).
  REGISTER   — pure-orphan folders (the tid has zero track_audio rows): the orphan
               is the only audio → insert a row pointing at it.
  PROMOTE    — coexist folder where the orphan is materially longer AND the
               registered reference is very short + acappella-shaped: register the
               orphan as the new reference, demote the old row to a non-reference
               acappella variant (kept, NOT cascade-deleted).
  REVIEW     — everything ambiguous: written to a TSV, nothing changed.

Default is DRY-RUN (prints a manifest, touches nothing). `--apply` performs the
DELETE/REGISTER actions and the safe deletes. PROMOTE is gated separately behind
`--apply-promotions` so the low-risk tiers can land first.

Run ON pi-storage (paths are local there), from the repo root:

    venvs/audio/bin/python scripts/reconcile_orphans.py            # dry-run
    venvs/audio/bin/python scripts/reconcile_orphans.py --apply    # + safe registers/deletes
    venvs/audio/bin/python scripts/reconcile_orphans.py --apply --apply-promotions

Reuses `core.db.insert_audio_or_reap` (Part A) for registration and
`ingest.corrections.log_correction` for the training-signal ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core import db as db_adapter
from core.models import AudioAsset
from core.result import Ok
from ingest.corrections import Correction, log_correction

_log = logging.getLogger("reconcile_orphans")

FINAL_EXTS = {".m4a", ".mp3", ".opus", ".flac", ".wav", ".ogg"}
INTERMEDIATE_EXTS = {".webm", ".part", ".ytdl", ".temp"}

KNOWN_PLATFORMS = {"youtube", "youtube_music", "soundcloud", "spotify"}


def _parse_canonical(filename: str) -> tuple[str, str] | None:
    """Split {tid}__{platform}__{player_id}.{ext} on the '__' delimiter (NOT a
    greedy regex) so player_ids that themselves contain '__' or a leading '_'
    (e.g. 't194__SJYL4', '_RHlmzvKMtA') parse correctly. Returns (platform,
    player_id) or None for a bare-name (spotdl 'Artist - Title.m4a') file."""
    stem = filename.rsplit(".", 1)[0]
    parts = stem.split("__")
    if len(parts) >= 3 and parts[1] in KNOWN_PLATFORMS:
        return parts[1], "__".join(parts[2:])
    return None

# PROMOTE heuristic: the registered reference must be this short (seconds) AND
# shorter than 0.6× the orphan, AND smell like an acappella, before we auto-flip.
PROMOTE_MAX_REF_S = 120.0
PROMOTE_RATIO = 0.6


# ───────────────────────── disk / probe helpers ─────────────────────────

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ffprobe(path: Path) -> tuple[float | None, str | None, int | None]:
    """Return (duration_s, codec, bitrate_kbps) via ffprobe, or Nones on failure."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration,bit_rate:stream=codec_name",
             "-select_streams", "a:0", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        if out.returncode != 0:
            return (None, None, None)
        data = json.loads(out.stdout or "{}")
        fmt = data.get("format", {})
        dur = float(fmt["duration"]) if fmt.get("duration") else None
        br = int(round(int(fmt["bit_rate"]) / 1000)) if fmt.get("bit_rate") else None
        streams = data.get("streams", [])
        codec = streams[0].get("codec_name") if streams else None
        return (dur, codec, br)
    except (subprocess.SubprocessError, ValueError, KeyError, json.JSONDecodeError):
        return (None, None, None)


# ───────────────────────── DB snapshot ─────────────────────────

@dataclass(frozen=True)
class RegRow:
    track_audio_id: int
    track_id: str
    path: str
    platform: str
    player_id: str | None
    duration_s: float | None
    stem: str
    is_reference: int


def _load_registered(db: Path) -> tuple[set[str], dict[str, list[RegRow]]]:
    """Return (set of registered paths, by_tid -> [RegRow])."""
    paths: set[str] = set()
    by_tid: dict[str, list[RegRow]] = defaultdict(list)
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        for r in conn.execute(
            "SELECT track_audio_id, track_id, path, platform, player_id, "
            "duration_s, stem, is_reference FROM track_audio"
        ):
            paths.add(r["path"])
            by_tid[r["track_id"]].append(RegRow(
                r["track_audio_id"], r["track_id"], r["path"], r["platform"],
                r["player_id"], r["duration_s"], r["stem"], r["is_reference"],
            ))
    return paths, by_tid


def _reference_row(rows: list[RegRow]) -> RegRow | None:
    """The selected reference: is_reference DESC, then highest taid (newest)."""
    if not rows:
        return None
    return sorted(rows, key=lambda r: (r.is_reference, r.track_audio_id), reverse=True)[0]


def _acappella_smell(db: Path, track_id: str, ref: RegRow) -> bool:
    if ref.stem and ref.stem.lower() == "acappella":
        return True
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        r = conn.execute(
            "SELECT full_name, version FROM track_metadata WHERE track_id=?",
            (track_id,),
        ).fetchone()
    if not r:
        return False
    blob = f"{r['full_name'] or ''} {r['version'] or ''}".lower()
    return "acappella" in blob or "acapella" in blob


# ───────────────────────── classification ─────────────────────────

@dataclass
class Orphan:
    path: Path
    track_id: str
    ext: str
    platform: str | None      # parsed from canonical name, else None (bare-name)
    player_id: str | None
    md5: str | None = None
    disposition: str = ""      # DELETE | REGISTER | PROMOTE | REVIEW
    reason: str = ""


def _scan_orphans(objects_root: Path, registered: set[str]) -> list[Orphan]:
    orphans: list[Orphan] = []
    for tid_dir in sorted(objects_root.iterdir()):
        if not tid_dir.is_dir():
            continue
        for f in sorted(tid_dir.iterdir()):
            if not f.is_file() or str(f) in registered:
                continue
            ext = f.suffix.lower()
            if ext not in FINAL_EXTS and ext not in INTERMEDIATE_EXTS:
                continue
            parsed = _parse_canonical(f.name)
            orphans.append(Orphan(
                path=f, track_id=tid_dir.name, ext=ext,
                platform=parsed[0] if parsed else None,
                player_id=parsed[1] if parsed else None,
            ))
    return orphans


def classify(db: Path, objects_root: Path) -> tuple[list[Orphan], set[str], dict[str, list[RegRow]]]:
    registered, by_tid = _load_registered(db)
    orphans = _scan_orphans(objects_root, registered)

    finals = [o for o in orphans if o.ext in FINAL_EXTS]
    inters = [o for o in orphans if o.ext in INTERMEDIATE_EXTS]

    # Intermediates: always trash.
    for o in inters:
        o.disposition, o.reason = "DELETE", "intermediate"

    # Hash final orphans + the registered reference file in each coexist folder,
    # so cross-folder byte-dups (wrong-match search hits) are caught even when
    # track_audio.sha256 is NULL.
    for o in finals:
        o.md5 = _sha256(o.path)
    md5_folders: dict[str, set[str]] = defaultdict(set)
    for o in finals:
        md5_folders[o.md5].add(o.track_id)
    coexist_tids = {o.track_id for o in finals if by_tid.get(o.track_id)}
    reg_md5: dict[str, set[str]] = defaultdict(set)
    for tid in coexist_tids:
        ref = _reference_row(by_tid[tid])
        if ref and Path(ref.path).is_file():
            reg_md5[_sha256(Path(ref.path))].add(tid)

    for o in finals:
        has_reg = bool(by_tid.get(o.track_id))
        # Wrong-match dup: this md5 appears in >1 distinct folder among orphans,
        # OR matches a registered file in a *different* folder.
        dup_folders = set(md5_folders[o.md5]) | {
            t for t in reg_md5.get(o.md5, set()) if t != o.track_id
        }
        is_cross_dup = len(dup_folders) > 1
        if is_cross_dup:
            # Safe to delete only if every involved folder keeps registered audio
            # (deletion can't strand a track). Else send to REVIEW.
            safe = all(bool(by_tid.get(t)) for t in md5_folders[o.md5])
            if safe:
                o.disposition, o.reason = "DELETE", f"wrong-match dup (folders={sorted(dup_folders)})"
            else:
                o.disposition, o.reason = "REVIEW", f"cross-folder dup, a folder is pure-orphan (folders={sorted(dup_folders)})"
            continue

        if not has_reg:
            o.disposition, o.reason = "REGISTER", "pure orphan (tid has no track_audio)"
            continue

        # COEXIST → compare against the registered reference.
        ref = _reference_row(by_tid[o.track_id])
        odur, _, _ = _ffprobe(o.path)
        rdur = ref.duration_s if ref else None
        if odur and rdur and odur > rdur and rdur < PROMOTE_MAX_REF_S \
                and rdur < PROMOTE_RATIO * odur and _acappella_smell(db, o.track_id, ref):
            o.disposition = "PROMOTE"
            o.reason = (f"orphan {odur:.0f}s > ref taid{ref.track_audio_id} {rdur:.0f}s "
                        f"(acappella-shaped) -> promote, demote ref")
        else:
            o.disposition = "REVIEW"
            o.reason = (f"coexist: orphan={odur and round(odur)}s "
                        f"ref taid{ref and ref.track_audio_id}={rdur and round(rdur)}s "
                        f"acap={_acappella_smell(db, o.track_id, ref) if ref else '?'}")
    return orphans, registered, by_tid


# ───────────────────────── actions (apply) ─────────────────────────

def _canonicalize_bare(o: Orphan) -> Path:
    """Rename a bare-name spotdl orphan to the canonical convention; return new path."""
    dst = o.path.with_name(f"{o.track_id}__spotify__reconciled{o.ext}")
    o.path.rename(dst)
    return dst


def _do_register(db: Path, o: Orphan, *, as_reference: bool) -> int | None:
    path = o.path
    platform, player_id = o.platform, o.player_id
    if platform is None:  # bare-name spotdl orphan
        path = _canonicalize_bare(o)
        platform, player_id = "spotify", None
    dur, codec, br = _ffprobe(path)
    asset = AudioAsset(
        track_audio_id=None, track_id=o.track_id, platform=platform,
        source_url=f"reconcile://orphan/{path.name}", player_id=player_id or "",
        path=str(path), sha256=_sha256(path), duration_s=dur, sample_rate=None,
        codec=codec or o.ext.lstrip("."), bitrate_kbps=br,
    )
    r = db_adapter.insert_audio_or_reap(db, asset)
    if not isinstance(r, Ok):
        _log.error("  register FAILED %s: %s", o.track_id, r)
        return None
    taid = r.value
    if as_reference:
        with sqlite3.connect(db) as conn:
            conn.execute("UPDATE track_audio SET is_reference=1 WHERE track_audio_id=?", (taid,))
            conn.commit()
    log_correction(db, Correction(
        track_id=o.track_id, axis="version", action="add",
        new_track_audio_id=taid, new_platform=platform, new_player_id=player_id,
        new_url=asset.source_url, stem_value="regular",
        reason=o.reason, source="reconcile_orphans",
    ))
    return taid


def _do_promote(db: Path, o: Orphan, by_tid: dict[str, list[RegRow]]) -> int | None:
    ref = _reference_row(by_tid[o.track_id])
    if ref is None:
        return None
    new_taid = _do_register(db, o, as_reference=True)
    if new_taid is None:
        return None
    # Demote the old reference IN PLACE — keep its stems/analysis (no cascade).
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE track_audio SET is_reference=0, stem='acappella' "
            "WHERE track_audio_id=?", (ref.track_audio_id,))
        conn.commit()
    log_correction(db, Correction(
        track_id=o.track_id, axis="stem", action="replace",
        old_track_audio_id=ref.track_audio_id, old_platform=ref.platform,
        old_player_id=ref.player_id,
        new_track_audio_id=new_taid, new_platform=o.platform,
        new_player_id=o.player_id, stem_value="acappella",
        reason=o.reason, source="reconcile_orphans",
    ))
    return new_taid


# ───────────────────────── manifest / main ─────────────────────────

def _print_manifest(orphans: list[Orphan]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    by_disp: dict[str, list[Orphan]] = defaultdict(list)
    for o in orphans:
        counts[o.disposition] += 1
        by_disp[o.disposition].append(o)
    print("\n================== RECONCILE MANIFEST ==================")
    for disp in ("DELETE", "REGISTER", "PROMOTE", "REVIEW"):
        items = by_disp.get(disp, [])
        print(f"\n{disp}: {len(items)}")
        for o in items[:200]:
            print(f"  [{o.track_id}] {o.path.name}  -- {o.reason}")
    print(f"\nTOTAL orphans: {len(orphans)}  "
          f"({dict((k, counts[k]) for k in counts)})")
    return dict(counts)


def _run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    objects_root = Path(args.audio_root) / "objects"
    if not objects_root.is_dir():
        _log.error("objects root not found: %s", objects_root)
        return 2

    orphans, _registered, by_tid = classify(args.db, objects_root)
    counts = _print_manifest(orphans)

    # REVIEW TSV always written (it's the human worklist).
    review = [o for o in orphans if o.disposition == "REVIEW"]
    if review:
        with open(args.review_tsv, "w") as fh:
            fh.write("track_id\tpath\treason\n")
            for o in review:
                fh.write(f"{o.track_id}\t{o.path}\t{o.reason}\n")
        print(f"\nREVIEW worklist -> {args.review_tsv}")

    if not args.apply:
        print("\n(dry-run - nothing changed. Re-run with --apply to act.)")
        return 0

    # APPLY: deletes + registers (promotions gated separately).
    n_del = n_reg = n_prom = 0
    for o in orphans:
        if o.disposition == "DELETE":
            o.path.unlink(missing_ok=True)
            n_del += 1
        elif o.disposition == "REGISTER":
            if _do_register(args.db, o, as_reference=False) is not None:
                n_reg += 1
        elif o.disposition == "PROMOTE" and args.apply_promotions:
            if _do_promote(args.db, o, by_tid) is not None:
                n_prom += 1
    print(f"\nAPPLIED: deleted={n_del}  registered={n_reg}  promoted={n_prom}"
          f"{'' if args.apply_promotions else '  (promotions skipped - pass --apply-promotions)'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path,
                   default=Path(os.environ.get("TRACKLIST_DB",
                                               "/mnt/storage/data/db/music_database.db")))
    p.add_argument("--audio-root", type=Path,
                   default=Path(os.environ.get("TRACKLIST_AUDIO_ROOT", "/mnt/storage")))
    p.add_argument("--apply", action="store_true",
                   help="perform DELETE + REGISTER actions (default: dry-run)")
    p.add_argument("--apply-promotions", action="store_true",
                   help="also perform PROMOTE actions (requires --apply)")
    p.add_argument("--review-tsv", type=Path, default=Path("reconcile_review.tsv"))
    return _run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
