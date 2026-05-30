"""Stream `dj_set_rows.raw_html`, tokenize each row, and materialize the
parsed structure into four tables:

  - track_metadata      (1 row per track_id, aggregated across sets)
  - set_track_slots     (1 row per played slot — aligner / claim view)
  - track_suggestions   (1 row per sug_id)
  - track_id_links      (linked-tracklist hints from IDTrack.linked_items)

Idempotent — re-running clears and rebuilds. Safe to run on the live DB
because we only DELETE FROM the derived tables we own.

Run on pi-storage (the DB is local — no RPC needed):

    venvs/web_crawler/bin/python -m tokenizer.materialize \\
        --db /mnt/storage/data/db/music_database.db
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any

# adjust path so 'tokenizer' resolves whether invoked as -m or as a script
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bs4 import BeautifulSoup

from tokenizer._parser import BS_PARSER
from tokenizer.tokenizer import classify_row
from tokenizer.identity_axes import (
    derive_claimed_variant,
    scrape_claimed_stem,
    scrape_claimed_version,
)
from tokenizer.track_tokenizer import parse_track_row as parse_track_main
from tokenizer.track_tokenizer import TrackRow
from tokenizer.suggestion_tokenizer import parse_suggestion_row

log = logging.getLogger("tokenizer.materialize")

_BATCH_INSERT = 1000


# -----------------------------------------------------------------------------
# track_metadata aggregator (in-memory, ~50k unique track_ids fits easily)
# -----------------------------------------------------------------------------

class _MetadataAccumulator:
    """Holds per-track_id merged 'best so far' view across all sets seen."""

    def __init__(self) -> None:
        self._d: dict[str, dict[str, Any]] = {}

    def update(self, tr: TrackRow, set_id: str) -> None:
        if not tr.track_key or not tr.is_ided:
            return
        cur = self._d.get(tr.track_key)
        if cur is None:
            cur = {
                "track_id": tr.track_key,
                "title": None,
                "artists_json": None,
                "full_name": None,
                "genre": None,
                "duration_seconds": None,
                "is_remixish": 0,
                "version": None,
                "has_youtube": 0,
                "has_soundcloud": 0,
                "has_spotify": 0,
                "has_apple": 0,
                "plays_total": 0,
                "set_count": 0,
                "_seen_sets": set(),
                "artwork_url": None,
            }

        # First-non-null winner for stable text fields
        if cur["title"] is None and tr.title:
            cur["title"] = tr.title
        if cur["artists_json"] is None and tr.artists:
            cur["artists_json"] = json.dumps(list(tr.artists), ensure_ascii=False)
        if cur["full_name"] is None and tr.full_name:
            cur["full_name"] = tr.full_name
        if cur["genre"] is None and tr.genre:
            cur["genre"] = tr.genre
        if cur["duration_seconds"] is None and tr.duration_seconds:
            cur["duration_seconds"] = tr.duration_seconds
        if cur["version"] is None and tr.version_tag:
            cur["version"] = scrape_claimed_version(tr.version_tag)
        if cur["artwork_url"] is None and tr.artwork_url:
            cur["artwork_url"] = tr.artwork_url

        # OR of has_* flags
        if tr.media_flags.youtube:
            cur["has_youtube"] = 1
        if tr.media_flags.soundcloud:
            cur["has_soundcloud"] = 1
        if tr.media_flags.spotify:
            cur["has_spotify"] = 1
        if tr.media_flags.apple:
            cur["has_apple"] = 1
        if tr.is_remixish:
            cur["is_remixish"] = 1

        # Max plays
        if tr.plays and tr.plays > (cur["plays_total"] or 0):
            cur["plays_total"] = tr.plays

        # Distinct sets this track appears in
        cur["_seen_sets"].add(set_id)
        cur["set_count"] = len(cur["_seen_sets"])

        self._d[tr.track_key] = cur

    def finalize_rows(self) -> list[dict[str, Any]]:
        return [{k: v for k, v in d.items() if k != "_seen_sets"} for d in self._d.values()]


# -----------------------------------------------------------------------------
# Flushers
# -----------------------------------------------------------------------------

def _flush_suggestions(conn: sqlite3.Connection, buf: list[tuple]) -> None:
    if not buf:
        return
    conn.executemany(
        "INSERT OR REPLACE INTO track_suggestions ("
        "sug_id, set_id, tlp_id, pos, track_slug, track_display, artist_title, "
        "suggester_user_id, suggester_name, suggestion_timestamp, "
        "is_remix, has_youtube, has_soundcloud, has_spotify"
        ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        buf,
    )
    conn.commit()


def _flush_metadata(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    placeholders = ",".join("?" * len(keys))
    sql = f"INSERT OR REPLACE INTO track_metadata ({','.join(keys)}) VALUES ({placeholders})"
    conn.executemany(sql, [tuple(r[k] for k in keys) for r in rows])
    conn.commit()


def _slot_label(tr: TrackRow) -> str | None:
    """Mirror pull_set_for_alignment's published section / w/ label."""
    raw = tr.track_number_raw
    if raw == "w/":
        return None  # layered rows get labels at flush time from primary
    if raw and raw.strip().isdigit():
        return raw.strip().zfill(3)
    return None


def _flush_slots(conn: sqlite3.Connection, buf: list[tuple]) -> None:
    if not buf:
        return
    conn.executemany(
        "INSERT OR REPLACE INTO set_track_slots ("
        "set_id, row_index, tlp_id, recording_id, track_id, source, slot_label, "
        "is_concurrent, cue_seconds, cue_time_seconds, claimed_version, "
        "claimed_stem, claimed_variant, full_name, title, artists_json, duration_seconds"
        ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        buf,
    )
    conn.commit()


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def materialize(db_path: Path, batch_size: int = 10_000) -> dict[str, int]:
    log.info("opening %s", db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Ensure the 3 destination tables exist (schema.sql runs IF NOT EXISTS)
    schema_path = _REPO_ROOT / "web_crawler" / "database" / "schema.sql"
    log.info("applying schema (idempotent IF NOT EXISTS)")
    conn.executescript(schema_path.read_text(encoding="utf-8"))
    conn.commit()

    log.info("clearing destination tables for clean rebuild")
    # Note: track_id_links is populated by a separate focused pass
    # (`tokenizer.id_links` — to be written), so we leave its rows intact here.
    conn.executescript("""
        DELETE FROM track_metadata;
        DELETE FROM set_track_slots;
        DELETE FROM track_suggestions;
    """)
    conn.commit()

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM dj_set_rows")
    total = cur.fetchone()[0]
    log.info("streaming %s rows from dj_set_rows", f"{total:,}")

    metadata = _MetadataAccumulator()
    slot_buf: list[tuple] = []
    sug_buf: list[tuple] = []
    counts = {"track": 0, "slot": 0, "suggestion": 0, "text": 0, "errors": 0}
    # Per-set w/ layering: primary label + w/ counter (matches pull_set_for_alignment).
    slot_state: dict[str, tuple[str | None, int]] = {}

    offset = 0
    last_log_pct = -10
    while True:
        cur.execute(
            "SELECT row_id, set_id, row_index, raw_html FROM dj_set_rows "
            "ORDER BY row_id LIMIT ? OFFSET ?",
            (batch_size, offset),
        )
        batch = cur.fetchall()
        if not batch:
            break

        for row in batch:
            raw = row["raw_html"] or ""
            if not raw:
                continue
            try:
                # Single BS parse per row, dispatch by outer-div class set.
                # This replaces the old (classify_row + parser) double-parse
                # and skips the id_tokenizer second-lens entirely (track_id_links
                # gets populated by a focused later pass).
                soup = BeautifulSoup(raw, BS_PARSER)
                outer = soup.find("div")
                if outer is None:
                    continue
                outer_classes = set(outer.get("class") or [])

                if "tlpItem" in outer_classes:
                    tr = parse_track_main(raw)  # internally parses once with BS_PARSER too
                    if tr.track_key:
                        if tr.is_ided:
                            metadata.update(tr, row["set_id"])
                            counts["track"] += 1

                        sid = row["set_id"]
                        primary, w_ctr = slot_state.get(sid, (None, 0))
                        base = _slot_label(tr)
                        if base is not None:
                            primary, w_ctr = base, 0
                            label = base
                        elif tr.is_concurrent and primary is not None:
                            w_ctr += 1
                            label = f"{primary}w{w_ctr}"
                        else:
                            label = None
                        slot_state[sid] = (primary, w_ctr)

                        source = "synthetic" if tr.track_key.startswith("tlp") else "scraped"

                        claimed_stem = scrape_claimed_stem(tr.full_name)
                        slot_buf.append((
                            sid,
                            int(row["row_index"]),
                            tr.data_id,
                            tr.track_key,
                            tr.track_key,
                            source,
                            label,
                            int(tr.is_concurrent),
                            tr.cue_seconds,
                            tr.cue_time_seconds,
                            scrape_claimed_version(tr.version_tag),
                            claimed_stem,
                            derive_claimed_variant(tr.full_name),
                            tr.full_name,
                            tr.title,
                            json.dumps(list(tr.artists), ensure_ascii=False)
                            if tr.artists else None,
                            tr.duration_seconds,
                        ))
                        counts["slot"] += 1

                        if len(slot_buf) >= _BATCH_INSERT:
                            _flush_slots(conn, slot_buf)
                            slot_buf.clear()

                elif "sugTog" in outer_classes:
                    sug = parse_suggestion_row(outer)
                    sug_buf.append((
                        sug.sug_id, row["set_id"], sug.tlp_id, sug.pos,
                        sug.track_slug, sug.track_display, sug.artist_title,
                        sug.suggester_user_id, sug.suggester_name,
                        sug.suggestion_timestamp,
                        int(bool(sug.is_remix)) if sug.is_remix is not None else None,
                        int(bool(sug.has_youtube)),
                        int(bool(sug.has_soundcloud)),
                        int(bool(sug.has_spotify)),
                    ))
                    counts["suggestion"] += 1

                elif "bItmH" in outer_classes:
                    counts["text"] += 1

            except Exception as e:
                counts["errors"] += 1
                log.debug("row_id=%s error: %s", row["row_id"], e)

            if len(sug_buf) >= _BATCH_INSERT:
                _flush_suggestions(conn, sug_buf)
                sug_buf.clear()

        offset += batch_size
        pct = int(100 * offset / max(total, 1))
        if pct - last_log_pct >= 5:
            log.info(
                "%d/%s (%d%%) — track=%d slot=%d sug=%d errors=%d",
                offset, f"{total:,}", pct,
                counts["track"], counts["slot"], counts["suggestion"], counts["errors"],
            )
            last_log_pct = pct

    # Final tail flushes
    _flush_slots(conn, slot_buf)
    _flush_suggestions(conn, sug_buf)

    # Upsert track_metadata (single big batch — ~50k rows expected)
    md_rows = metadata.finalize_rows()
    log.info("upserting %d unique track_metadata rows", len(md_rows))
    _flush_metadata(conn, md_rows)

    conn.close()

    result = {
        "track_metadata": len(md_rows),
        "set_track_slots": counts["slot"],
        "track_suggestions": counts["suggestion"],
        "errors": counts["errors"],
    }
    log.info("DONE — %s", result)
    return result


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--db", required=True, type=Path,
                   help="Path to music_database.db")
    p.add_argument("--batch-size", type=int, default=10_000,
                   help="Rows fetched per cycle (default: 10000)")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.db.exists():
        print(f"DB not found: {args.db}", file=sys.stderr)
        return 2

    materialize(args.db, args.batch_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
