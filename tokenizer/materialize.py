"""Stream `dj_set_rows.raw_html`, tokenize each row, and materialize the
parsed structure into three tables:

  - track_metadata      (1 row per track_id, aggregated across sets)
  - track_suggestions   (1 row per sug_id)
  - track_id_links      (linked-tracklist hints from IDTrack.linked_items)

Idempotent — re-running clears and rebuilds. Safe to run on the live DB
because we only DELETE FROM the three derived tables we own.

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

from tokenizer.tokenizer import classify_row
from tokenizer.track_tokenizer import parse_track_row as parse_track_main
from tokenizer.track_tokenizer import TrackRow
from tokenizer.suggestion_tokenizer import parse_suggestion_row
from tokenizer.id_tokenizer import parse_track_row as parse_id_lens

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
                "version_tag": None,
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
        if cur["version_tag"] is None and tr.version_tag:
            cur["version_tag"] = tr.version_tag
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


def _flush_links(conn: sqlite3.Connection, buf: list[tuple]) -> None:
    if not buf:
        return
    conn.executemany(
        "INSERT OR IGNORE INTO track_id_links ("
        "set_id, tlp_id, linker_user_name, linker_user_href, linker_user_followers, "
        "linked_tracklist_href, linked_tracklist_text"
        ") VALUES (?,?,?,?,?,?,?)",
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
    conn.executescript("""
        DELETE FROM track_metadata;
        DELETE FROM track_suggestions;
        DELETE FROM track_id_links;
    """)
    conn.commit()

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM dj_set_rows")
    total = cur.fetchone()[0]
    log.info("streaming %s rows from dj_set_rows", f"{total:,}")

    metadata = _MetadataAccumulator()
    sug_buf: list[tuple] = []
    link_buf: list[tuple] = []
    counts = {"track": 0, "suggestion": 0, "text": 0, "id_links": 0, "errors": 0}

    offset = 0
    last_log_pct = -10
    while True:
        cur.execute(
            "SELECT row_id, set_id, raw_html FROM dj_set_rows "
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
                kind = classify_row(raw)
                if kind == "track":
                    tr = parse_track_main(raw)
                    if tr.track_key:
                        metadata.update(tr, row["set_id"])
                        counts["track"] += 1
                    # second lens: id_tokenizer for linked-tracklist hints
                    outer = BeautifulSoup(raw, "html.parser").find("div", class_="tlpItem")
                    if outer is not None:
                        id_t = parse_id_lens(outer)
                        for link in id_t.linked_items:
                            link_buf.append((
                                row["set_id"], id_t.tlp_id,
                                link.user_name, link.user_href,
                                link.user_followers_text,
                                link.linked_tracklist_href,
                                link.linked_tracklist_text,
                            ))
                            counts["id_links"] += 1

                elif kind == "suggestion":
                    outer = BeautifulSoup(raw, "html.parser").find("div", class_="sugTog")
                    if outer is not None:
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

                elif kind == "text":
                    counts["text"] += 1

            except Exception as e:
                counts["errors"] += 1
                log.debug("row_id=%s error: %s", row["row_id"], e)

            if len(sug_buf) >= _BATCH_INSERT:
                _flush_suggestions(conn, sug_buf)
                sug_buf.clear()
            if len(link_buf) >= _BATCH_INSERT:
                _flush_links(conn, link_buf)
                link_buf.clear()

        offset += batch_size
        pct = int(100 * offset / max(total, 1))
        if pct - last_log_pct >= 5:
            log.info(
                "%d/%s (%d%%) — track=%d sug=%d links=%d errors=%d",
                offset, f"{total:,}", pct,
                counts["track"], counts["suggestion"], counts["id_links"], counts["errors"],
            )
            last_log_pct = pct

    # Final tail flushes
    _flush_suggestions(conn, sug_buf)
    _flush_links(conn, link_buf)

    # Upsert track_metadata (single big batch — ~50k rows expected)
    md_rows = metadata.finalize_rows()
    log.info("upserting %d unique track_metadata rows", len(md_rows))
    _flush_metadata(conn, md_rows)

    conn.close()

    result = {
        "track_metadata": len(md_rows),
        "track_suggestions": counts["suggestion"],
        "track_id_links": counts["id_links"],
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
