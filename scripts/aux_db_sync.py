"""Ingest scattered popularity-research caches into data/analysis/aux.db.

Idempotent — safe to re-run any time the underlying caches grow. Reads:

  data/analysis/spotify_release_dates.csv  -> track_meta.release_year (per spotify_id)
  data/analysis/bb_track_meta.csv          -> track_meta.{artist,title,spotify_id}
  data/analysis/lastfm_track_info.json     -> track_lastfm.*
  data/analysis/billboard_yearend.json     -> chart_yearend ('billboard_hot100')
  data/analysis/billboard_weekly_current.csv -> chart_song_history
                                              ('billboard_hot100_weekly': all-time
                                              weekly Hot 100 1958–present, one row
                                              per unique song with peak_position and
                                              weeks_on_chart aggregated)

After the popularity run finishes, also rebuilds track_chart_match by re-running
the fuzzy artist+title matcher against both the year-end chart and the weekly
song-history (cheap, all-in-memory).

The aux DB is research-only, gitignored, rebuildable. Don't promote anything
to the main DB until the signal is proven useful.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
import sys
import urllib.request
from pathlib import Path

WEEKLY_CSV_URL = (
    "https://raw.githubusercontent.com/utdata/rwd-billboard-data/"
    "main/data-out/hot-100-current.csv"
)

CACHE_DIR = Path("data/analysis")
AUX_DB = CACHE_DIR / "aux.db"
SPOTIFY_CSV = CACHE_DIR / "spotify_release_dates.csv"
BB_META_CSV = CACHE_DIR / "bb_track_meta.csv"
LASTFM_JSON = CACHE_DIR / "lastfm_track_info.json"
BILLBOARD_JSON = CACHE_DIR / "billboard_yearend.json"
BILLBOARD_WEEKLY_CSV = CACHE_DIR / "billboard_weekly_current.csv"

SCHEMA = """
-- track_chart_match is fully rebuilt on every run, so drop+recreate is safe and
-- lets us evolve its schema (e.g. relaxing rank from NOT NULL to nullable so
-- weekly-chart matches can coexist with year-end matches under the same PK).
DROP TABLE IF EXISTS track_chart_match;

CREATE TABLE IF NOT EXISTS track_meta (
  track_id            TEXT PRIMARY KEY,
  artist              TEXT,
  title               TEXT,
  spotify_id          TEXT,
  release_year        INTEGER,
  release_year_source TEXT,
  fetched_at          DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_track_meta_spotify ON track_meta(spotify_id);

CREATE TABLE IF NOT EXISTS track_lastfm (
  track_id    TEXT PRIMARY KEY,
  lfm_artist  TEXT,
  lfm_title   TEXT,
  mbid        TEXT,
  url         TEXT,
  listeners   INTEGER,
  playcount   INTEGER,
  error_code  INTEGER,
  fetched_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (track_id) REFERENCES track_meta(track_id)
);

CREATE TABLE IF NOT EXISTS chart_yearend (
  chart_name TEXT    NOT NULL,
  year       INTEGER NOT NULL,
  rank       INTEGER NOT NULL,
  title      TEXT,
  artist     TEXT,
  PRIMARY KEY (chart_name, year, rank)
);
CREATE INDEX IF NOT EXISTS idx_chart_yearend_year ON chart_yearend(chart_name, year);

CREATE TABLE IF NOT EXISTS chart_song_history (
  chart_name      TEXT    NOT NULL,
  title           TEXT    NOT NULL,
  performer       TEXT    NOT NULL,
  peak_position   INTEGER NOT NULL,
  weeks_on_chart  INTEGER NOT NULL,
  debut_date      TEXT    NOT NULL,
  debut_year      INTEGER NOT NULL,
  last_chart_date TEXT    NOT NULL,
  PRIMARY KEY (chart_name, title, performer)
);
CREATE INDEX IF NOT EXISTS idx_chart_history_debut_year
  ON chart_song_history(chart_name, debut_year);

CREATE TABLE IF NOT EXISTS track_chart_match (
  track_id        TEXT    NOT NULL,
  chart_name      TEXT    NOT NULL,
  chart_year      INTEGER NOT NULL,
  rank            INTEGER,         -- year-end ranking (1..100); NULL for weekly matches
  peak_position   INTEGER,         -- all-time weekly peak (1..100); NULL for year-end matches
  weeks_on_chart  INTEGER,         -- total weeks on chart;        NULL for year-end matches
  matched_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (track_id, chart_name),
  FOREIGN KEY (track_id) REFERENCES track_meta(track_id)
);

CREATE TABLE IF NOT EXISTS analysis_results (
  analysis_name TEXT NOT NULL,
  metric        TEXT NOT NULL,
  group_key     TEXT NOT NULL,
  value         REAL,
  unit          TEXT,
  notes         TEXT,
  computed_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (analysis_name, metric, group_key)
);
"""

RESULT_JSONS = [
    (CACHE_DIR / "bb_era_orthogonality.json", "bb_era_orthogonality_v1"),
    (CACHE_DIR / "bb_popularity.json", "bb_popularity_v1"),
]


def ingest_result_json(cur: sqlite3.Cursor, path: Path, analysis_name: str) -> int:
    """Flatten a results JSON into (metric, group_key, value) rows. Skips
    non-scalar fields (examples lists, etc.) — those live in the JSON file."""
    if not path.exists():
        return 0
    d = json.loads(path.read_text())
    count = 0

    def emit(metric: str, group_key: str, value, unit: str | None = None,
             notes: str | None = None) -> None:
        nonlocal count
        if value is None:
            return
        if isinstance(value, (int, float)):
            cur.execute("""
                INSERT INTO analysis_results
                  (analysis_name, metric, group_key, value, unit, notes)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(analysis_name, metric, group_key) DO UPDATE SET
                  value = excluded.value, unit = excluded.unit,
                  notes = excluded.notes, computed_at = CURRENT_TIMESTAMP
            """, (analysis_name, metric, group_key, float(value), unit, notes))
            count += 1

    def walk(prefix: str, group: str, obj) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    walk(prefix, group + "/" + k if group else k, v)
                else:
                    emit(prefix + ("/" + group if group else ""), k, v)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(prefix, group + f"[{i}]", v)

    # Walk top-level keys, treating dicts as group dimensions
    for k, v in d.items():
        if isinstance(v, (int, float)):
            emit(k, "all", v)
        elif isinstance(v, dict):
            for gk, gv in v.items():
                if isinstance(gv, (int, float)):
                    emit(k, gk, gv)
                elif isinstance(gv, dict):
                    for inner_k, inner_v in gv.items():
                        if isinstance(inner_v, (int, float)):
                            emit(f"{k}/{inner_k}", gk, inner_v)
        elif isinstance(v, list):
            # skip list-valued fields (range tuples, examples)
            pass
    return count

PAREN_RE = re.compile(r"\([^)]*\)")
FEAT_RE = re.compile(r"\b(feat|ft|featuring|w/|with)\.?\b.*$", re.I)
NON_ALNUM = re.compile(r"[^a-z0-9]+")


def norm(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = PAREN_RE.sub(" ", s)
    s = FEAT_RE.sub(" ", s)
    s = NON_ALNUM.sub(" ", s).strip()
    return " ".join(s.split())


def main() -> int:
    if not CACHE_DIR.exists():
        print(f"no cache dir at {CACHE_DIR}", file=sys.stderr)
        return 1
    conn = sqlite3.connect(AUX_DB)
    conn.executescript(SCHEMA)
    cur = conn.cursor()

    # --- spotify release-year cache ---
    sp_year: dict[str, int | None] = {}
    if SPOTIFY_CSV.exists():
        for sid, y in csv.reader(open(SPOTIFY_CSV)):
            sp_year[sid] = int(y) if y else None

    # --- BB track meta (track_id <-> artist/title/spotify_id) ---
    bb_meta: dict[str, dict] = {}
    if BB_META_CSV.exists():
        for r in csv.DictReader(open(BB_META_CSV)):
            tid = r.get("track_id")
            if not tid:
                continue
            existing = bb_meta.get(tid, {})
            existing.update({
                "artist": r.get("artist") or existing.get("artist"),
                "title": r.get("title") or existing.get("title"),
                "spotify_id": r.get("sid") or existing.get("spotify_id"),
            })
            bb_meta[tid] = existing

    print(f"track_meta: {len(bb_meta)} BB tracks, {len(sp_year)} spotify_id→year entries")
    for tid, d in bb_meta.items():
        sid = d.get("spotify_id") or None
        y = sp_year.get(sid) if sid else None
        cur.execute("""
            INSERT INTO track_meta (track_id, artist, title, spotify_id,
                                    release_year, release_year_source)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(track_id) DO UPDATE SET
              artist = excluded.artist,
              title = excluded.title,
              spotify_id = excluded.spotify_id,
              release_year = excluded.release_year,
              release_year_source = excluded.release_year_source
        """, (tid, d.get("artist"), d.get("title"), sid, y,
              "spotify_embed" if y else None))

    # --- Last.fm cache (keyed by "artist|||title") -> need to resolve to track_ids ---
    if LASTFM_JSON.exists():
        lfm = json.loads(LASTFM_JSON.read_text())
        # build (artist,title) -> [track_id] index
        pair_to_tids: dict[tuple[str, str], list[str]] = {}
        for tid, d in bb_meta.items():
            a, t = d.get("artist"), d.get("title")
            if a and t:
                pair_to_tids.setdefault((a, t), []).append(tid)

        loaded = 0
        for key, info in lfm.items():
            if "|||" not in key:
                continue
            a, t = key.split("|||", 1)
            tids = pair_to_tids.get((a, t), [])
            err = info.get("error")
            for tid in tids:
                cur.execute("""
                    INSERT INTO track_lastfm
                      (track_id, lfm_artist, lfm_title, mbid, url,
                       listeners, playcount, error_code)
                    VALUES (?,?,?,?,?,?,?,?)
                    ON CONFLICT(track_id) DO UPDATE SET
                      lfm_artist = excluded.lfm_artist,
                      lfm_title  = excluded.lfm_title,
                      mbid       = excluded.mbid,
                      url        = excluded.url,
                      listeners  = excluded.listeners,
                      playcount  = excluded.playcount,
                      error_code = excluded.error_code,
                      fetched_at = CURRENT_TIMESTAMP
                """, (tid, info.get("artist"), info.get("name"),
                      info.get("mbid"), info.get("url"),
                      info.get("listeners"), info.get("playcount"), err))
                loaded += 1
        print(f"track_lastfm: {loaded} rows loaded")
    else:
        print("track_lastfm: cache not yet present, skipping")

    # --- Billboard year-end ---
    if BILLBOARD_JSON.exists():
        bb_chart = json.loads(BILLBOARD_JSON.read_text())
        cur.execute("DELETE FROM chart_yearend WHERE chart_name = 'billboard_hot100'")
        total = 0
        for year, entries in bb_chart.items():
            for rank, title, artist in entries:
                cur.execute("""
                    INSERT OR REPLACE INTO chart_yearend
                      (chart_name, year, rank, title, artist)
                    VALUES ('billboard_hot100', ?, ?, ?, ?)
                """, (int(year), int(rank), title, artist))
                total += 1
        print(f"chart_yearend: {total} billboard_hot100 entries across {len(bb_chart)} years")
    else:
        print("chart_yearend: billboard cache not yet present, skipping")

    # --- Billboard weekly Hot 100 (all-time per-song aggregates) ---
    # Source has one row per (chart_week, current_position); collapse to one row
    # per (title, performer) keeping the best peak, max weeks-on-chart, and chart
    # debut date. The full archive is ~700k rows but only ~32k unique songs.
    if not BILLBOARD_WEEKLY_CSV.exists():
        print(f"fetching weekly Hot 100 CSV → {BILLBOARD_WEEKLY_CSV.name} …")
        BILLBOARD_WEEKLY_CSV.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(WEEKLY_CSV_URL, timeout=60) as resp, \
             open(BILLBOARD_WEEKLY_CSV, "wb") as out_f:
            out_f.write(resp.read())
    if BILLBOARD_WEEKLY_CSV.exists():
        cur.execute("DELETE FROM chart_song_history WHERE chart_name = 'billboard_hot100_weekly'")
        agg: dict[tuple[str, str], dict] = {}
        with open(BILLBOARD_WEEKLY_CSV) as f:
            for row in csv.DictReader(f):
                # The "current" CSV uses chart_week/peak_pos/wks_on_chart;
                # archives use chart_date/peak_position/weeks_on_chart. Support both.
                d = row.get("chart_week") or row.get("chart_date")
                peak = row.get("peak_pos") or row.get("peak_position")
                woc = row.get("wks_on_chart") or row.get("weeks_on_chart")
                if not (d and peak and woc):
                    continue
                title = (row["title"] or "").strip()
                performer = (row["performer"] or "").strip()
                if not (title and performer):
                    continue
                try:
                    peak_i = int(peak)
                    woc_i = int(woc)
                except ValueError:
                    continue
                key = (title, performer)
                s = agg.get(key)
                if s is None:
                    agg[key] = {"peak": peak_i, "woc": woc_i, "debut": d, "last": d}
                else:
                    if peak_i < s["peak"]: s["peak"] = peak_i
                    if woc_i > s["woc"]: s["woc"] = woc_i
                    if d < s["debut"]: s["debut"] = d
                    if d > s["last"]: s["last"] = d
        for (title, performer), s in agg.items():
            cur.execute("""
                INSERT INTO chart_song_history
                  (chart_name, title, performer, peak_position, weeks_on_chart,
                   debut_date, debut_year, last_chart_date)
                VALUES ('billboard_hot100_weekly', ?, ?, ?, ?, ?, ?, ?)
            """, (title, performer, s["peak"], s["woc"],
                  s["debut"], int(s["debut"][:4]), s["last"]))
        print(f"chart_song_history: {len(agg)} unique billboard_hot100_weekly songs "
              f"(1958-{max(s['last'][:4] for s in agg.values())})")
    else:
        print("chart_song_history: billboard weekly CSV not yet present, skipping")

    # --- Rebuild track_chart_match: year-end pass ---
    chart_by_year: dict[int, dict[str, tuple[int, str]]] = {}
    for year, rank, title, artist in cur.execute(
        "SELECT year, rank, title, artist FROM chart_yearend WHERE chart_name='billboard_hot100'"
    ):
        chart_by_year.setdefault(year, {})[norm(title)] = (rank, artist)

    def artist_match(track_artist_norm: str, chart_artist_raw: str) -> bool:
        """Accept if normalized track artist contains the chart artist's first
        token, or vice-versa. Mirrors the year-end matcher's leniency for
        ft./feat./vs. credit differences."""
        ca = norm(chart_artist_raw)
        if not track_artist_norm or not ca:
            return False
        first_token = ca.split()[0] if ca.split() else ""
        return ca in track_artist_norm or (first_token and first_token in track_artist_norm)

    matched_ye = 0
    for tid, artist, title, ry in cur.execute(
        "SELECT track_id, artist, title, release_year FROM track_meta WHERE release_year IS NOT NULL"
    ).fetchall():
        if not (artist and title and ry):
            continue
        nt = norm(title)
        na = norm(artist)
        if not nt:
            continue
        for dy in (0, 1, -1):
            y = ry + dy
            hit = chart_by_year.get(y, {}).get(nt)
            if not hit:
                continue
            rank, chart_artist = hit
            if na and not artist_match(na, chart_artist):
                continue
            cur.execute("""
                INSERT INTO track_chart_match
                  (track_id, chart_name, chart_year, rank)
                VALUES (?, 'billboard_hot100', ?, ?)
                ON CONFLICT(track_id, chart_name) DO UPDATE SET
                  chart_year = excluded.chart_year,
                  rank = excluded.rank,
                  matched_at = CURRENT_TIMESTAMP
            """, (tid, y, rank))
            matched_ye += 1
            break
    print(f"track_chart_match: {matched_ye} BB tracks matched to billboard_hot100 year-end")

    # --- Rebuild track_chart_match: weekly all-time pass ---
    # Bucket the song history by debut_year for fast lookup. Match window of
    # ±2 years on release_year tolerates release-date / chart-debut drift
    # (radio service adds, late chart entries, slight Spotify-vs-original
    # release year mismatches).
    history_by_year: dict[int, dict[str, tuple[int, int, str]]] = {}
    for title, performer, peak, woc, debut_year in cur.execute(
        """SELECT title, performer, peak_position, weeks_on_chart, debut_year
           FROM chart_song_history
           WHERE chart_name='billboard_hot100_weekly'"""
    ):
        # title-normalized key; value carries (peak, weeks_on_chart, chart_performer)
        history_by_year.setdefault(debut_year, {})[norm(title)] = (peak, woc, performer)

    matched_wk = 0
    for tid, artist, title, ry in cur.execute(
        "SELECT track_id, artist, title, release_year FROM track_meta WHERE release_year IS NOT NULL"
    ).fetchall():
        if not (artist and title and ry):
            continue
        nt = norm(title)
        na = norm(artist)
        if not nt:
            continue
        for dy in (0, 1, -1, 2, -2):
            y = ry + dy
            hit = history_by_year.get(y, {}).get(nt)
            if not hit:
                continue
            peak, woc, chart_performer = hit
            if na and not artist_match(na, chart_performer):
                continue
            cur.execute("""
                INSERT INTO track_chart_match
                  (track_id, chart_name, chart_year, peak_position, weeks_on_chart)
                VALUES (?, 'billboard_hot100_weekly', ?, ?, ?)
                ON CONFLICT(track_id, chart_name) DO UPDATE SET
                  chart_year = excluded.chart_year,
                  peak_position = excluded.peak_position,
                  weeks_on_chart = excluded.weeks_on_chart,
                  matched_at = CURRENT_TIMESTAMP
            """, (tid, y, peak, woc))
            matched_wk += 1
            break
    print(f"track_chart_match: {matched_wk} BB tracks matched to billboard_hot100 weekly all-time")

    # --- Headline analysis results (flattened from JSON) ---
    total_results = 0
    for path, name in RESULT_JSONS:
        n = ingest_result_json(cur, path, name)
        if n:
            print(f"analysis_results: {n} metrics from {path.name} ({name})")
            total_results += n

    conn.commit()
    conn.close()

    # Summary
    conn = sqlite3.connect(AUX_DB)
    for tbl in ("track_meta", "track_lastfm", "chart_yearend", "chart_song_history",
                "track_chart_match", "analysis_results"):
        (n,) = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()
        print(f"  {tbl}: {n} rows")
    print(f"\naux DB: {AUX_DB} ({AUX_DB.stat().st_size/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
