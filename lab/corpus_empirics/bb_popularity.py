"""Is Big Bootie acapella choice driven by at-time-of-release popularity?

Combines two genre-blind, era-aware signals:

  1) Wikipedia "Billboard Year-End Hot 100 singles of <year>" tables (1958-2024)
     — captures whether a track was a US chart hit in its release year. Pure
     at-time-of-release signal.

  2) Last.fm `track.getInfo` (listeners + playcount) — cumulative since ~2002
     but listener pool is community-skewed older/indie vs Spotify, so it
     captures cult / staying-power rather than the streaming-era recency bias.

Reports BB acapella vs BB instrumental on both signals, plus a four-quadrant
analysis: was the track a hit AND remembered, hit AND forgotten, deep-cut AND
remembered, deep-cut AND obscure.

All HTTP is cached on disk — re-runs are free.
"""

from __future__ import annotations

import csv
import html
import json
import os
import re
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, median

# ---------------- config ----------------
DB = Path("data/db/music_database.db")
CACHE_DIR = Path("data/analysis")
META_CSV = CACHE_DIR / "bb_track_meta.csv"
SPOTIFY_YEARS_CSV = CACHE_DIR / "spotify_release_dates.csv"
BILLBOARD_JSON = CACHE_DIR / "billboard_yearend.json"
LASTFM_JSON = CACHE_DIR / "lastfm_track_info.json"
OUT_JSON = CACHE_DIR / "bb_popularity.json"
UA = "tracklist-engine-research/0.1 (https://github.com/jca225)"


def load_env() -> None:
    if not Path(".env").exists():
        return
    for line in open(".env"):
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.strip().split("=", 1)
            os.environ.setdefault(k, v)


# ---------------- text normalization ----------------
PAREN_RE = re.compile(r"\([^)]*\)")
FEAT_RE = re.compile(r"\b(feat|ft|featuring|w/|with)\.?\b.*$", re.I)
NON_ALNUM = re.compile(r"[^a-z0-9]+")


def norm(s: str) -> str:
    if not s:
        return ""
    s = html.unescape(s).lower()
    s = PAREN_RE.sub(" ", s)
    s = FEAT_RE.sub(" ", s)
    s = NON_ALNUM.sub(" ", s).strip()
    s = " ".join(s.split())
    return s


# ---------------- per-row extraction ----------------
NAME_RE = re.compile(rb'itemprop="name"[^>]*content="([^"]+)"|content="([^"]+)"\s+itemprop="name"')
ARTIST_RE = re.compile(rb'itemprop="byArtist"[^>]*content="([^"]+)"|content="([^"]+)"\s+itemprop="byArtist"')


def extract_meta(raw_html: bytes) -> tuple[str, str]:
    n = NAME_RE.search(raw_html)
    a = ARTIST_RE.search(raw_html)
    name = ""
    artist = ""
    if n:
        name = html.unescape((n.group(1) or n.group(2)).decode())
    if a:
        artist = html.unescape((a.group(1) or a.group(2)).decode())
    # title = name minus the leading "artist - " if present
    title = name
    if artist and name.lower().startswith(artist.lower()):
        rest = name[len(artist):].lstrip(" -")
        if rest:
            title = rest
    # strip leading "ft. X -" / "feat. X -" prefix from title
    title = re.sub(r"^\s*(ft|feat|featuring)\.?\s+[^-]+-\s*", "", title, flags=re.I).strip()
    return artist.strip(), title.strip()


def pull_bb_meta(conn: sqlite3.Connection) -> list[dict]:
    if META_CSV.exists():
        return [dict(r) for r in csv.DictReader(open(META_CSV))]
    q = """
    WITH bb_sets AS (
      SELECT set_id, CAST(substr(date_played, 1, 4) AS INTEGER) AS set_year
      FROM dj_sets WHERE LOWER(title) LIKE '%big bootie%'),
    bb_rows AS (
      SELECT r.set_id, r.row_index,
             json_extract(r.data_attrs_json, '$."data-trackid"') AS track_id,
             r.raw_html,
             CASE WHEN r.text_excerpt LIKE 'w/%' THEN 'acapella'
                  WHEN r.text_excerpt GLOB '[0-9]*' THEN 'instrumental' END AS role
      FROM dj_set_rows r WHERE r.set_id IN (SELECT set_id FROM bb_sets))
    SELECT b.set_id, s.set_year, b.row_index, b.role, b.track_id, b.raw_html,
           m.player_id AS sid
    FROM bb_rows b JOIN bb_sets s USING (set_id)
    LEFT JOIN dj_set_track_media_links m
      ON m.track_id = b.track_id AND m.set_id = b.set_id AND m.platform = 'spotify'
    WHERE b.role IS NOT NULL AND b.track_id IS NOT NULL
    ORDER BY b.set_id, b.row_index
    """
    rows = []
    for r in conn.execute(q):
        d = dict(r)
        artist, title = extract_meta(d["raw_html"].encode() if isinstance(d["raw_html"], str) else d["raw_html"])
        d.pop("raw_html")
        d["artist"] = artist
        d["title"] = title
        rows.append(d)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(META_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


# ---------------- Billboard Year-End scraping ----------------
def fetch_billboard(year: int) -> list[tuple[int, str, str]]:
    url = f"https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_{year}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        body = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "replace")
    except Exception:
        return []
    m = re.search(r'<table[^>]*class="[^"]*wikitable[^"]*"[^>]*>(.*?)</table>', body, re.S)
    if not m:
        return []
    table = m.group(1)
    out = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.S):
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, re.S)
        if len(cells) < 3:
            continue
        def clean(s: str) -> str:
            return html.unescape(re.sub(r"<[^>]+>", "", s)).strip().strip('"').strip()
        try:
            rank = int(clean(cells[0]))
        except ValueError:
            continue
        title = clean(cells[1])
        artist = clean(cells[2])
        out.append((rank, title, artist))
    return out


def load_billboard(years: range) -> dict[int, list[tuple[int, str, str]]]:
    if BILLBOARD_JSON.exists():
        data = json.loads(BILLBOARD_JSON.read_text())
        return {int(k): [tuple(t) for t in v] for k, v in data.items()}
    out: dict[int, list[tuple[int, str, str]]] = {}
    for i, y in enumerate(years, 1):
        out[y] = fetch_billboard(y)
        print(f"  billboard {y}: {len(out[y])} entries")
        time.sleep(0.5)  # polite to wikipedia
    BILLBOARD_JSON.write_text(json.dumps(out))
    return out


# ---------------- Last.fm ----------------
def lastfm_call(api_key: str, artist: str, track: str) -> dict | None:
    q = urllib.parse.urlencode({
        "method": "track.getInfo", "api_key": api_key,
        "artist": artist, "track": track,
        "autocorrect": 1, "format": "json",
    })
    req = urllib.request.Request(f"https://ws.audioscrobbler.com/2.0/?{q}",
                                  headers={"User-Agent": UA})
    try:
        body = urllib.request.urlopen(req, timeout=15).read()
        d = json.loads(body)
    except Exception:
        return None
    if "error" in d:
        return {"error": d.get("error"), "message": d.get("message", "")}
    t = d.get("track")
    if not t:
        return None
    listeners = int(t.get("listeners") or 0)
    playcount = int(t.get("playcount") or 0)
    art = t.get("artist")
    if isinstance(art, dict):
        art = art.get("name")
    return {"name": t.get("name"), "artist": art,
            "listeners": listeners, "playcount": playcount,
            "mbid": t.get("mbid") or "",
            "url": t.get("url") or ""}


def lastfm_batch(api_key: str, queries: list[tuple[str, str]]) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if LASTFM_JSON.exists():
        cache = json.loads(LASTFM_JSON.read_text())
    keys = [f"{a}|||{t}" for a, t in queries]
    missing = [(a, t, k) for (a, t), k in zip(queries, keys) if k not in cache]
    if not missing:
        return cache
    print(f"  fetching {len(missing)} new Last.fm queries (cache has {len(cache)})…")
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(lastfm_call, api_key, a, t): k for a, t, k in missing}
        for fut in as_completed(futures):
            k = futures[fut]
            cache[k] = fut.result() or {"error": -1}
            done += 1
            if done % 200 == 0:
                LASTFM_JSON.write_text(json.dumps(cache))
                print(f"    {done}/{len(missing)} ({time.time()-t0:.0f}s)")
    LASTFM_JSON.write_text(json.dumps(cache))
    print(f"  done in {time.time()-t0:.0f}s")
    return cache


# ---------------- Matching BB→Billboard ----------------
def index_billboard(bb: dict[int, list]) -> dict[int, dict[str, tuple[int, str, str]]]:
    out: dict[int, dict[str, tuple[int, str, str]]] = {}
    for year, entries in bb.items():
        idx = {}
        for rank, title, artist in entries:
            idx[norm(title)] = (rank, title, artist)
        out[year] = idx
    return out


def match_chart(title: str, artist: str, idx: dict[int, dict],
                year: int, window: int = 1) -> tuple[int, int] | None:
    """Returns (chart_year, rank) if matched, else None. window=1 also checks year+1
    (December-released tracks chart the following year-end)."""
    nt = norm(title)
    na = norm(artist)
    if not nt:
        return None
    for dy in [0, 1, -1][:1 + window * 2]:
        y = year + dy
        if y not in idx:
            continue
        hit = idx[y].get(nt)
        if not hit:
            continue
        rank, _, chart_artist = hit
        if na and not (na in norm(chart_artist) or norm(chart_artist).split()[0] in na):
            continue  # title matched, but artist differs — skip (avoids covers)
        return y, rank
    return None


# ---------------- Statistics helpers ----------------
def mannwhitney(a: list[float], b: list[float]) -> tuple[float, float, float]:
    pool = [(v, "a") for v in a] + [(v, "b") for v in b]
    pool.sort()
    ra = rb = 0.0
    i = 0
    while i < len(pool):
        j = i
        while j + 1 < len(pool) and pool[j + 1][0] == pool[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            if pool[k][1] == "a":
                ra += avg
            else:
                rb += avg
        i = j + 1
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return 0.0, 0.0, 0.0
    Ua = ra - n_a * (n_a + 1) / 2
    Ub = n_a * n_b - Ua
    rbc = 1 - 2 * min(Ua, Ub) / (n_a * n_b)
    sign = 1 if (mean(a) > mean(b)) else -1
    return Ua, Ub, sign * abs(rbc)


def pct(xs: list, p: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(p * len(s)))]


# ---------------- Main ----------------
def main() -> int:
    load_env()
    api_key = os.environ.get("LASTFM_API_KEY")
    if not api_key:
        print("ERROR: LASTFM_API_KEY not set", file=sys.stderr)
        return 1

    print("loading BB rows + metadata…")
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = pull_bb_meta(conn)
    print(f"  {len(rows)} BB rows")

    # spotify release year cache
    sp_year = {}
    if SPOTIFY_YEARS_CSV.exists():
        for sid, y in csv.reader(open(SPOTIFY_YEARS_CSV)):
            sp_year[sid] = int(y) if y else None

    # attach release year per row (from spotify cache when available)
    for r in rows:
        sid = r.get("sid")
        r["release_year"] = sp_year.get(sid) if sid else None

    # unique (artist, title) pairs for last.fm
    pair_to_tids: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in rows:
        if r["artist"] and r["title"]:
            pair_to_tids[(r["artist"], r["title"])].add(r["track_id"])
    print(f"  {len(pair_to_tids)} unique (artist, title) pairs")

    print("\nloading Billboard Year-End Hot 100 (1958-2024)…")
    bb_chart = load_billboard(range(1958, 2025))
    bb_idx = index_billboard(bb_chart)
    print(f"  loaded {sum(len(v) for v in bb_chart.values())} chart entries across {sum(1 for v in bb_chart.values() if v)} years")

    print("\nquerying Last.fm…")
    queries = list(pair_to_tids.keys())
    lfm = lastfm_batch(api_key, queries)

    # ---------------- aggregate per-track signal ----------------
    track_signal: dict[str, dict] = {}
    for r in rows:
        tid = r["track_id"]
        if tid in track_signal:
            continue
        if not r["artist"] or not r["title"]:
            continue
        key = f"{r['artist']}|||{r['title']}"
        lfm_d = lfm.get(key) or {}
        if "error" in lfm_d:
            listeners = playcount = None
        else:
            listeners = lfm_d.get("listeners")
            playcount = lfm_d.get("playcount")
        chart = None
        if r["release_year"]:
            chart = match_chart(r["title"], r["artist"], bb_idx, r["release_year"])
        track_signal[tid] = {
            "role": r["role"],
            "artist": r["artist"],
            "title": r["title"],
            "release_year": r["release_year"],
            "listeners": listeners,
            "playcount": playcount,
            "charted_year": chart[0] if chart else None,
            "charted_rank": chart[1] if chart else None,
        }

    # ---------------- analysis ----------------
    def role_set(role: str) -> list[dict]:
        return [v for v in track_signal.values() if v["role"] == role]

    aca = role_set("acapella")
    ins = role_set("instrumental")

    def chart_rate(tracks: list[dict]) -> tuple[int, int, float]:
        with_year = [t for t in tracks if t["release_year"]]
        charted = [t for t in with_year if t["charted_year"]]
        return len(charted), len(with_year), (len(charted) / len(with_year) if with_year else 0.0)

    aca_c, aca_n, aca_rate = chart_rate(aca)
    ins_c, ins_n, ins_rate = chart_rate(ins)

    def listeners_dist(tracks: list[dict]) -> dict:
        xs = [t["listeners"] for t in tracks if t["listeners"] is not None]
        if not xs:
            return {"n": 0}
        return {
            "n": len(xs),
            "median": int(median(xs)),
            "mean": int(mean(xs)),
            "p10": int(pct(xs, 0.10)),
            "p25": int(pct(xs, 0.25)),
            "p75": int(pct(xs, 0.75)),
            "p90": int(pct(xs, 0.90)),
            "max": int(max(xs)),
        }

    aca_lst = [t["listeners"] for t in aca if t["listeners"] is not None]
    ins_lst = [t["listeners"] for t in ins if t["listeners"] is not None]
    _, _, rbc = mannwhitney(aca_lst, ins_lst)

    # Four quadrant
    def quad(t: dict) -> str | None:
        if t["listeners"] is None:
            return None
        hi_lfm = t["listeners"] >= 100_000
        chart = t["charted_year"] is not None
        if chart and hi_lfm: return "hit_remembered"
        if chart and not hi_lfm: return "hit_forgotten"
        if not chart and hi_lfm: return "deepcut_remembered"
        return "deepcut_obscure"

    def quad_breakdown(tracks):
        out = defaultdict(int)
        for t in tracks:
            q = quad(t)
            if q: out[q] += 1
        return dict(out)

    result = {
        "n_acapella_tracks": len(aca),
        "n_instrumental_tracks": len(ins),
        "chart_rate": {
            "acapella": {"charted": aca_c, "with_year": aca_n, "rate": round(aca_rate, 3)},
            "instrumental": {"charted": ins_c, "with_year": ins_n, "rate": round(ins_rate, 3)},
        },
        "listeners_distribution": {
            "acapella": listeners_dist(aca),
            "instrumental": listeners_dist(ins),
            "mannwhitney_rank_biserial": round(rbc, 3),
        },
        "four_quadrant_acapella": quad_breakdown(aca),
        "four_quadrant_instrumental": quad_breakdown(ins),
    }

    # examples per quadrant
    examples = {}
    for q in ("hit_remembered", "hit_forgotten", "deepcut_remembered", "deepcut_obscure"):
        ex = sorted([t for t in aca if quad(t) == q],
                    key=lambda t: -(t["listeners"] or 0))[:3]
        examples[q] = [{"artist": t["artist"], "title": t["title"], "year": t["release_year"],
                        "listeners": t["listeners"], "rank": t["charted_rank"]} for t in ex]
    result["acapella_examples"] = examples

    print("\n=== results ===")
    print(json.dumps(result, indent=2))
    OUT_JSON.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
