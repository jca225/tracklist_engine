"""Pulls Spotify Daily Top 200 chart history for BB tracks via kworb.net.

kworb.net mirrors the Spotify Charts archive (charts.spotify.com), which has
daily / weekly Top 200 by country going back to ~December 2017. Each
per-track URL exposes the song's lifetime per-country peak position, weeks
on chart, and total stream counts.

Coverage caveat: Spotify Charts data starts late 2017, so pre-2017 catalog
tracks will have no signal here even if hugely popular at release. This is
the modern-era complement to the Hot 100 signals already in aux.db.

Strategy: BB tracks have Spotify IDs already attached
(`dj_set_track_media_links.player_id WHERE platform='spotify'`), so matching
is exact-by-ID — no fuzzy matching needed. For each Spotify ID:

  - GET https://kworb.net/spotify/track/{spotify_id}.html
  - 404 → never charted on any country's Spotify Top 200 (cache as 'uncharted')
  - 200 → parse header row (country codes), Total row (lifetime streams),
          Peak row (per-country best position), and count weekly data rows
          (weeks_on_chart proxy)

Caches results in `data/analysis/spotify_kworb.json` so re-runs are free.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Source the (track_id → spotify_id) mapping from the cached BB metadata CSV
# rather than re-querying the 11GB main DB. The CSV is produced by
# eda/corpus_empirics/bb_popularity.py:pull_bb_meta() (which runs a one-time SQL extract,
# bypasses raw_html each row, and caches as flat columns) — far faster than
# re-running that query for our purposes.
BB_META_CSV = Path("data/analysis/bb_track_meta.csv")
CACHE = Path("data/analysis/spotify_kworb.json")
UA = "tracklist-engine-research/0.1 (https://github.com/jca225)"

# Subset of countries we'll record peak/streams for explicitly. We persist the
# full per-country breakdown in the cache anyway — these are just the ones
# surfaced as flat columns. Global+US are the headline numbers; the others let
# us derive "n_countries_charted" as a global-reach proxy.
KEY_COUNTRIES = ("Global", "US", "GB", "DE", "BR", "MX", "FR", "AU", "CA", "ES")


def pull_bb_spotify_ids() -> dict[str, str]:
    """Returns {track_id: spotify_id} for every BB track that has one,
    sourced from the bb_track_meta.csv cache (much faster than re-querying
    the main DB)."""
    if not BB_META_CSV.exists():
        print(f"ERROR: {BB_META_CSV} not found. Run eda/corpus_empirics/bb_popularity.py first "
              "to populate the BB metadata cache.", file=sys.stderr)
        sys.exit(1)
    out: dict[str, str] = {}
    with open(BB_META_CSV) as f:
        for r in csv.DictReader(f):
            tid = r.get("track_id"); sid = r.get("sid")
            if tid and sid:
                out[tid] = sid  # collapse multiple set-level rows
    return out


# ---------------- parsing ----------------
# Header row: <th>Date</th><th>Global</th><th>US</th>... or first <tr> with
# country-code <td>s. The actual table on kworb uses <td> tags throughout.
TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
# Cells may be <td> (body) or <th> (header) — accept both
TD_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S)
SPAN_PEAK = re.compile(r'<span class="p">([^<]+)</span>')
SPAN_STREAMS = re.compile(r'<span class="s">([^<]+)</span>')


def parse_track_page(html: str) -> dict | None:
    """Returns the parsed chart history, or None if the page has no chart table."""
    # Find the main chart-history table. It contains a row starting with <td>Peak</td>.
    m = re.search(r"<table[^>]*>.*?<tr[^>]*><td>Peak</td>.*?</table>", html, re.S)
    if not m:
        return None
    table = m.group(0)
    rows = TR_RE.findall(table)
    if len(rows) < 3:
        return None

    def cells(row: str) -> list[str]:
        return TD_RE.findall(row)

    header = cells(rows[0])
    if not header:
        return None
    # Header[0] is "Date"; the rest are country codes (with Global first)
    country_cols = [strip_tags(c).strip() for c in header[1:]]

    # Row 1 = "Total" (lifetime streams per country)
    total_row = cells(rows[1])
    totals: dict[str, int | None] = {}
    if total_row and strip_tags(total_row[0]).strip() == "Total":
        for cc, cell in zip(country_cols, total_row[1:]):
            txt = strip_tags(cell).strip().replace(",", "")
            totals[cc] = int(txt) if txt.isdigit() else None

    # Row 2 = "Peak" (best peak position per country)
    peak_row = cells(rows[2])
    peaks: dict[str, dict | None] = {}
    if peak_row and strip_tags(peak_row[0]).strip() == "Peak":
        for cc, cell in zip(country_cols, peak_row[1:]):
            p_match = SPAN_PEAK.search(cell)
            if not p_match:
                peaks[cc] = None
                continue
            try:
                peak = int(p_match.group(1))
            except ValueError:
                peaks[cc] = None
                continue
            s_match = SPAN_STREAMS.search(cell)
            max_week_streams = None
            if s_match:
                txt = s_match.group(1).replace(",", "")
                if txt.isdigit():
                    max_week_streams = int(txt)
            peaks[cc] = {"peak": peak, "max_week_streams": max_week_streams}

    # Remaining rows = per-week data (date, then per-country peak+streams)
    data_rows = rows[3:]
    weeks_global = sum(
        1 for r in data_rows
        if (c := cells(r)) and c[1].strip() not in ("", "--")
    )
    # Debut = first row where Global cell is not "--"
    debut_date = None
    for r in data_rows:
        c = cells(r)
        if len(c) < 2:
            continue
        date = strip_tags(c[0]).strip()
        if c[1].strip() not in ("", "--") and date:
            debut_date = date
            break

    n_countries_charted = sum(1 for v in peaks.values() if v is not None)

    out: dict = {
        "country_cols": country_cols,
        "totals": totals,
        "peaks": peaks,
        "n_countries_charted": n_countries_charted,
        "weeks_on_chart_global": weeks_global,
        "debut_date": debut_date,
    }
    # Surface key-country flat fields for convenience
    for cc in KEY_COUNTRIES:
        p = peaks.get(cc)
        out[f"peak_{cc.lower()}"] = p["peak"] if p else None
        out[f"streams_{cc.lower()}"] = totals.get(cc)
    return out


def strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s)


# ---------------- fetching ----------------
def fetch_track(sid: str) -> dict:
    """Returns a result dict with status + parsed signal (or charted=False on 404)."""
    url = f"https://kworb.net/spotify/track/{sid}.html"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"charted": False, "status": 404}
        return {"charted": None, "status": e.code, "error": str(e)}
    except Exception as e:
        return {"charted": None, "status": -1, "error": str(e)}
    parsed = parse_track_page(html)
    if not parsed:
        return {"charted": False, "status": 200, "parse_failed": True}
    parsed["charted"] = True
    parsed["status"] = 200
    return parsed


def fetch_all(sid_by_tid: dict[str, str], cache: dict, workers: int = 8) -> dict:
    """Concurrent fetch with a small thread pool. Flushes cache to disk every
    25 records so a crash loses at most ~25 fetches, and prints progress
    line-flushed every 25 records so the operator can see it live."""
    missing = [(tid, sid) for tid, sid in sid_by_tid.items() if sid not in cache]
    if not missing:
        print(f"  cache complete ({len(cache)} entries)", flush=True)
        return cache
    print(f"  fetching {len(missing)} new tracks (cache has {len(cache)})…",
          flush=True)
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_track, sid): (tid, sid) for tid, sid in missing}
        for fut in as_completed(futures):
            tid, sid = futures[fut]
            try:
                cache[sid] = fut.result()
            except Exception as e:
                cache[sid] = {"charted": None, "status": -1, "error": str(e)}
            done += 1
            if done % 25 == 0:
                CACHE.write_text(json.dumps(cache))
                rate = done / (time.time() - t0)
                eta = (len(missing) - done) / rate if rate > 0 else 0
                print(f"    {done}/{len(missing)} "
                      f"({(time.time()-t0):.0f}s elapsed, "
                      f"{rate:.1f}/s, ETA {eta:.0f}s)", flush=True)
    CACHE.write_text(json.dumps(cache))
    print(f"  done in {time.time()-t0:.0f}s", flush=True)
    return cache


# ---------------- main ----------------
def main() -> int:
    sid_by_tid = pull_bb_spotify_ids()
    print(f"BB tracks with spotify_id: {len(sid_by_tid)}", flush=True)

    cache: dict[str, dict] = {}
    if CACHE.exists():
        cache = json.loads(CACHE.read_text())

    cache = fetch_all(sid_by_tid, cache)

    # ---------------- summary ----------------
    charted = [v for v in cache.values() if v.get("charted")]
    uncharted = [v for v in cache.values() if v.get("charted") is False]
    errored = [v for v in cache.values() if v.get("charted") is None]
    print(f"\ncache summary: {len(charted)} charted, "
          f"{len(uncharted)} uncharted, {len(errored)} errored")
    if not charted:
        return 0

    def pct_with(field: str) -> int:
        return sum(1 for v in charted if v.get(field) is not None)

    print(f"  with global peak: {pct_with('peak_global')}")
    print(f"  with US peak:     {pct_with('peak_us')}")
    print(f"  with GB peak:     {pct_with('peak_gb')}")

    # Peak distribution (Global)
    peaks_global = [v["peak_global"] for v in charted if v.get("peak_global")]
    if peaks_global:
        peaks_global.sort()
        n = len(peaks_global)
        print(f"\nGlobal peak distribution (n={n}):")
        print(f"  #1 peak:        {sum(1 for p in peaks_global if p == 1)}")
        print(f"  top-10 peak:    {sum(1 for p in peaks_global if p <= 10)}")
        print(f"  top-40 peak:    {sum(1 for p in peaks_global if p <= 40)}")
        print(f"  top-100 peak:   {sum(1 for p in peaks_global if p <= 100)}")
        print(f"  top-200 peak:   {n}")
        print(f"  median:         {peaks_global[n//2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
