"""Is acapella-era choice orthogonal to instrumental-era choice in Big Bootie?

Pulls all Big Bootie mashup slots from the local DB, treats the `NN` row in each
slot as the instrumental host and `w/` rows as layered acapellas (Two Friends'
typical convention), fetches release year for spotify-linked tracks from the
unauthenticated open.spotify.com embed page (cached locally), then tests:

  H0: gap = acapella_year - instrumental_year is centered on 0 (matched era)
  H1: instrumental-year and acapella-year are independent (orthogonal)

Reports paired Pearson + Spearman, the gap-distribution, and a shuffle-null
(permute acapella years across mashup slots within a volume) for the observed
correlation. Orthogonal => observed corr sits inside the null distribution.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from corpus_empirics.stats import pearson
from urllib.request import Request, urlopen

DB = Path("data/db/music_database.db")
CACHE = Path("data/analysis/spotify_release_dates.csv")
OUT = Path("data/analysis/bb_era_orthogonality.json")

RELEASE_RE = re.compile(rb'"releaseDate":\{"isoString":"(\d{4})-')
UA = "Mozilla/5.0 (analysis-script)"


def load_cache() -> dict[str, int | None]:
    if not CACHE.exists():
        return {}
    out: dict[str, int | None] = {}
    with CACHE.open() as f:
        for sid, year in csv.reader(f):
            out[sid] = int(year) if year else None
    return out


def save_cache(c: dict[str, int | None]) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("w", newline="") as f:
        w = csv.writer(f)
        for sid, year in sorted(c.items()):
            w.writerow([sid, year if year is not None else ""])


def fetch_year(sid: str) -> tuple[str, int | None]:
    url = f"https://open.spotify.com/embed/track/{sid}"
    try:
        req = Request(url, headers={"User-Agent": UA})
        with urlopen(req, timeout=10) as r:
            body = r.read()
        m = RELEASE_RE.search(body)
        return sid, int(m.group(1)) if m else None
    except Exception:
        return sid, None


def pull_rows(conn: sqlite3.Connection) -> list[dict]:
    q = """
    WITH bb_sets AS (
      SELECT set_id, title, CAST(substr(date_played, 1, 4) AS INTEGER) AS set_year
      FROM dj_sets WHERE LOWER(title) LIKE '%big bootie%'),
    bb_rows AS (
      SELECT r.set_id, r.row_index,
             json_extract(r.data_attrs_json, '$."data-trackid"') AS track_id,
             r.text_excerpt,
             CASE WHEN r.text_excerpt LIKE 'w/%' THEN 'acapella'
                  WHEN r.text_excerpt GLOB '[0-9]*' THEN 'instrumental' END AS role
      FROM dj_set_rows r WHERE r.set_id IN (SELECT set_id FROM bb_sets))
    SELECT b.set_id, s.title, s.set_year, b.row_index, b.role, b.track_id,
           m.player_id AS spotify_id
    FROM bb_rows b
    JOIN bb_sets s USING (set_id)
    LEFT JOIN dj_set_track_media_links m
      ON m.track_id = b.track_id AND m.set_id = b.set_id AND m.platform = 'spotify'
    WHERE b.role IS NOT NULL AND b.track_id IS NOT NULL
    ORDER BY b.set_id, b.row_index
    """
    return [dict(r) for r in conn.execute(q)]


def assign_slots(rows: list[dict]) -> list[dict]:
    """Walk rows in row_index order: every `instrumental` row opens a new slot,
    subsequent `acapella` rows belong to that slot until the next instrumental."""
    out: list[dict] = []
    slot_id = -1
    cur_instrumental: dict | None = None
    for r in rows:
        if r["role"] == "instrumental":
            slot_id += 1
            cur_instrumental = r
            r["slot_id"] = slot_id
            r["instr_track_id"] = r["track_id"]
            out.append(r)
        else:
            if cur_instrumental is None:
                continue  # acapella before any instrumental — skip
            r["slot_id"] = slot_id
            r["instr_track_id"] = cur_instrumental["track_id"]
            out.append(r)
    return out


def main() -> None:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = pull_rows(conn)
    print(f"Pulled {len(rows)} BB rows ({sum(1 for r in rows if r['role']=='instrumental')} instrumentals, "
          f"{sum(1 for r in rows if r['role']=='acapella')} acapellas)")

    rows = assign_slots(rows)
    spotify_ids = sorted({r["spotify_id"] for r in rows if r["spotify_id"]})
    print(f"{len(spotify_ids)} unique spotify track IDs to look up")

    cache = load_cache()
    missing = [s for s in spotify_ids if s not in cache]
    print(f"  {len(cache)} in cache, {len(missing)} to fetch")

    if missing:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(fetch_year, s) for s in missing]
            for i, fut in enumerate(as_completed(futures), 1):
                sid, year = fut.result()
                cache[sid] = year
                if i % 100 == 0:
                    save_cache(cache)
                    print(f"  {i}/{len(missing)} fetched ({time.time()-t0:.0f}s)")
        save_cache(cache)
        print(f"  done in {time.time()-t0:.0f}s")

    # Build pairs: (instrumental_year, acapella_year) per acapella row that has both.
    pairs: list[tuple[int, int, int, int]] = []  # instr_year, acap_year, set_year, slot_id
    instr_year_by_slot: dict[int, int] = {}
    for r in rows:
        if r["role"] == "instrumental" and r["spotify_id"]:
            y = cache.get(r["spotify_id"])
            if y is not None:
                instr_year_by_slot[r["slot_id"]] = y
    for r in rows:
        if r["role"] == "acapella" and r["spotify_id"]:
            y = cache.get(r["spotify_id"])
            iy = instr_year_by_slot.get(r["slot_id"])
            if y is not None and iy is not None:
                pairs.append((iy, y, r["set_year"], r["slot_id"]))

    print(f"\n{len(pairs)} (instrumental, acapella) year-pairs with both sides resolved")
    if not pairs:
        print("not enough data — exiting")
        return

    instr = [p[0] for p in pairs]
    acap = [p[1] for p in pairs]
    setyr = [p[2] for p in pairs]
    gaps = [a - i for i, a in zip(instr, acap)]

    # Pearson + Spearman without scipy.
    def spearman(xs, ys):
        def rank(vs):
            order = sorted(range(len(vs)), key=lambda i: vs[i])
            ranks = [0.0] * len(vs)
            i = 0
            while i < len(vs):
                j = i
                while j + 1 < len(vs) and vs[order[j + 1]] == vs[order[i]]:
                    j += 1
                avg = (i + j) / 2 + 1
                for k in range(i, j + 1):
                    ranks[order[k]] = avg
                i = j + 1
            return ranks
        return pearson(rank(xs), rank(ys))

    r_p = pearson(instr, acap)
    r_s = spearman(instr, acap)

    # Partial correlation: control for set_year (era of the BB volume itself).
    # Both instr and acap years drift forward as the BB series ages — that alone
    # could induce a spurious positive correlation. Strip the linear effect of
    # set_year from both, then re-correlate the residuals.
    def residuals(ys, xs):
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs)
        b = num / den if den else 0.0
        a = my - b * mx
        return [y - (a + b * x) for x, y in zip(xs, ys)]

    instr_resid = residuals(instr, setyr)
    acap_resid = residuals(acap, setyr)
    r_partial = pearson(instr_resid, acap_resid)

    # Shuffle null: permute acapella years across slots (within full corpus)
    # and recompute Pearson. If observed sits inside the null, the choice is
    # orthogonal.
    import random
    random.seed(0)
    null = []
    for _ in range(2000):
        shuffled = acap[:]
        random.shuffle(shuffled)
        null.append(pearson(instr, shuffled))
    null.sort()
    p_two = sum(1 for v in null if abs(v) >= abs(r_p)) / len(null)

    # Gap stats.
    n = len(gaps)
    gmean = sum(gaps) / n
    gvar = sum((g - gmean) ** 2 for g in gaps) / n
    gsd = gvar ** 0.5
    gabs = sorted(abs(g) for g in gaps)
    gmed_abs = gabs[n // 2]

    # Bucket distribution
    bins = [(-100, -10), (-10, -5), (-5, -1), (-1, 1), (1, 5), (5, 10), (10, 100)]
    bin_labels = ["acap >10y older", "5–10y older", "1–5y older", "same year (±1)",
                  "1–5y newer", "5–10y newer", "acap >10y newer"]
    counts = [sum(1 for g in gaps if lo <= g < hi) for lo, hi in bins]

    result = {
        "n_pairs": n,
        "instr_year_range": [min(instr), max(instr)],
        "acap_year_range": [min(acap), max(acap)],
        "pearson_r": round(r_p, 4),
        "spearman_r": round(r_s, 4),
        "partial_r_controlling_set_year": round(r_partial, 4),
        "shuffle_null_p_two_sided": round(p_two, 4),
        "shuffle_null_ci_95": [round(null[int(0.025 * len(null))], 4),
                                round(null[int(0.975 * len(null))], 4)],
        "gap_mean": round(gmean, 2),
        "gap_sd": round(gsd, 2),
        "gap_median_abs": gmed_abs,
        "gap_distribution": dict(zip(bin_labels, counts)),
    }

    print("\n=== results ===")
    print(json.dumps(result, indent=2))
    OUT.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
