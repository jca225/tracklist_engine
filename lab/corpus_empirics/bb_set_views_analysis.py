"""Does instrumental-popularity (or acapella-popularity) predict BB set popularity?

Set popularity = YouTube view count of the volume's upload (user-provided).
Track popularity = our aux.db signals (Last.fm listeners, Hot 100 year-end
hit-rate).

Controls for age — older volumes have had more years to accumulate views — by
computing partial correlations against `set_year` and via views-per-year.

Persists view counts to aux.db `set_views` so future runs can use them.
"""

from __future__ import annotations

import re
import sqlite3
import sys
from datetime import date
from pathlib import Path
from statistics import median

from corpus_empirics.stats import partial_pearson, pearson

AUX_DB = Path("data/analysis/aux.db")
MAIN_DB = Path("data/db/music_database.db")
TODAY_YEAR = 2026

# Volume → YouTube view count (M = millions, K = thousands). Vol 26 had no
# count visible in the source listing, treat as missing. Vol 2 was missing
# from the source.
VIEWS: dict[int, float] = {
    25: 3_140_000, 24: 3_470_000, 23: 4_180_000, 22: 4_750_000, 21: 5_980_000,
    20: 5_580_000, 19: 7_390_000, 18: 11_900_000, 17: 15_300_000,
    16: 7_820_000,  15: 20_000_000, 14: 11_700_000, 13: 13_000_000,
    12: 7_580_000,  11: 21_700_000, 10: 9_930_000, 9: 3_630_000,
    8: 2_610_000,   7: 1_090_000,   6: 1_320_000,  5: 939_000,
    4: 820_000,     3: 666_000,     1: 976_000,
}


def parse_volume(title: str) -> int | None:
    """Extract a volume number from a dj_sets.title string."""
    if "big bootie" not in title.lower():
        return None
    if "@ big bootie" in title.lower():
        return None  # live set, not the studio mix
    # Match Vol N, Volume N, Mix N, Mix Episode N (BB11 was "Episode 11")
    for pat in [r"vol\.?\s*(\d+)\b", r"volume\s*(\d+)\b", r"mix\s+0?(\d+)\b",
                r"episode\s+(\d+)\b"]:
        m = re.search(pat, title.lower())
        if m:
            return int(m.group(1))
    return None


SCHEMA = """
CREATE TABLE IF NOT EXISTS aux.set_views (
  set_id      TEXT NOT NULL,
  platform    TEXT NOT NULL,
  view_count  INTEGER,
  source      TEXT,
  observed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (set_id, platform)
);
"""



def main() -> int:
    if not AUX_DB.exists() or not MAIN_DB.exists():
        print("missing aux.db or main DB; run aux_db_sync first", file=sys.stderr)
        return 1

    conn = sqlite3.connect(MAIN_DB)
    conn.row_factory = sqlite3.Row
    # Map volume → set_id (studio mixes only, skip live "@ Big Bootie Land" sets)
    rows = conn.execute("""
        SELECT set_id, title, CAST(substr(date_played,1,4) AS INTEGER) AS set_year
        FROM dj_sets WHERE LOWER(title) LIKE '%big bootie%'
    """).fetchall()
    vol_to_set: dict[int, dict] = {}
    for r in rows:
        v = parse_volume(r["title"])
        if v is None or v in vol_to_set:
            continue
        vol_to_set[v] = dict(r)
    print(f"resolved {len(vol_to_set)} BB volumes from dj_sets")

    # Attach aux.db for queries
    conn.execute(f"ATTACH DATABASE '{AUX_DB}' AS aux")
    conn.executescript(SCHEMA)

    # Persist user-provided view counts
    for vol, views in VIEWS.items():
        s = vol_to_set.get(vol)
        if not s:
            continue
        conn.execute("""
            INSERT INTO aux.set_views (set_id, platform, view_count, source)
            VALUES (?, 'youtube', ?, 'user_provided_2026_05_18')
            ON CONFLICT(set_id, platform) DO UPDATE SET
              view_count = excluded.view_count,
              source = excluded.source,
              observed_at = CURRENT_TIMESTAMP
        """, (s["set_id"], int(views)))
    conn.commit()
    print(f"persisted {len(VIEWS)} view counts to aux.set_views")

    # ----- per-volume aggregated track-popularity metrics -----
    print("\nper-volume metrics:")
    rows_out = []
    for vol in sorted(VIEWS.keys()):
        s = vol_to_set.get(vol)
        if not s:
            print(f"  Vol {vol}: no set_id"); continue
        # Pull tracks in this set with role + popularity signals via aux.db
        track_rows = conn.execute("""
        WITH bb_rows AS (
          SELECT json_extract(r.data_attrs_json,'$."data-trackid"') AS tid,
                 CASE WHEN r.text_excerpt LIKE 'w/%' THEN 'acapella'
                      WHEN r.text_excerpt GLOB '[0-9]*' THEN 'instrumental' END AS role
          FROM dj_set_rows r WHERE r.set_id = ?)
        SELECT b.role, m.release_year, lf.listeners,
               CASE WHEN cm.rank IS NOT NULL THEN 1 ELSE 0 END AS charted
        FROM bb_rows b
        LEFT JOIN aux.track_meta m ON m.track_id = b.tid
        LEFT JOIN aux.track_lastfm lf ON lf.track_id = b.tid
        LEFT JOIN aux.track_chart_match cm ON cm.track_id = b.tid
        WHERE b.role IS NOT NULL
        """, (s["set_id"],)).fetchall()

        instr_lst = [r["listeners"] for r in track_rows if r["role"]=="instrumental" and r["listeners"] is not None]
        aca_lst   = [r["listeners"] for r in track_rows if r["role"]=="acapella" and r["listeners"] is not None]
        instr_chart = [r["charted"] for r in track_rows if r["role"]=="instrumental" and r["release_year"]]
        aca_chart   = [r["charted"] for r in track_rows if r["role"]=="acapella" and r["release_year"]]

        age = TODAY_YEAR - (s["set_year"] or TODAY_YEAR)
        views = VIEWS[vol]
        rec = {
            "vol": vol,
            "set_id": s["set_id"],
            "set_year": s["set_year"],
            "age": age,
            "views": views,
            "vpy": views / max(age, 1),  # views per year (rough age-normalization)
            "n_instr": len(instr_lst),
            "med_instr_listeners": median(instr_lst) if instr_lst else None,
            "instr_chart_rate": sum(instr_chart)/len(instr_chart) if instr_chart else None,
            "n_aca": len(aca_lst),
            "med_aca_listeners": median(aca_lst) if aca_lst else None,
            "aca_chart_rate": sum(aca_chart)/len(aca_chart) if aca_chart else None,
        }
        rows_out.append(rec)

    fmt = "{vol:>3} {set_year:>4} {age:>3} {views:>9.2g}  vpy={vpy:>7.1f}  inst_n={n_instr:>2} med_inst={mi:>7}  ch_inst={ci:>4}  aca_n={n_aca:>3} med_aca={ma:>8}  ch_aca={ca:>4}"
    print(f"  {'vol':>3} {'year':>4} {'age':>3} {'views':>9}  {'views/yr':>11}  {'#inst':>5}  {'med_inst':>8}  {'ch_inst':>7}  {'#aca':>5}  {'med_aca':>10}  {'ch_aca':>7}")
    for r in rows_out:
        print(fmt.format(
            vol=r["vol"], set_year=r["set_year"] or 0, age=r["age"], views=r["views"], vpy=r["vpy"],
            n_instr=r["n_instr"], mi=int(r["med_instr_listeners"]) if r["med_instr_listeners"] is not None else "—",
            ci=f"{r['instr_chart_rate']:.2f}" if r["instr_chart_rate"] is not None else "—",
            n_aca=r["n_aca"], ma=int(r["med_aca_listeners"]) if r["med_aca_listeners"] is not None else "—",
            ca=f"{r['aca_chart_rate']:.2f}" if r["aca_chart_rate"] is not None else "—",
        ))

    # ----- correlations -----
    valid = [r for r in rows_out if r["med_instr_listeners"] is not None
                                 and r["med_aca_listeners"] is not None
                                 and r["set_year"] is not None]
    print(f"\nn={len(valid)} volumes with all metrics")
    if len(valid) < 5:
        return 0

    views = [r["views"] for r in valid]
    vpy   = [r["vpy"] for r in valid]
    year  = [r["set_year"] for r in valid]
    age   = [r["age"] for r in valid]
    mi    = [r["med_instr_listeners"] for r in valid]
    ma    = [r["med_aca_listeners"] for r in valid]
    ci    = [r["instr_chart_rate"] for r in valid]
    ca    = [r["aca_chart_rate"] for r in valid]

    print("\n=== correlations ===")
    print(f"  pearson(views, set_year)              = {pearson(views, year):+.3f}   "
          f"(older volumes have had more time -> higher views; expect negative)")
    print(f"  pearson(views, age)                   = {pearson(views, age):+.3f}")
    print()
    print(f"  pearson(views, median_instr_listeners) = {pearson(views, mi):+.3f}")
    print(f"  partial controlling for set_year       = {partial_pearson(views, mi, year):+.3f}")
    print(f"  pearson(views_per_year, med_instr)     = {pearson(vpy, mi):+.3f}")
    print()
    print(f"  pearson(views, median_aca_listeners)   = {pearson(views, ma):+.3f}")
    print(f"  partial controlling for set_year       = {partial_pearson(views, ma, year):+.3f}")
    print(f"  pearson(views_per_year, med_aca)       = {pearson(vpy, ma):+.3f}")
    print()
    print(f"  pearson(views, instr_chart_rate)       = {pearson(views, ci):+.3f}")
    print(f"  partial controlling for set_year       = {partial_pearson(views, ci, year):+.3f}")
    print()
    print(f"  pearson(views, aca_chart_rate)         = {pearson(views, ca):+.3f}")
    print(f"  partial controlling for set_year       = {partial_pearson(views, ca, year):+.3f}")

    # Persist headline correlations to analysis_results
    cur = conn.cursor()
    findings = [
        ("pearson_views_set_year",         "all", pearson(views, year)),
        ("pearson_views_med_instr",        "all", pearson(views, mi)),
        ("partial_views_med_instr_setyr",  "all", partial_pearson(views, mi, year)),
        ("pearson_views_med_aca",          "all", pearson(views, ma)),
        ("partial_views_med_aca_setyr",    "all", partial_pearson(views, ma, year)),
        ("pearson_views_instr_chart_rate", "all", pearson(views, ci)),
        ("pearson_views_aca_chart_rate",   "all", pearson(views, ca)),
        ("partial_views_aca_chart_rate_setyr","all", partial_pearson(views, ca, year)),
        ("pearson_vpy_med_instr",          "all", pearson(vpy, mi)),
        ("pearson_vpy_med_aca",            "all", pearson(vpy, ma)),
        ("n_volumes",                       "all", float(len(valid))),
    ]
    for metric, group, val in findings:
        cur.execute("""
            INSERT INTO aux.analysis_results
              (analysis_name, metric, group_key, value)
            VALUES ('bb_set_views_v1', ?, ?, ?)
            ON CONFLICT(analysis_name, metric, group_key) DO UPDATE SET
              value = excluded.value, computed_at = CURRENT_TIMESTAMP
        """, (metric, group, val))
    conn.commit()
    print(f"\npersisted {len(findings)} headline metrics to aux.analysis_results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
