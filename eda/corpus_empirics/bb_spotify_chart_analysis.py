"""Compares Spotify Top 200 popularity signals against the Billboard signals
in `aux.db` for predicting BB acapella selection and BB set-views.

The Spotify Charts archive (charts.spotify.com, mirrored at kworb.net) covers
only ~Dec 2017 onward — so this analysis is restricted to tracks whose
release year fits that window. For older catalog acapellas (where most of
the layered vocals come from) Spotify gives no signal.

Three questions answered:

  1) For tracks in the Spotify-coverage window (2017+): does Spotify peak
     position predict views better than Billboard year-end / weekly Hot 100?

  2) Are there acapellas that hit Spotify Top 200 but were missed by both
     Billboard year-end AND weekly Hot 100? (i.e. the user's "biased proxy"
     concern revisited with a streaming-era signal)

  3) Cross-validation: does Spotify peak position correlate with Billboard
     peak position on the tracks where both exist?

Reads aux.db (must already have `track_spotify_charts` populated by
`scripts/aux_db_sync.py` after `scripts/bb_spotify_charts.py` writes the cache).
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path
from statistics import mean, median

AUX_DB = Path("data/analysis/aux.db")
MAIN_DB = Path("data/db/music_database.db")
OUT_JSON = Path("data/analysis/bb_spotify_chart.json")
TODAY_YEAR = 2026

VIEWS: dict[int, float] = {
    25: 3_140_000, 24: 3_470_000, 23: 4_180_000, 22: 4_750_000, 21: 5_980_000,
    20: 5_580_000, 19: 7_390_000, 18: 11_900_000, 17: 15_300_000,
    16: 7_820_000,  15: 20_000_000, 14: 11_700_000, 13: 13_000_000,
    12: 7_580_000,  11: 21_700_000, 10: 9_930_000, 9: 3_630_000,
    8: 2_610_000,   7: 1_090_000,   6: 1_320_000,  5: 939_000,
    4: 820_000,     3: 666_000,     1: 976_000,
}


def parse_volume(title: str) -> int | None:
    if "big bootie" not in title.lower(): return None
    if "@ big bootie" in title.lower(): return None
    for pat in [r"vol\.?\s*(\d+)\b", r"volume\s*(\d+)\b",
                r"mix\s+0?(\d+)\b", r"episode\s+(\d+)\b"]:
        m = re.search(pat, title.lower())
        if m: return int(m.group(1))
    return None


def pearson(xs, ys):
    n = len(xs)
    if n < 2: return float("nan")
    mx = sum(xs) / n; my = sum(ys) / n
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    dx = sum((x-mx)**2 for x in xs)**0.5
    dy = sum((y-my)**2 for y in ys)**0.5
    return num/(dx*dy) if dx and dy else float("nan")


def spearman(xs, ys):
    """Spearman rank correlation."""
    def rank(vs):
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        r = [0.0] * len(vs)
        i = 0
        while i < len(vs):
            j = i
            while j+1 < len(vs) and vs[order[j+1]] == vs[order[i]]:
                j += 1
            avg = (i+j)/2 + 1
            for k in range(i, j+1): r[order[k]] = avg
            i = j+1
        return r
    return pearson(rank(xs), rank(ys))


# ---------------- track-level ----------------
def pull_tracks(conn) -> list[dict]:
    return [dict(r) for r in conn.execute("""
    WITH bb_sets AS (SELECT set_id FROM dj_sets WHERE LOWER(title) LIKE '%big bootie%'),
         bb_rows AS (
           SELECT DISTINCT
                  json_extract(r.data_attrs_json,'$."data-trackid"') AS tid,
                  CASE WHEN r.text_excerpt LIKE 'w/%' THEN 'acapella'
                       WHEN r.text_excerpt GLOB '[0-9]*' THEN 'instrumental' END AS role
           FROM dj_set_rows r
           WHERE r.set_id IN (SELECT set_id FROM bb_sets))
    SELECT b.tid AS track_id, b.role,
           m.release_year, m.spotify_id, lf.listeners,
           ye.rank          AS yearend_rank,
           wk.peak_position AS weekly_peak,
           wk.weeks_on_chart AS weekly_woc,
           sp.status AS sp_status,
           sp.peak_global, sp.peak_us, sp.peak_gb,
           sp.streams_global, sp.streams_us,
           sp.n_countries_charted, sp.weeks_on_chart_global AS sp_weeks
    FROM bb_rows b
    LEFT JOIN aux.track_meta m ON m.track_id = b.tid
    LEFT JOIN aux.track_lastfm lf ON lf.track_id = b.tid
    LEFT JOIN aux.track_chart_match ye
      ON ye.track_id = b.tid AND ye.chart_name = 'billboard_hot100'
    LEFT JOIN aux.track_chart_match wk
      ON wk.track_id = b.tid AND wk.chart_name = 'billboard_hot100_weekly'
    LEFT JOIN aux.track_spotify_charts sp ON sp.spotify_id = m.spotify_id
    WHERE b.role IS NOT NULL
    """)]


def track_level(rows):
    aca = [r for r in rows if r["role"] == "acapella"]
    ins = [r for r in rows if r["role"] == "instrumental"]
    aca_with_sp = [r for r in aca if r["sp_status"] is not None]
    ins_with_sp = [r for r in ins if r["sp_status"] is not None]
    aca_modern = [r for r in aca if r["release_year"] and r["release_year"] >= 2015]
    ins_modern = [r for r in ins if r["release_year"] and r["release_year"] >= 2015]

    print(f"\n--- coverage ---")
    print(f"  acapellas total: {len(aca)}, with spotify_id: {len(aca_with_sp)}, "
          f"release_year ≥ 2015: {len(aca_modern)}")
    print(f"  instrumentals total: {len(ins)}, with spotify_id: {len(ins_with_sp)}, "
          f"release_year ≥ 2015: {len(ins_modern)}")

    def sp_status(tracks):
        out = {"charted": 0, "uncharted": 0, "no_data": 0, "error": 0}
        for t in tracks:
            s = t["sp_status"]
            if s == "charted": out["charted"] += 1
            elif s == "uncharted": out["uncharted"] += 1
            elif s == "error": out["error"] += 1
            else: out["no_data"] += 1
        return out

    print(f"\n--- Spotify status breakdown ---")
    print(f"  acapellas:     {sp_status(aca)}")
    print(f"  instrumentals: {sp_status(ins)}")
    print(f"  acapellas modern (≥2015):     {sp_status(aca_modern)}")
    print(f"  instrumentals modern (≥2015): {sp_status(ins_modern)}")

    # Peak distribution (Global, among acapellas with peak_global set)
    def peak_dist(tracks, field):
        xs = [t[field] for t in tracks if t.get(field) is not None]
        if not xs: return {"n": 0}
        return {
            "n": len(xs),
            "median": int(median(xs)),
            "mean": round(mean(xs), 1),
            "top1":   sum(1 for x in xs if x == 1),
            "top10":  sum(1 for x in xs if x <= 10),
            "top40":  sum(1 for x in xs if x <= 40),
            "top100": sum(1 for x in xs if x <= 100),
            "top200": len(xs),  # by definition, on the chart
        }

    print(f"\n--- Spotify Global peak distribution ---")
    print(f"  acapella:    {peak_dist(aca, 'peak_global')}")
    print(f"  instrumental:{peak_dist(ins, 'peak_global')}")
    print(f"\n--- Spotify US peak distribution ---")
    print(f"  acapella:    {peak_dist(aca, 'peak_us')}")
    print(f"  instrumental:{peak_dist(ins, 'peak_us')}")

    # Hit-rates by chart cut, restricted to tracks that COULD chart on Spotify
    # (have a spotify_id AND release_year ≥ 2015 so the Spotify window applies)
    def hit_rate_modern(tracks, pred):
        modern = [t for t in tracks if t["release_year"] and t["release_year"] >= 2015
                  and t["sp_status"] is not None]
        hits = [t for t in modern if pred(t)]
        return len(hits), len(modern), (len(hits) / len(modern) if modern else 0)

    rates = {}
    for name, pred in [
        ("yearend",        lambda t: t["yearend_rank"] is not None),
        ("hot100_weekly",  lambda t: t["weekly_peak"] is not None),
        ("hot100_top40",   lambda t: t["weekly_peak"] is not None and t["weekly_peak"] <= 40),
        ("hot100_top10",   lambda t: t["weekly_peak"] is not None and t["weekly_peak"] <= 10),
        ("spotify_global", lambda t: t["peak_global"] is not None),
        ("spotify_us",     lambda t: t["peak_us"] is not None),
        ("spotify_g_top40",lambda t: t["peak_global"] is not None and t["peak_global"] <= 40),
        ("spotify_g_top10",lambda t: t["peak_global"] is not None and t["peak_global"] <= 10),
    ]:
        a_h, a_n, a_r = hit_rate_modern(aca, pred)
        i_h, i_n, i_r = hit_rate_modern(ins, pred)
        rates[name] = {"aca": a_r, "ins": i_r, "n_aca": a_n, "n_ins": i_n,
                       "a_h": a_h, "i_h": i_h}

    print(f"\n--- modern-window (release_year ≥ 2015) hit rates ---")
    print(f"  {'cut':<18}  {'acap':>5}  {'instr':>6}  ratio")
    for name, r in rates.items():
        ratio = (r["aca"]/r["ins"]) if r["ins"] else float("inf")
        print(f"  {name:<18}  {r['aca']:5.3f}  {r['ins']:6.3f}  {ratio:>5.2f}x  "
              f"(aca {r['a_h']}/{r['n_aca']}, ins {r['i_h']}/{r['n_ins']})")

    # The "missed-by-Billboard-but-on-Spotify" bucket — answers the user's
    # bias concern directly
    print(f"\n--- acapellas charted on Spotify but missed by both Billboard signals ---")
    on_sp_not_bb = [
        t for t in aca
        if t["peak_global"] is not None
        and t["yearend_rank"] is None
        and t["weekly_peak"] is None
    ]
    print(f"  count: {len(on_sp_not_bb)}")
    if on_sp_not_bb:
        peaks = [t["peak_global"] for t in on_sp_not_bb]
        print(f"  global peak distribution: median={median(peaks)}, "
              f"top10={sum(1 for p in peaks if p<=10)}, "
              f"top40={sum(1 for p in peaks if p<=40)}, "
              f"top100={sum(1 for p in peaks if p<=100)}")
        with_lfm = [t for t in on_sp_not_bb if t["listeners"] is not None]
        hi = sum(1 for t in with_lfm if t["listeners"] >= 100_000)
        print(f"  with Last.fm listeners: {len(with_lfm)}, ≥100k: {hi} "
              f"({hi/len(with_lfm)*100:.0f}%)" if with_lfm else "  no Last.fm data")

    # Cross-validation: where Spotify global peak and Hot 100 weekly peak both exist,
    # do they agree?
    both = [t for t in aca + ins
            if t["peak_global"] is not None and t["weekly_peak"] is not None]
    if len(both) > 20:
        sp_peaks = [t["peak_global"] for t in both]
        bb_peaks = [t["weekly_peak"] for t in both]
        r = pearson(sp_peaks, bb_peaks)
        sr = spearman(sp_peaks, bb_peaks)
        print(f"\n--- cross-validation (n={len(both)} tracks on both charts) ---")
        print(f"  pearson(spotify_peak_global, hot100_weekly_peak) = {r:+.3f}")
        print(f"  spearman                                          = {sr:+.3f}")
        # Top-10 agreement
        sp_t10 = [t["peak_global"] <= 10 for t in both]
        bb_t10 = [t["weekly_peak"] <= 10 for t in both]
        agree_t10 = sum(1 for a, b in zip(sp_t10, bb_t10) if a == b)
        both_t10 = sum(1 for a, b in zip(sp_t10, bb_t10) if a and b)
        sp_only_t10 = sum(1 for a, b in zip(sp_t10, bb_t10) if a and not b)
        bb_only_t10 = sum(1 for a, b in zip(sp_t10, bb_t10) if b and not a)
        print(f"  top-10 agreement: {agree_t10}/{len(both)} "
              f"(both={both_t10}, sp-only={sp_only_t10}, bb-only={bb_only_t10})")

    return {
        "n_aca": len(aca), "n_ins": len(ins),
        "n_aca_with_spotify": len(aca_with_sp),
        "n_aca_modern": len(aca_modern),
        "sp_status_breakdown": {
            "acapella": sp_status(aca),
            "instrumental": sp_status(ins),
        },
        "peak_distribution": {
            "acapella_global": peak_dist(aca, "peak_global"),
            "instrumental_global": peak_dist(ins, "peak_global"),
            "acapella_us": peak_dist(aca, "peak_us"),
        },
        "modern_hit_rates": rates,
        "spotify_only_acapellas": {
            "count": len(on_sp_not_bb),
        },
    }


# ---------------- set-level ----------------
def set_level(conn):
    rows = conn.execute("""
        SELECT set_id, title, CAST(substr(date_played,1,4) AS INTEGER) AS set_year
        FROM dj_sets WHERE LOWER(title) LIKE '%big bootie%'
    """).fetchall()
    vol_to_set = {}
    for r in rows:
        v = parse_volume(r["title"])
        if v is None or v in vol_to_set: continue
        vol_to_set[v] = dict(r)

    per_vol = []
    for vol in sorted(VIEWS.keys()):
        s = vol_to_set.get(vol)
        if not s: continue
        track_rows = conn.execute("""
        WITH bb_rows AS (
          SELECT json_extract(r.data_attrs_json,'$."data-trackid"') AS tid,
                 CASE WHEN r.text_excerpt LIKE 'w/%' THEN 'acapella'
                      WHEN r.text_excerpt GLOB '[0-9]*' THEN 'instrumental' END AS role
          FROM dj_set_rows r WHERE r.set_id = ?)
        SELECT b.role, m.release_year, lf.listeners,
               ye.rank          AS yearend_rank,
               wk.peak_position AS weekly_peak,
               sp.peak_global, sp.peak_us, sp.weeks_on_chart_global AS sp_weeks
        FROM bb_rows b
        LEFT JOIN aux.track_meta m ON m.track_id = b.tid
        LEFT JOIN aux.track_lastfm lf ON lf.track_id = b.tid
        LEFT JOIN aux.track_chart_match ye
          ON ye.track_id = b.tid AND ye.chart_name = 'billboard_hot100'
        LEFT JOIN aux.track_chart_match wk
          ON wk.track_id = b.tid AND wk.chart_name = 'billboard_hot100_weekly'
        LEFT JOIN aux.track_spotify_charts sp ON sp.spotify_id = m.spotify_id
        WHERE b.role IS NOT NULL
        """, (s["set_id"],)).fetchall()

        aca = [r for r in track_rows if r["role"] == "acapella"]

        with_yr = [t for t in aca if t["release_year"]]
        if not with_yr:
            continue
        n = len(with_yr)
        sp_charted   = [t for t in with_yr if t["peak_global"] is not None]
        sp_g_top40   = [t for t in with_yr if t["peak_global"] is not None and t["peak_global"] <= 40]
        sp_g_top10   = [t for t in with_yr if t["peak_global"] is not None and t["peak_global"] <= 10]
        sp_us_top40  = [t for t in with_yr if t["peak_us"] is not None and t["peak_us"] <= 40]
        ye           = [t for t in with_yr if t["yearend_rank"] is not None]
        bb_t10       = [t for t in with_yr if t["weekly_peak"] is not None and t["weekly_peak"] <= 10]

        per_vol.append({
            "vol": vol, "set_id": s["set_id"], "set_year": s["set_year"],
            "views": VIEWS[vol],
            "n_aca": n,
            "sp_charted_rate":  len(sp_charted) / n,
            "sp_g_top40_rate":  len(sp_g_top40) / n,
            "sp_g_top10_rate":  len(sp_g_top10) / n,
            "sp_us_top40_rate": len(sp_us_top40) / n,
            "ye_rate":          len(ye) / n,
            "bb_t10_rate":      len(bb_t10) / n,
            "n_sp_g_top10":     len(sp_g_top10),
            "n_bb_t10":         len(bb_t10),
        })

    valid = [v for v in per_vol if v["n_aca"] >= 5]
    print(f"\nn={len(valid)} volumes for set-level regression")

    if not valid:
        return {"per_volume": [], "univariate": {}, "n": 0}

    views = [v["views"] for v in valid]
    feats = {
        "ye_rate":          [v["ye_rate"] for v in valid],
        "bb_t10_rate":      [v["bb_t10_rate"] for v in valid],
        "sp_charted_rate":  [v["sp_charted_rate"] for v in valid],
        "sp_g_top40_rate":  [v["sp_g_top40_rate"] for v in valid],
        "sp_g_top10_rate":  [v["sp_g_top10_rate"] for v in valid],
        "sp_us_top40_rate": [v["sp_us_top40_rate"] for v in valid],
    }
    print(f"\n--- univariate r with set views ---")
    print(f"  {'feature':<22}  {'r':>7}  {'r²':>6}")
    out_uni = {}
    for name, xs in feats.items():
        r = pearson(views, xs)
        out_uni[name] = {"r": round(r, 3), "r2": round(r*r, 3)}
        print(f"  {name:<22}  {r:+.3f}  {r*r:.3f}")

    # Multivariate fits
    def multireg_r2(ys, *cols):
        n = len(ys); k = len(cols)
        X = [[1.0] + [cols[j][i] for j in range(k)] for i in range(n)]
        XtX = [[sum(X[i][a]*X[i][b] for i in range(n)) for b in range(k+1)] for a in range(k+1)]
        Xty = [sum(X[i][a]*ys[i] for i in range(n)) for a in range(k+1)]
        def solve(A, b):
            m = len(A); M = [row[:] + [b[i]] for i,row in enumerate(A)]
            for i in range(m):
                pv = max(range(i, m), key=lambda r: abs(M[r][i]))
                M[i], M[pv] = M[pv], M[i]
                if abs(M[i][i]) < 1e-12: return None
                p = M[i][i]
                for j in range(i, m+1): M[i][j] /= p
                for r in range(m):
                    if r == i: continue
                    fac = M[r][i]
                    for j in range(i, m+1): M[r][j] -= fac * M[i][j]
            return [M[i][m] for i in range(m)]
        beta = solve(XtX, Xty)
        if beta is None: return float("nan")
        yhat = [sum(beta[a]*X[i][a] for a in range(k+1)) for i in range(n)]
        my = sum(ys)/n
        ss_res = sum((ys[i]-yhat[i])**2 for i in range(n))
        ss_tot = sum((y-my)**2 for y in ys)
        return 1 - ss_res/ss_tot if ss_tot else float("nan")

    fits = {
        "sp_g_top10 only":                  [feats["sp_g_top10_rate"]],
        "bb_t10 only":                      [feats["bb_t10_rate"]],
        "ye only":                          [feats["ye_rate"]],
        "sp_g_top10 + bb_t10":              [feats["sp_g_top10_rate"], feats["bb_t10_rate"]],
        "sp_g_top10 + ye":                  [feats["sp_g_top10_rate"], feats["ye_rate"]],
        "all top-tier":                     [feats["sp_g_top10_rate"], feats["bb_t10_rate"], feats["ye_rate"]],
    }
    print(f"\n--- multivariate R² (set views vs acapella-side features) ---")
    out_multi = {}
    for name, cols in fits.items():
        r2 = multireg_r2(views, *cols)
        out_multi[name] = round(r2, 3)
        print(f"  {name:<32}  R² = {r2:.3f}")

    return {
        "per_volume": [{"vol": v["vol"], "year": v["set_year"], "views": v["views"],
                        "sp_g_top10_rate": v["sp_g_top10_rate"],
                        "bb_t10_rate": v["bb_t10_rate"],
                        "ye_rate": v["ye_rate"]}
                       for v in valid],
        "univariate": out_uni,
        "multivariate_r2": out_multi,
        "n": len(valid),
    }


# ---------------- main ----------------
def main():
    if not AUX_DB.exists() or not MAIN_DB.exists():
        print("missing aux.db or main DB", file=sys.stderr); return 1
    conn = sqlite3.connect(MAIN_DB)
    conn.row_factory = sqlite3.Row
    conn.execute(f"ATTACH DATABASE '{AUX_DB}' AS aux")

    n_sp = conn.execute("SELECT COUNT(*) FROM aux.track_spotify_charts").fetchone()[0]
    if n_sp == 0:
        print("track_spotify_charts is empty — run scripts/bb_spotify_charts.py "
              "and scripts/aux_db_sync.py first", file=sys.stderr)
        return 1
    print(f"=== track_spotify_charts: {n_sp} rows ===")

    print("\n=== track-level ===")
    rows = pull_tracks(conn)
    track_results = track_level(rows)

    print("\n=== set-level ===")
    set_results = set_level(conn)

    out = {"track_level": track_results, "set_level": set_results}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT_JSON}")

    # Persist headline metrics
    cur = conn.cursor()
    headline = []
    for name, r in track_results["modern_hit_rates"].items():
        headline.append((f"modern_hit_rate_{name}", "acapella", float(r["aca"])))
        headline.append((f"modern_hit_rate_{name}", "instrumental", float(r["ins"])))
    for feat, d in set_results.get("univariate", {}).items():
        headline.append((f"pearson_views_{feat}", "all", float(d["r"])))
    for fit, r2 in set_results.get("multivariate_r2", {}).items():
        headline.append((f"multi_r2_{fit.replace(' ', '_')}", "all", float(r2)))
    for metric, group, val in headline:
        cur.execute("""
            INSERT INTO aux.analysis_results
              (analysis_name, metric, group_key, value)
            VALUES ('bb_spotify_chart_v1', ?, ?, ?)
            ON CONFLICT(analysis_name, metric, group_key) DO UPDATE SET
              value = excluded.value, computed_at = CURRENT_TIMESTAMP
        """, (metric, group, val))
    conn.commit()
    print(f"persisted {len(headline)} metrics to aux.analysis_results (bb_spotify_chart_v1)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
