"""Re-runs the BB popularity findings using the weekly Hot 100 (all-time) signals.

The original [bb_popularity.py](bb_popularity.py) only used Billboard
Year-End Hot 100 — a narrow definition of "chart hit" that requires sustained
chart presence over a calendar year. This script also looks at:

  - **ever charted (weekly)** — broader: did the song ever spend a week on
    Hot 100 at any peak position (1-100)?
  - **peak position** — continuous signal, 1 = #1 ever, 100 = grazed #100
    for a single week.
  - **weeks_on_chart** — duration on Hot 100; cultural-moment longevity.

Re-does two analyses with these features:

  (1) Track-level acap-vs-instr comparison: hit-rate by broader chart definition,
      peak-position distributions, four-quadrant split using weekly-or-not.

  (2) Set-level (per-BB-volume) views regression: do these broader / continuous
      chart features predict YouTube views better than year-end-binary alone?

Reads aux.db (must already be populated by `eda/corpus_empirics/aux_db_sync.py`).
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

AUX_DB = Path("data/analysis/aux.db")
MAIN_DB = Path("data/db/music_database.db")
OUT_JSON = Path("data/analysis/bb_weekly_chart.json")
TODAY_YEAR = 2026

# Reuse the volume-views map from bb_set_views_analysis.py (single source of truth).
VIEWS: dict[int, float] = {
    25: 3_140_000, 24: 3_470_000, 23: 4_180_000, 22: 4_750_000, 21: 5_980_000,
    20: 5_580_000, 19: 7_390_000, 18: 11_900_000, 17: 15_300_000,
    16: 7_820_000,  15: 20_000_000, 14: 11_700_000, 13: 13_000_000,
    12: 7_580_000,  11: 21_700_000, 10: 9_930_000, 9: 3_630_000,
    8: 2_610_000,   7: 1_090_000,   6: 1_320_000,  5: 939_000,
    4: 820_000,     3: 666_000,     1: 976_000,
}


def parse_volume(title: str) -> int | None:
    if "big bootie" not in title.lower():
        return None
    if "@ big bootie" in title.lower():
        return None
    for pat in [r"vol\.?\s*(\d+)\b", r"volume\s*(\d+)\b",
                r"mix\s+0?(\d+)\b", r"episode\s+(\d+)\b"]:
        m = re.search(pat, title.lower())
        if m:
            return int(m.group(1))
    return None


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
    dy = (sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def partial_pearson(xs, ys, zs):
    """Pearson(x,y) controlling for z."""
    def resid(vs, ctrl):
        mc = sum(ctrl) / len(ctrl)
        mv = sum(vs) / len(vs)
        num = sum((c - mc) * (v - mv) for c, v in zip(ctrl, vs))
        den = sum((c - mc) ** 2 for c in ctrl)
        b = num / den if den else 0.0
        a = mv - b * mc
        return [v - (a + b * c) for c, v in zip(ctrl, vs)]
    return pearson(resid(xs, zs), resid(ys, zs))


def pct(xs: list, p: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(p * len(s)))]


# ---------------- track-level pull ----------------
def pull_track_signals(conn: sqlite3.Connection) -> list[dict]:
    """One row per (BB track, role) with all popularity signals attached."""
    q = """
    WITH bb_sets AS (
      SELECT set_id FROM dj_sets WHERE LOWER(title) LIKE '%big bootie%'),
    bb_rows AS (
      SELECT DISTINCT
             json_extract(r.data_attrs_json, '$."data-trackid"') AS track_id,
             CASE WHEN r.text_excerpt LIKE 'w/%' THEN 'acapella'
                  WHEN r.text_excerpt GLOB '[0-9]*' THEN 'instrumental' END AS role
      FROM dj_set_rows r
      WHERE r.set_id IN (SELECT set_id FROM bb_sets))
    SELECT b.track_id, b.role,
           m.release_year,
           lf.listeners,
           ye.rank          AS yearend_rank,
           wk.peak_position AS weekly_peak,
           wk.weeks_on_chart AS weekly_woc
    FROM bb_rows b
    LEFT JOIN aux.track_meta m ON m.track_id = b.track_id
    LEFT JOIN aux.track_lastfm lf ON lf.track_id = b.track_id
    LEFT JOIN aux.track_chart_match ye
      ON ye.track_id = b.track_id AND ye.chart_name = 'billboard_hot100'
    LEFT JOIN aux.track_chart_match wk
      ON wk.track_id = b.track_id AND wk.chart_name = 'billboard_hot100_weekly'
    WHERE b.role IS NOT NULL AND b.track_id IS NOT NULL
    """
    rows = [dict(r) for r in conn.execute(q)]
    return rows


# ---------------- track-level analysis ----------------
def track_level(rows: list[dict]) -> dict:
    aca = [r for r in rows if r["role"] == "acapella"]
    ins = [r for r in rows if r["role"] == "instrumental"]

    def rate(tracks: list[dict], pred) -> tuple[int, int, float]:
        with_year = [t for t in tracks if t["release_year"]]
        hits = [t for t in with_year if pred(t)]
        return len(hits), len(with_year), (len(hits) / len(with_year) if with_year else 0.0)

    def peak_dist(tracks: list[dict]) -> dict:
        xs = [t["weekly_peak"] for t in tracks if t["weekly_peak"] is not None]
        if not xs:
            return {"n": 0}
        return {
            "n": len(xs),
            "median": int(median(xs)),
            "mean": round(mean(xs), 1),
            "p10": int(pct(xs, 0.10)),
            "p25": int(pct(xs, 0.25)),
            "p75": int(pct(xs, 0.75)),
            "p90": int(pct(xs, 0.90)),
            "top1_count": sum(1 for x in xs if x == 1),
            "top10_count": sum(1 for x in xs if x <= 10),
            "top40_count": sum(1 for x in xs if x <= 40),
        }

    def woc_dist(tracks: list[dict]) -> dict:
        xs = [t["weekly_woc"] for t in tracks if t["weekly_woc"] is not None]
        if not xs:
            return {"n": 0}
        return {
            "n": len(xs),
            "median": int(median(xs)),
            "mean": round(mean(xs), 1),
            "max": int(max(xs)),
        }

    is_yearend = lambda t: t["yearend_rank"] is not None
    is_weekly  = lambda t: t["weekly_peak"] is not None
    is_top40   = lambda t: t["weekly_peak"] is not None and t["weekly_peak"] <= 40
    is_top10   = lambda t: t["weekly_peak"] is not None and t["weekly_peak"] <= 10

    aca_ye_c, aca_ye_n, aca_ye_r = rate(aca, is_yearend)
    ins_ye_c, ins_ye_n, ins_ye_r = rate(ins, is_yearend)
    aca_wk_c, aca_wk_n, aca_wk_r = rate(aca, is_weekly)
    ins_wk_c, ins_wk_n, ins_wk_r = rate(ins, is_weekly)
    aca_t40_c, _, aca_t40_r = rate(aca, is_top40)
    ins_t40_c, _, ins_t40_r = rate(ins, is_top40)
    aca_t10_c, _, aca_t10_r = rate(aca, is_top10)
    ins_t10_c, _, ins_t10_r = rate(ins, is_top10)

    # Four-quadrant: weekly-charted × ≥100k Last.fm
    def quad(t):
        if t["listeners"] is None:
            return None
        hi = t["listeners"] >= 100_000
        ch = t["weekly_peak"] is not None
        if ch and hi: return "hit_remembered"
        if ch and not hi: return "hit_forgotten"
        if not ch and hi: return "deepcut_remembered"
        return "deepcut_obscure"

    def quad_pct(tracks):
        out = defaultdict(int); total = 0
        for t in tracks:
            q = quad(t)
            if q: out[q] += 1; total += 1
        return {k: round(v / total, 3) if total else 0 for k, v in out.items()} | {"_n": total}

    # The interesting bucket: tracks that charted weekly but NOT year-end —
    # "really popular but missed by Hot 100 year-end". Same Last.fm split applied.
    def weekly_only(tracks):
        ch_wk_not_ye = [t for t in tracks if t["weekly_peak"] is not None and t["yearend_rank"] is None]
        with_lfm = [t for t in ch_wk_not_ye if t["listeners"] is not None]
        hi = [t for t in with_lfm if t["listeners"] >= 100_000]
        peaks = [t["weekly_peak"] for t in ch_wk_not_ye]
        return {
            "count": len(ch_wk_not_ye),
            "with_listeners": len(with_lfm),
            "high_listeners_rate": round(len(hi) / len(with_lfm), 3) if with_lfm else 0,
            "peak_median": int(median(peaks)) if peaks else None,
            "peak_distribution": {
                "top10": sum(1 for p in peaks if p <= 10),
                "11-40": sum(1 for p in peaks if 10 < p <= 40),
                "41-100": sum(1 for p in peaks if p > 40),
            },
        }

    return {
        "n_aca": len(aca), "n_ins": len(ins),
        "hit_rate": {
            "yearend": {
                "acapella": {"charted": aca_ye_c, "n": aca_ye_n, "rate": round(aca_ye_r, 3)},
                "instrumental": {"charted": ins_ye_c, "n": ins_ye_n, "rate": round(ins_ye_r, 3)},
            },
            "weekly_ever": {
                "acapella": {"charted": aca_wk_c, "n": aca_wk_n, "rate": round(aca_wk_r, 3)},
                "instrumental": {"charted": ins_wk_c, "n": ins_wk_n, "rate": round(ins_wk_r, 3)},
            },
            "top40": {
                "acapella": {"charted": aca_t40_c, "rate": round(aca_t40_r, 3)},
                "instrumental": {"charted": ins_t40_c, "rate": round(ins_t40_r, 3)},
            },
            "top10": {
                "acapella": {"charted": aca_t10_c, "rate": round(aca_t10_r, 3)},
                "instrumental": {"charted": ins_t10_c, "rate": round(ins_t10_r, 3)},
            },
        },
        "peak_position_dist": {
            "acapella": peak_dist(aca),
            "instrumental": peak_dist(ins),
        },
        "weeks_on_chart_dist": {
            "acapella": woc_dist(aca),
            "instrumental": woc_dist(ins),
        },
        "four_quadrant_weekly": {
            "acapella": quad_pct(aca),
            "instrumental": quad_pct(ins),
        },
        "weekly_charted_not_yearend": {
            "acapella": weekly_only(aca),
            "instrumental": weekly_only(ins),
        },
    }


# ---------------- set-level analysis ----------------
def set_level(conn: sqlite3.Connection) -> dict:
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

    per_vol: list[dict] = []
    for vol in sorted(VIEWS.keys()):
        s = vol_to_set.get(vol)
        if not s:
            continue
        track_rows = conn.execute("""
        WITH bb_rows AS (
          SELECT json_extract(r.data_attrs_json,'$."data-trackid"') AS tid,
                 CASE WHEN r.text_excerpt LIKE 'w/%' THEN 'acapella'
                      WHEN r.text_excerpt GLOB '[0-9]*' THEN 'instrumental' END AS role
          FROM dj_set_rows r WHERE r.set_id = ?)
        SELECT b.role, m.release_year, lf.listeners,
               ye.rank          AS yearend_rank,
               wk.peak_position AS weekly_peak,
               wk.weeks_on_chart AS weekly_woc
        FROM bb_rows b
        LEFT JOIN aux.track_meta m ON m.track_id = b.tid
        LEFT JOIN aux.track_lastfm lf ON lf.track_id = b.tid
        LEFT JOIN aux.track_chart_match ye
          ON ye.track_id = b.tid AND ye.chart_name = 'billboard_hot100'
        LEFT JOIN aux.track_chart_match wk
          ON wk.track_id = b.tid AND wk.chart_name = 'billboard_hot100_weekly'
        WHERE b.role IS NOT NULL
        """, (s["set_id"],)).fetchall()

        def role(name):
            return [r for r in track_rows if r["role"] == name]
        aca = role("acapella"); ins = role("instrumental")

        def agg(tracks):
            with_yr = [t for t in tracks if t["release_year"]]
            ye = [t for t in with_yr if t["yearend_rank"] is not None]
            wk = [t for t in with_yr if t["weekly_peak"] is not None]
            t40 = [t for t in with_yr if t["weekly_peak"] is not None and t["weekly_peak"] <= 40]
            t10 = [t for t in with_yr if t["weekly_peak"] is not None and t["weekly_peak"] <= 10]
            peaks = [t["weekly_peak"] for t in wk]
            wocs = [t["weekly_woc"] for t in wk]
            n = len(with_yr)
            return {
                "n": n,
                "n_ye": len(ye), "ye_rate": len(ye) / n if n else 0,
                "n_wk": len(wk), "wk_rate": len(wk) / n if n else 0,
                "n_t40": len(t40), "t40_rate": len(t40) / n if n else 0,
                "n_t10": len(t10), "t10_rate": len(t10) / n if n else 0,
                "mean_peak": (sum(peaks) / len(peaks)) if peaks else None,
                "mean_woc": (sum(wocs) / len(wocs)) if wocs else None,
            }

        per_vol.append({
            "vol": vol, "set_id": s["set_id"], "set_year": s["set_year"],
            "views": VIEWS[vol],
            "aca": agg(aca), "ins": agg(ins),
        })

    # Print per-volume table
    print("\nper-volume aggregates (acapella side):")
    print(f"  {'vol':>3} {'yr':>4} {'views':>9}  {'n_aca':>5} {'ye':>3} {'wk':>3} "
          f"{'t40':>3} {'t10':>3}  {'wk%':>5} {'t40%':>5} {'mean_peak':>9} {'mean_woc':>8}")
    for v in per_vol:
        a = v["aca"]
        print(f"  {v['vol']:>3} {v['set_year']:>4} {v['views']:>9,.0f}  "
              f"{a['n']:>5} {a['n_ye']:>3} {a['n_wk']:>3} {a['n_t40']:>3} {a['n_t10']:>3}  "
              f"{a['wk_rate']:>5.2f} {a['t40_rate']:>5.2f}  "
              f"{(a['mean_peak'] or 0):>9.1f} {(a['mean_woc'] or 0):>8.1f}")

    # Regressions: filter to volumes with enough acapella signal
    valid = [v for v in per_vol if v["aca"]["n"] >= 5 and v["set_year"] is not None]
    print(f"\nn={len(valid)} volumes in regression")

    def vec(key_path):
        out = []
        for v in valid:
            cur = v
            for k in key_path.split("."):
                cur = cur[k] if cur is not None else None
            out.append(cur)
        return out

    views = vec("views")
    year  = vec("set_year")

    feats = {
        # Original signals (for comparison)
        "aca_yearend_rate":     vec("aca.ye_rate"),
        "n_aca_yearend":        vec("aca.n_ye"),
        # New: weekly broader signal
        "aca_weekly_rate":      vec("aca.wk_rate"),
        "n_aca_weekly":         vec("aca.n_wk"),
        # New: peak-position tiers
        "aca_top40_rate":       vec("aca.t40_rate"),
        "n_aca_top40":          vec("aca.n_t40"),
        "aca_top10_rate":       vec("aca.t10_rate"),
        "n_aca_top10":          vec("aca.n_t10"),
        # New: continuous popularity intensity
        "mean_aca_peak":        [v if v is not None else 100 for v in vec("aca.mean_peak")],
        "mean_aca_woc":         [v if v is not None else 0   for v in vec("aca.mean_woc")],
        # Instrumental side (for comparison — we expect these to remain near zero)
        "ins_weekly_rate":      vec("ins.wk_rate"),
        "n_ins_weekly":         vec("ins.n_wk"),
        "ins_top40_rate":       vec("ins.t40_rate"),
    }

    rows_out = {}
    print("\n=== univariate correlations with set views ===")
    print(f"  {'feature':<22}  {'r':>7}  {'r²':>6}  partial_r|set_year")
    for name, xs in feats.items():
        r = pearson(views, xs)
        pr = partial_pearson(views, xs, year)
        rows_out[name] = {"r": round(r, 3), "r2": round(r*r, 3),
                          "partial_r_setyr": round(pr, 3)}
        print(f"  {name:<22}  {r:+.3f}  {r*r:.3f}   {pr:+.3f}")

    # Compare year-end vs weekly directly
    print("\n=== year-end vs weekly head-to-head ===")
    for pair_name, ye_feat, wk_feat in [
        ("rate    ", "aca_yearend_rate", "aca_weekly_rate"),
        ("count   ", "n_aca_yearend",    "n_aca_weekly"),
    ]:
        r_ye = pearson(views, feats[ye_feat])
        r_wk = pearson(views, feats[wk_feat])
        delta = abs(r_wk) - abs(r_ye)
        print(f"  {pair_name}  year-end r={r_ye:+.3f}   weekly r={r_wk:+.3f}   "
              f"|Δ|={delta:+.3f}  ({'weekly wins' if delta > 0.02 else 'year-end wins' if delta < -0.02 else 'tie'})")

    # Multivariate-ish: do peak-position / weeks-on-chart add explanatory power
    # beyond simple weekly-rate? OLS via numpy would be cleaner but stays pure-Python.
    def multireg_r2(ys, *xs_cols):
        """OLS R² with intercept via normal equations."""
        n = len(ys)
        k = len(xs_cols)
        # Build augmented X with intercept
        X = [[1.0] + [xs_cols[j][i] for j in range(k)] for i in range(n)]
        # X^T X
        XtX = [[sum(X[i][a]*X[i][b] for i in range(n)) for b in range(k+1)] for a in range(k+1)]
        Xty = [sum(X[i][a]*ys[i] for i in range(n)) for a in range(k+1)]
        # Gauss-Jordan inverse-multiply
        def solve(A, b):
            m = len(A)
            M = [row[:] + [b[i]] for i, row in enumerate(A)]
            for i in range(m):
                pv = max(range(i, m), key=lambda r: abs(M[r][i]))
                M[i], M[pv] = M[pv], M[i]
                if abs(M[i][i]) < 1e-12:
                    return None
                p = M[i][i]
                for j in range(i, m+1): M[i][j] /= p
                for r in range(m):
                    if r == i: continue
                    fac = M[r][i]
                    for j in range(i, m+1): M[r][j] -= fac * M[i][j]
            return [M[i][m] for i in range(m)]
        beta = solve(XtX, Xty)
        if beta is None:
            return float("nan")
        yhat = [sum(beta[a]*X[i][a] for a in range(k+1)) for i in range(n)]
        my = sum(ys)/n
        ss_res = sum((ys[i]-yhat[i])**2 for i in range(n))
        ss_tot = sum((y-my)**2 for y in ys)
        return 1 - ss_res/ss_tot if ss_tot > 0 else float("nan")

    print("\n=== multivariate R² with set views (acapella-side features) ===")
    multifits = {
        "weekly_rate":                 [feats["aca_weekly_rate"]],
        "yearend_rate":                [feats["aca_yearend_rate"]],
        "weekly_rate + n_weekly":      [feats["aca_weekly_rate"], feats["n_aca_weekly"]],
        "yearend_rate + n_yearend":    [feats["aca_yearend_rate"], feats["n_aca_yearend"]],
        "weekly_rate + mean_peak":     [feats["aca_weekly_rate"], feats["mean_aca_peak"]],
        "weekly_rate + mean_woc":      [feats["aca_weekly_rate"], feats["mean_aca_woc"]],
        "top10_rate + top40_rate":     [feats["aca_top10_rate"], feats["aca_top40_rate"]],
        "weekly + top10 + mean_woc":   [feats["aca_weekly_rate"], feats["aca_top10_rate"], feats["mean_aca_woc"]],
        "yearend + weekly + mean_woc": [feats["aca_yearend_rate"], feats["aca_weekly_rate"], feats["mean_aca_woc"]],
    }
    multifit_results = {}
    for name, cols in multifits.items():
        r2 = multireg_r2(views, *cols)
        multifit_results[name] = round(r2, 3)
        print(f"  {name:<32}  R² = {r2:.3f}")

    return {
        "n_volumes": len(valid),
        "univariate": rows_out,
        "multivariate_r2": multifit_results,
        "per_volume": [{"vol": v["vol"], "year": v["set_year"], "views": v["views"],
                        "aca_ye_rate": v["aca"]["ye_rate"],
                        "aca_wk_rate": v["aca"]["wk_rate"],
                        "aca_t40_rate": v["aca"]["t40_rate"],
                        "aca_mean_peak": v["aca"]["mean_peak"]}
                       for v in valid],
    }


# ---------------- main ----------------
def main() -> int:
    if not AUX_DB.exists() or not MAIN_DB.exists():
        print("missing aux.db or main DB", file=sys.stderr)
        return 1

    conn = sqlite3.connect(MAIN_DB)
    conn.row_factory = sqlite3.Row
    conn.execute(f"ATTACH DATABASE '{AUX_DB}' AS aux")

    print("=== track-level analysis ===")
    rows = pull_track_signals(conn)
    print(f"  {len(rows)} BB track-role rows ({sum(1 for r in rows if r['role']=='acapella')} acap, "
          f"{sum(1 for r in rows if r['role']=='instrumental')} instr)")
    track_results = track_level(rows)
    print("\n--- track-level hit-rate by chart definition ---")
    for cdef in ("yearend", "weekly_ever", "top40", "top10"):
        h = track_results["hit_rate"][cdef]
        a = h["acapella"]; i = h["instrumental"]
        ratio = (a["rate"] / i["rate"]) if i["rate"] else float("inf")
        print(f"  {cdef:<14}  acap {a['rate']:.3f}  instr {i['rate']:.3f}  ratio {ratio:.2f}x")

    print("\n--- weekly-charted-but-not-yearend (the user's 'missed by Hot 100' bucket) ---")
    wo = track_results["weekly_charted_not_yearend"]["acapella"]
    print(f"  acapellas:    {wo['count']} tracks, peak distribution: {wo['peak_distribution']}")
    print(f"  high-Last.fm-listeners rate among them: {wo['high_listeners_rate']:.0%}")
    wo_i = track_results["weekly_charted_not_yearend"]["instrumental"]
    print(f"  instrumentals: {wo_i['count']} tracks  (sanity check)")

    print("\n=== set-level analysis ===")
    set_results = set_level(conn)

    out = {"track_level": track_results, "set_level": set_results}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT_JSON}")

    # Persist headline values to aux.analysis_results so other readers can join
    conn.execute("""
        CREATE TABLE IF NOT EXISTS aux.analysis_results (
          analysis_name TEXT NOT NULL,
          metric        TEXT NOT NULL,
          group_key     TEXT NOT NULL,
          value         REAL,
          unit          TEXT,
          notes         TEXT,
          computed_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY (analysis_name, metric, group_key)
        )
    """)
    cur = conn.cursor()
    headline = []
    for cdef in ("yearend", "weekly_ever", "top40", "top10"):
        for role in ("acapella", "instrumental"):
            r = track_results["hit_rate"][cdef][role]
            headline.append((f"hit_rate_{cdef}", role, float(r["rate"])))
    for feat, d in set_results["univariate"].items():
        headline.append((f"pearson_views_{feat}", "all", float(d["r"])))
    for fit, r2 in set_results["multivariate_r2"].items():
        headline.append((f"multi_r2_{fit.replace(' ', '_')}", "all", float(r2)))
    for metric, group, val in headline:
        cur.execute("""
            INSERT INTO aux.analysis_results
              (analysis_name, metric, group_key, value)
            VALUES ('bb_weekly_chart_v1', ?, ?, ?)
            ON CONFLICT(analysis_name, metric, group_key) DO UPDATE SET
              value = excluded.value, computed_at = CURRENT_TIMESTAMP
        """, (metric, group, val))
    conn.commit()
    print(f"persisted {len(headline)} headline metrics to aux.analysis_results (bb_weekly_chart_v1)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
