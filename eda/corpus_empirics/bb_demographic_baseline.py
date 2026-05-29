"""H2 — Cohort × Han-sensitivity demographic baseline for BB set views.

For each BB volume released in year Y, predict views as a sum over birth
cohorts b:

  V_expected(Y) = Σ_b pop_US(b) · Han(Y - b) · watch_years(b, Y, today)

where Han(Y - b) is the Bigaussian sensitivity curve fitted by Han et al.
(2022) to NCM users, evaluated at the listener's age when the volume
released. watch_years counts the overlap between the listener's active
age range [12, 35] and the volume's availability window [Y, today].

This is a music-free, structural baseline. Residuals from it isolate the
"music quality" component — the part of views NOT explained by demographic
mechanics. Then we ask: do our music features (chart density, format
maturity) predict residual_views?

If yes → music features explain genuine quality. If no → the apparent music
signal was a demographic artifact.

Persists results to aux.analysis_results under analysis_name='bb_demographic_baseline_v1'.
"""

from __future__ import annotations

import sqlite3
import sys
from math import exp, log
from collections import defaultdict

DB = "data/db/music_database.db"
AUX = "data/analysis/aux.db"
TODAY = 2026

# Han et al. 2022 Bigaussian fit (all users)
XC, Y0, H, W1, W2 = 12.88, 0.43, 0.87, 13.18, 7.26

# US births per year in thousands (CDC NCHS approx; close enough for BB's
# western-leaning audience). Values rounded to 4 sig figs.
US_BIRTHS = {
    1980: 3612, 1981: 3629, 1982: 3681, 1983: 3638, 1984: 3669,
    1985: 3761, 1986: 3757, 1987: 3809, 1988: 3910, 1989: 4041,
    1990: 4158, 1991: 4111, 1992: 4065, 1993: 4000, 1994: 3953,
    1995: 3900, 1996: 3891, 1997: 3881, 1998: 3942, 1999: 3959,
    2000: 4059, 2001: 4026, 2002: 4022, 2003: 4090, 2004: 4112,
    2005: 4138, 2006: 4266, 2007: 4317, 2008: 4248, 2009: 4131,
    2010: 3999, 2011: 3954, 2012: 3953, 2013: 3932, 2014: 3988,
    2015: 3978, 2016: 3946, 2017: 3855, 2018: 3792, 2019: 3748,
    2020: 3605, 2021: 3664, 2022: 3667, 2023: 3591, 2024: 3600,
    2025: 3550,
}

AGE_MIN, AGE_MAX = 12, 35  # active "music sensitivity" age window


def han(x: float) -> float:
    w = W1 if x < XC else W2
    return Y0 + H * exp(-0.5 * ((x - XC) / w) ** 2)


def expected_demand(Y_set: int, today: int = TODAY,
                    age_min: int = AGE_MIN, age_max: int = AGE_MAX) -> float:
    """Convolution of birth cohorts with Han sensitivity, weighted by
    each cohort's watch-window overlap with [Y_set, today].

    Returns a number in arbitrary units (proportional to expected views
    under the structural model). Use ratios / log differences, not
    absolute magnitudes."""
    total = 0.0
    for b, pop in US_BIRTHS.items():
        active_start = b + age_min
        active_end = b + age_max
        watch_start = max(Y_set, active_start)
        watch_end = min(today, active_end)
        years_watching = max(0.0, watch_end - watch_start)
        if years_watching <= 0:
            continue
        sensitivity = han(Y_set - b)
        total += pop * sensitivity * years_watching
    return total


def pearson(xs, ys):
    n = len(xs)
    if n < 2: return float("nan")
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    dx = (sum((x-mx)**2 for x in xs))**0.5
    dy = (sum((y-my)**2 for y in ys))**0.5
    return num/(dx*dy) if dx and dy else float("nan")


def fit_ols(y, X):
    n, k = len(y), len(X)
    cols = [[1.0]*n] + X
    XtX = [[sum(cols[i][r]*cols[j][r] for r in range(n)) for j in range(k+1)] for i in range(k+1)]
    Xty = [sum(cols[i][r]*y[r] for r in range(n)) for i in range(k+1)]
    A = [row[:]+[Xty[i]] for i, row in enumerate(XtX)]
    m = k+1
    for i in range(m):
        p = max(range(i, m), key=lambda r: abs(A[r][i]))
        A[i], A[p] = A[p], A[i]
        d = A[i][i]
        if abs(d) < 1e-12: return None, None, None
        for j in range(i, m+1): A[i][j] /= d
        for r in range(m):
            if r == i: continue
            f = A[r][i]
            for j in range(i, m+1): A[r][j] -= f*A[i][j]
    b = [A[i][m] for i in range(m)]
    yhat = [b[0] + sum(b[j+1]*X[j][r] for j in range(k)) for r in range(n)]
    sse = sum((y[r]-yhat[r])**2 for r in range(n))
    my = sum(y)/n
    sst = sum((y[r]-my)**2 for r in range(n))
    return b, 1 - sse/sst, [y[r]-yhat[r] for r in range(n)]


def cen(xs):
    m = sum(xs)/len(xs)
    return [x-m for x in xs]


def main() -> int:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    conn.execute(f"ATTACH DATABASE '{AUX}' AS aux")

    rows = conn.execute("""
    WITH bb AS (
      SELECT set_id, title, CAST(substr(date_played,1,4) AS INTEGER) AS set_year
      FROM dj_sets WHERE LOWER(title) LIKE '%big bootie%'
        AND LOWER(title) NOT LIKE '%@ big bootie%'),
    rws AS (
      SELECT r.set_id, json_extract(r.data_attrs_json,'$."data-trackid"') AS tid,
             CASE WHEN r.text_excerpt LIKE 'w/%' THEN 'aca'
                  WHEN r.text_excerpt GLOB '[0-9]*' THEN 'ins' END AS role
      FROM dj_set_rows r WHERE r.set_id IN (SELECT set_id FROM bb))
    SELECT b.set_id, b.title, b.set_year,
      SUM(CASE WHEN rw.role='aca' THEN 1 ELSE 0 END) AS n_aca,
      SUM(CASE WHEN rw.role='aca' AND cm.rank IS NOT NULL THEN 1 ELSE 0 END) AS n_aca_charted,
      sv.view_count
    FROM bb b
    JOIN rws rw ON rw.set_id=b.set_id
    LEFT JOIN aux.track_chart_match cm ON cm.track_id=rw.tid
    JOIN aux.set_views sv ON sv.set_id=b.set_id AND sv.platform='youtube'
    GROUP BY b.set_id
    ORDER BY b.set_year
    """).fetchall()

    print(f"n={len(rows)} BB volumes\n")
    print("=== Demographic baseline (cohort × Han-sensitivity × watch-window) ===\n")
    print(f"  {'title':45s} year  views        expected_demand    log-residual")
    print(f"  {'-'*45}  ----  -----------  -----------------  ------------")

    vols = []
    for r in rows:
        Y = r["set_year"]
        ed = expected_demand(Y)
        v = r["view_count"]
        log_resid = log(v) - log(ed)
        vols.append({
            "title": r["title"], "set_year": Y, "views": v,
            "exp_demand": ed, "log_resid": log_resid,
            "n_aca": r["n_aca"], "n_aca_charted": r["n_aca_charted"],
            "chart_rate": (r["n_aca_charted"]/r["n_aca"] if r["n_aca"] else 0),
        })

    for d in vols:
        print(f"  {d['title'][:43]:45s} {d['set_year']}  {d['views']:>11,d}  "
              f"{d['exp_demand']:>17.2e}  {d['log_resid']:+.2f}")

    # 1) Does the baseline alone explain views?
    views = [d["views"] for d in vols]
    log_views = [log(v) for v in views]
    exp_dem = [d["exp_demand"] for d in vols]
    log_exp = [log(e) for e in exp_dem]
    log_resid = [d["log_resid"] for d in vols]

    print(f"\n=== Does the structural baseline explain views? ===")
    print(f"  pearson(views, expected_demand)            = {pearson(views, exp_dem):+.3f}")
    print(f"  pearson(log_views, log_expected_demand)    = {pearson(log_views, log_exp):+.3f}")

    _, r2_raw, _ = fit_ols(views, [cen(exp_dem)])
    _, r2_log, _ = fit_ols(log_views, [cen(log_exp)])
    print(f"  R² of views ~ expected_demand (raw)        = {r2_raw:.3f}")
    print(f"  R² of log(views) ~ log(expected_demand)    = {r2_log:.3f}")

    # 2) Do music features predict the RESIDUAL?
    print(f"\n=== Do music features predict the residual (log_views − log_expected)? ===")
    print(f"  i.e., is there a 'music quality' signal independent of demographics?\n")

    rate = [d["chart_rate"] for d in vols]
    n_aca = [d["n_aca"] for d in vols]
    n_ch = [d["n_aca_charted"] for d in vols]

    print(f"  pearson(log_resid, chart_rate)             = {pearson(log_resid, rate):+.3f}")
    print(f"  pearson(log_resid, n_aca)                  = {pearson(log_resid, n_aca):+.3f}")
    print(f"  pearson(log_resid, n_aca_charted)          = {pearson(log_resid, n_ch):+.3f}")

    _, r2_resid_rate, _ = fit_ols(log_resid, [cen(rate)])
    _, r2_resid_full, _ = fit_ols(log_resid,
        [cen(rate), cen(n_aca), cen(n_ch)])
    print(f"\n  R² of log_resid ~ chart_rate                       = {r2_resid_rate:.3f}")
    print(f"  R² of log_resid ~ chart_rate + n_aca + n_aca_charted = {r2_resid_full:.3f}")

    # 3) Compare: structural baseline + music features vs music-only.
    print(f"\n=== Combined: structural baseline + music features ===")
    _, r2_combined, _ = fit_ols(views,
        [cen(exp_dem), cen(rate), cen(n_aca), cen(n_ch)])
    _, r2_music_only, _ = fit_ols(views,
        [cen(rate), cen(n_aca), cen(n_ch)])
    print(f"  R² music-only (chart_rate + n_aca + n_aca_charted)   = {r2_music_only:.3f}")
    print(f"  R² baseline + music                                  = {r2_combined:.3f}")
    print(f"  Δ from adding demographic baseline                   = {r2_combined - r2_music_only:+.3f}")

    # 4) Residual ranking — which volumes most over/under-perform?
    print(f"\n=== Top 5 over- and under-performers (vs demographic prediction) ===")
    ranked = sorted(vols, key=lambda d: -d["log_resid"])
    print(f"  OVER-performers (more views than demographics predict):")
    for d in ranked[:5]:
        print(f"    {d['title'][:50]:52s}  log_resid = {d['log_resid']:+.2f}")
    print(f"  UNDER-performers:")
    for d in ranked[-5:]:
        print(f"    {d['title'][:50]:52s}  log_resid = {d['log_resid']:+.2f}")

    # Persist
    aux = sqlite3.connect(AUX)
    cur = aux.cursor()
    findings = [
        ("pearson_views_expected_demand", "all", pearson(views, exp_dem)),
        ("pearson_log_views_log_expected", "all", pearson(log_views, log_exp)),
        ("r2_views_demand_baseline", "all", r2_raw),
        ("r2_log_views_log_demand", "all", r2_log),
        ("pearson_log_resid_chart_rate", "all", pearson(log_resid, rate)),
        ("pearson_log_resid_n_aca", "all", pearson(log_resid, n_aca)),
        ("pearson_log_resid_n_aca_charted", "all", pearson(log_resid, n_ch)),
        ("r2_log_resid_music_only", "all", r2_resid_full),
        ("r2_combined_demand_plus_music", "all", r2_combined),
        ("r2_music_only", "all", r2_music_only),
        ("delta_r2_adding_demand", "all", r2_combined - r2_music_only),
    ]
    for metric, group, val in findings:
        cur.execute("""
            INSERT INTO analysis_results
              (analysis_name, metric, group_key, value)
            VALUES ('bb_demographic_baseline_v1', ?, ?, ?)
            ON CONFLICT(analysis_name, metric, group_key) DO UPDATE SET
              value = excluded.value, computed_at = CURRENT_TIMESTAMP
        """, (metric, group, float(val)))
    aux.commit()
    print(f"\npersisted {len(findings)} metrics to aux.analysis_results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
