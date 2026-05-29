"""H1 — Han et al. (2022) peak-sensitivity weighting of BB acapellas.

For a modal viewer aged A at the volume's release, each acapella in the
volume has an associated weight w = Bigaussian(A - delta) where
delta = set_year - acap_year and the Bigaussian uses Han et al.'s fitted
parameters (peak at age 13 with asymmetric tails — wider into the past,
narrower into the post-peak years).

Tests whether weighting chart_rate / n_aca_charted by this audience-sensitivity
curve improves on the unweighted music-only regression (R² = 0.39 baseline).

Sensitivity analysis: modal age A ∈ {18, 20, 22, 24} — BB skews EDM-young,
so we expect somewhere in this range.

Persists results to aux.analysis_results under analysis_name='bb_han_weighted_v1'.
"""

from __future__ import annotations

import sqlite3
import sys
from math import exp
from collections import defaultdict

DB = "data/db/music_database.db"
AUX = "data/analysis/aux.db"

# Han et al. 2022, Section "Sensitivity" — Bigaussian fit to all users.
# y(x) = y0 + H * exp(-0.5 * ((x-xc)/w)^2), with w=w1 if x<xc else w=w2.
XC, Y0, H, W1, W2 = 12.88, 0.43, 0.87, 13.18, 7.26


def han(x: float) -> float:
    """Sensitivity to music released when the listener was age x."""
    w = W1 if x < XC else W2
    return Y0 + H * exp(-0.5 * ((x - XC) / w) ** 2)


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
        if abs(d) < 1e-12: return None, None
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
    return b, 1 - sse/sst


def cen(xs):
    m = sum(xs)/len(xs)
    return [x-m for x in xs]


def main() -> int:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    conn.execute(f"ATTACH DATABASE '{AUX}' AS aux")

    rows = conn.execute("""
    WITH bb AS (
      SELECT set_id, CAST(substr(date_played,1,4) AS INTEGER) AS set_year
      FROM dj_sets WHERE LOWER(title) LIKE '%big bootie%'
        AND LOWER(title) NOT LIKE '%@ big bootie%'),
    rws AS (
      SELECT r.set_id, json_extract(r.data_attrs_json,'$."data-trackid"') AS tid,
             CASE WHEN r.text_excerpt LIKE 'w/%' THEN 'aca'
                  WHEN r.text_excerpt GLOB '[0-9]*' THEN 'ins' END AS role
      FROM dj_set_rows r WHERE r.set_id IN (SELECT set_id FROM bb))
    SELECT b.set_id, b.set_year, m.release_year,
           CASE WHEN cm.rank IS NOT NULL THEN 1 ELSE 0 END AS charted,
           lf.listeners, sv.view_count
    FROM bb b
    JOIN rws rw ON rw.set_id=b.set_id
    LEFT JOIN aux.track_meta m ON m.track_id=rw.tid
    LEFT JOIN aux.track_lastfm lf ON lf.track_id=rw.tid
    LEFT JOIN aux.track_chart_match cm ON cm.track_id=rw.tid
    JOIN aux.set_views sv ON sv.set_id=b.set_id AND sv.platform='youtube'
    WHERE rw.role='aca'
    """).fetchall()

    # Group by volume
    by_vol: dict[str, dict] = defaultdict(lambda: {"acaps": []})
    for r in rows:
        sid = r["set_id"]
        by_vol[sid]["set_year"] = r["set_year"]
        by_vol[sid]["views"] = r["view_count"]
        by_vol[sid]["acaps"].append({
            "release_year": r["release_year"],
            "charted": r["charted"],
            "listeners": r["listeners"],
        })

    # Test for each modal-age assumption.
    print(f"n={len(by_vol)} BB volumes\n")
    print("=== Han-weighted feature regressions (modal-age sensitivity) ===\n")

    results_to_persist = []

    # Baseline: raw chart_rate + n_aca + n_aca_charted (matches earlier R²=0.39)
    vols = sorted(by_vol.keys(), key=lambda s: by_vol[s]["set_year"])
    views = [by_vol[v]["views"] for v in vols]

    def per_vol_feat(modal_age: int):
        out = []
        for v in vols:
            d = by_vol[v]
            n_aca = len(d["acaps"])
            n_charted = sum(a["charted"] for a in d["acaps"] if a["release_year"])
            # Han weights per acap that has a release year
            weights = []
            charted_weights = []
            for a in d["acaps"]:
                if a["release_year"] is None: continue
                delta = d["set_year"] - a["release_year"]
                viewer_age_at_release = modal_age - delta
                w = han(viewer_age_at_release)
                weights.append(w)
                if a["charted"]:
                    charted_weights.append(w)
            han_weighted_chart_density = (
                sum(charted_weights) / sum(weights) if weights else 0
            )
            han_weighted_n_charted = sum(charted_weights)
            chart_rate = n_charted / n_aca if n_aca else 0
            out.append({
                "set_year": d["set_year"], "n_aca": n_aca,
                "n_aca_charted": n_charted, "chart_rate": chart_rate,
                "han_chart_density": han_weighted_chart_density,
                "han_n_charted": han_weighted_n_charted,
            })
        return out

    # Baseline (unweighted, what we had before)
    feat = per_vol_feat(modal_age=20)
    rate    = [f["chart_rate"] for f in feat]
    n_aca   = [f["n_aca"] for f in feat]
    n_ch    = [f["n_aca_charted"] for f in feat]

    _, r2_baseline = fit_ols(views, [cen(rate), cen(n_aca), cen(n_ch)])
    print(f"  BASELINE  views ~ chart_rate + n_aca + n_aca_charted             R² = {r2_baseline:.3f}")

    print(f"\n  modal_age   univariate         + n_aca, n_aca_ch    Δ vs baseline")
    print(f"  ---------   ----------------    -----------------    -------------")

    best = None
    for A in (16, 18, 20, 22, 24, 26):
        feat = per_vol_feat(modal_age=A)
        hcd  = [f["han_chart_density"] for f in feat]
        hnc  = [f["han_n_charted"] for f in feat]

        r_hcd = pearson(views, hcd)
        _, r2_uni = fit_ols(views, [cen(hcd)])
        _, r2_full = fit_ols(views, [cen(hcd), cen(n_aca), cen(hnc)])

        delta = r2_full - r2_baseline
        marker = " ←" if best is None or r2_full > best[2] else ""
        print(f"  age {A}       r={r_hcd:+.3f}  R²={r2_uni:.3f}    R²={r2_full:.3f}            "
              f"{delta:+.3f}{marker}")
        if best is None or r2_full > best[2]:
            best = (A, r2_uni, r2_full, delta)

        results_to_persist.extend([
            ("pearson_han_chart_density", f"modal_age_{A}", r_hcd),
            ("r2_han_chart_density_univariate", f"modal_age_{A}", r2_uni),
            ("r2_han_full_model", f"modal_age_{A}", r2_full),
            ("delta_r2_vs_baseline", f"modal_age_{A}", delta),
        ])

    print(f"\n  Best fit at modal_age = {best[0]} ({'improves' if best[3] > 0 else 'worse than'} baseline by {best[3]:+.3f})")

    # Compare features: is han_chart_density doing different work from chart_rate?
    feat = per_vol_feat(modal_age=best[0])
    hcd = [f["han_chart_density"] for f in feat]
    print(f"\n  pearson(han_chart_density@{best[0]}, raw chart_rate) = {pearson(hcd, rate):+.3f}")
    print(f"  pearson(han_chart_density@{best[0]}, set_year)        = "
          f"{pearson(hcd, [f['set_year'] for f in feat]):+.3f}")

    # Persist
    aux = sqlite3.connect(AUX)
    cur = aux.cursor()
    for metric, group, val in results_to_persist:
        cur.execute("""
            INSERT INTO analysis_results
              (analysis_name, metric, group_key, value)
            VALUES ('bb_han_weighted_v1', ?, ?, ?)
            ON CONFLICT(analysis_name, metric, group_key) DO UPDATE SET
              value = excluded.value, computed_at = CURRENT_TIMESTAMP
        """, (metric, group, float(val)))
    aux.commit()
    print(f"\npersisted {len(results_to_persist)} metrics to aux.analysis_results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
