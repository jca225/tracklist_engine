"""Does acappella density vary by DJ — beyond the uploader's annotation style?

The naive cross-tab (acappella-rows per set, grouped by `creator_name`) is CONFOUNDED:
`creator_name` is the 1001tracklists *uploader*, not the DJ, so it measures who
annotates acappellas thoroughly, not who plays them. The DJ isn't in a clean column
(`artists` is empty corpus-wide) — it lives in `title` (e.g. "Two Friends - Big Bootie
Mix Volume 12", "it's murph @ Club Space"). This script:

  1. Parses a normalized DJ from `title`.
  2. Counts acappella indicators per set from raw rows (text "(acap" / "acapella"),
     denominator = dj_sets.total_tracks → per-set acap_rate.
  3. CONFOUND CONTROL (the point): tests whether between-DJ variation in acap_rate
     exceeds what the uploader explains, via a within-uploader label-permutation null
     (shuffle DJ labels across sets that share an uploader, preserving each uploader's
     annotation rate). Real DJ effect => observed between-DJ variance sits ABOVE the null.
  4. Reports DJ↔uploader collinearity (a DJ transcribed by one uploader can't be
     separated — flagged, not silently trusted).
  5. Within-set POSITION: acap density by normalized row-order quintile (do acappellas
     cluster early/late?).

CAVEAT: acappella detection here is raw-text (annotation-dependent). The clean version
uses the pi-materialized `set_track_slots.claimed_stem` (empty in the local dev DB) — pass
--db to a pi-synced copy for that. Findings from the text proxy are directional, not final.

Usage:
    venvs/audio/bin/python -m corpus_empirics.dj_acappella_density [--db PATH] [--min-sets 20]
    # (run from eda/ so `corpus_empirics` is importable, mirroring bb_era_orthogonality)
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics as st
import sys
from pathlib import Path

_EDA = Path(__file__).resolve().parents[1]  # eda/ (for the corpus_empirics import)
_REPO = Path(__file__).resolve().parents[2]  # repo root (for data/ paths)
if str(_EDA) not in sys.path:
    sys.path.insert(0, str(_EDA))

from corpus_empirics.stats import fit_ols  # noqa: E402,F401  (available for a follow-up OLS)

DB_DEFAULT = _REPO / "data/db/music_database.db"
OUT = _REPO / "data/analysis/dj_acappella_density.json"

_ACAP_RE = re.compile(r"acapella|acappella|\(acap", re.I)
# DJ from title: text before the first " @ " (venue sets) or " - " (named mixes)
_SPLIT_RE = re.compile(r"\s+@\s+|\s+-\s+")


def parse_dj(title: str | None) -> str | None:
    if not title:
        return None
    head = _SPLIT_RE.split(title.strip(), maxsplit=1)[0]
    dj = head.strip().lower()
    dj = re.sub(r"^(dj|it's|its)\s+", "", dj)  # light normalization
    return dj or None


def between_group_var(groups: dict[str, list[float]]) -> float:
    """Variance of group means, weighted by group size (an ANOVA-style DJ signal)."""
    all_vals = [v for vs in groups.values() for v in vs]
    if len(all_vals) < 2:
        return 0.0
    grand = st.mean(all_vals)
    return sum(len(vs) * (st.mean(vs) - grand) ** 2 for vs in groups.values()) / len(
        all_vals
    )


def permute_within_uploader(
    records, n_perm: int, seed_stride: int = 7
) -> tuple[float, float, int]:
    """Observed between-DJ variance vs a null that shuffles DJ labels WITHIN each
    uploader stratum (preserving per-uploader annotation rate). Returns
    (observed, p_value, n_ge). Deterministic permutation (index rotation) — no RNG."""
    # observed
    obs_groups: dict[str, list[float]] = {}
    for r in records:
        obs_groups.setdefault(r["dj"], []).append(r["acap_rate"])
    observed = between_group_var(obs_groups)

    # strata: sets sharing an uploader
    by_up: dict[str, list[dict]] = {}
    for r in records:
        by_up.setdefault(r["uploader"], []).append(r)

    n_ge = 0
    for p in range(1, n_perm + 1):
        perm_groups: dict[str, list[float]] = {}
        for rows in by_up.values():
            djs = [r["dj"] for r in rows]
            k = (p * seed_stride) % max(1, len(djs))
            rot = djs[k:] + djs[:k]  # rotate DJ labels within this uploader
            for r, dj in zip(rows, rot):
                perm_groups.setdefault(dj, []).append(r["acap_rate"])
        if between_group_var(perm_groups) >= observed:
            n_ge += 1
    return observed, (n_ge + 1) / (n_perm + 1), n_ge


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DB_DEFAULT)
    ap.add_argument("--min-sets", type=int, default=20)
    ap.add_argument("--n-perm", type=int, default=500)
    args = ap.parse_args(argv)

    c = sqlite3.connect(args.db)
    # per-set acappella-row count from raw rows
    acap_by_set: dict[str, int] = {}
    pos_hits: list[float] = []  # normalized row-order position of each acap row
    set_maxrow: dict[str, int] = {}
    for sid, ri, html, txt in c.execute(
        "select set_id, row_index, raw_html, text_excerpt from dj_set_rows"
    ):
        blob = f"{html or ''} {txt or ''}"
        set_maxrow[sid] = max(set_maxrow.get(sid, 0), ri or 0)
        if _ACAP_RE.search(blob):
            acap_by_set[sid] = acap_by_set.get(sid, 0) + 1

    records = []
    for sid, title, uploader, total in c.execute(
        "select set_id, title, creator_name, total_tracks from dj_sets "
        "where total_tracks > 0 and creator_name is not null"
    ):
        dj = parse_dj(title)
        if not dj:
            continue
        n_acap = acap_by_set.get(sid, 0)
        records.append(
            {
                "set_id": sid,
                "dj": dj,
                "uploader": uploader,
                "n_acap": n_acap,
                "total": total,
                "acap_rate": n_acap / total,
            }
        )
        mx = set_maxrow.get(sid, 0)
        # (position pass done separately below via a second scan would double IO; approx here)

    # position: second targeted pass only over acap rows (cheap)
    for sid, ri in c.execute(
        "select set_id, row_index from dj_set_rows "
        "where lower(raw_html) like '%acapella%' or lower(raw_html) like '%acappella%'"
    ):
        mx = set_maxrow.get(sid, 0)
        if mx > 0:
            pos_hits.append((ri or 0) / mx)

    # keep DJs with enough sets
    from collections import Counter

    dj_counts = Counter(r["dj"] for r in records)
    keep = {d for d, n in dj_counts.items() if n >= args.min_sets}
    kept = [r for r in records if r["dj"] in keep]

    # per-DJ summary
    per_dj: dict[str, list[float]] = {}
    dj_uploaders: dict[str, set] = {}
    for r in kept:
        per_dj.setdefault(r["dj"], []).append(r["acap_rate"])
        dj_uploaders.setdefault(r["dj"], set()).add(r["uploader"])
    summary = sorted(
        (
            {
                "dj": d,
                "n_sets": len(v),
                "mean_acap_rate": round(st.mean(v), 3),
                "n_uploaders": len(dj_uploaders[d]),
            }
            for d, v in per_dj.items()
        ),
        key=lambda x: -x["mean_acap_rate"],
    )

    # DJ↔uploader collinearity: fraction of kept DJs seen through >1 uploader
    multi = sum(1 for d in per_dj if len(dj_uploaders[d]) > 1)
    collinearity = 1 - multi / max(1, len(per_dj))

    # confound-controlled test
    observed, pval, n_ge = permute_within_uploader(kept, args.n_perm)

    # position quintiles
    quint = [0] * 5
    for p in pos_hits:
        quint[min(4, int(p * 5))] += 1
    tot_pos = sum(quint) or 1
    pos_frac = [round(q / tot_pos, 3) for q in quint]

    result = {
        "n_sets_total": len(records),
        "n_djs_kept": len(per_dj),
        "min_sets": args.min_sets,
        "dj_uploader_collinearity": round(collinearity, 3),
        "top_djs": summary[:10],
        "bottom_djs": summary[-6:],
        "confound_test": {
            "observed_between_dj_var": observed,
            "p_value_within_uploader_perm": round(pval, 4),
            "n_perm": args.n_perm,
            "verdict": (
                "DJ effect survives uploader control"
                if pval < 0.05
                else "NOT separable from uploader annotation (null)"
            ),
        },
        "acap_position_quintiles_early_to_late": pos_frac,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, default=str))

    print(f"kept {len(per_dj)} DJs (>= {args.min_sets} sets), {len(kept)} sets")
    print(
        f"DJ↔uploader collinearity: {collinearity:.0%} of DJs seen through ONE uploader "
        f"({'high — separation weak' if collinearity > 0.6 else 'ok'})"
    )
    print("\ntop acappella-rate DJs:")
    for s in summary[:8]:
        print(
            f"  {s['dj'][:30]:<30} n={s['n_sets']:<4} rate={s['mean_acap_rate']:.2f} "
            f"uploaders={s['n_uploaders']}"
        )
    print(f"\nCONFOUND TEST (does DJ variation exceed uploader annotation?):")
    print(
        f"  observed between-DJ var={observed:.4f}  p={pval:.3f}  → {result['confound_test']['verdict']}"
    )
    print(f"\nacap position (early→late quintiles): {pos_frac}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
