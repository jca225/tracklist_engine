#!/usr/bin/env python3
"""Failure-mode analysis over span_table.csv.

Reads the per-span table (build_span_table.py) and produces the "where & why"
view: stratified error by axis x span-class x set, IMPACT-WEIGHTED by GT-seconds
lost (not span count), single dominant-cause attribution per failing span, and a
cross-set consistency check. Descriptive only — per-stratum n is small (5-65), so
NO fitted models (house rule: small-n regressions aren't findings).

Usage:
    venvs/audio/bin/python -m eda.alignment.failure_analysis.analyze
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent / "out"

# a span "succeeds" if >= this fraction of its GT seconds decode within 2s
TRAJ_OK = 0.5
PLACEMENT_MISS_S = 15.0  # set_start error beyond which the decode window misses
DISTINCT_HI = 0.4  # audit frac_distinct above which repeats are instance-ambiguous


def _cause(row) -> str:
    """Single BINDING cause per span, in dependency order: a wrong identity makes
    everything downstream moot, a missed window makes the decode moot, etc."""
    if row.identity_hit == 0:
        return "identity"
    if row.set_start_err_s > PLACEMENT_MISS_S:
        return "placement"
    # consequential mis-route: truth is acappella but pipeline routed it to chroma
    # (or vice-versa) due to stale claimed_stem — regular<->instrumental both use
    # chroma, so only the acappella boundary changes the feature/decoder.
    if (row.gt_stem == "acappella") != (row.pred_stem == "acappella"):
        return "mis-route"
    if row.span_class == "oddratio":
        return "tempo/octave"
    if row.span_class == "loop":
        return "loop-instance"
    if pd.notna(row.frac_distinct) and row.frac_distinct >= DISTINCT_HI:
        return "instance-ambiguity"
    return "decode-residual"


def main() -> int:
    csv_name = sys.argv[1] if len(sys.argv) > 1 else "span_table.csv"
    df = pd.read_csv(OUT_DIR / csv_name)
    m = df[df.span_class.notna() & (df.span_class != "")].copy()
    m["identity_hit"] = m["identity_hit"].fillna(1)  # no-GT-overlap: don't penalize
    m["success"] = m.traj_strict >= TRAJ_OK
    m["seconds"] = m.gt_duration_s
    m["sec_lost"] = m.seconds * (1 - m.traj_strict)

    tot_sec = m.seconds.sum()
    print(f"matched spans: {len(m)}  |  GT seconds: {tot_sec:.0f}")
    print(f"success rule: traj_strict >= {TRAJ_OK} (>=50% of span decodes within 2s)\n")

    # ---- 1. stratify: axis x span-class, impact-weighted ------------------
    print("=" * 78)
    print("1. TRAJECTORY by axis x span-class (n, mean traj, GT-sec, %corpus-sec)")
    print("=" * 78)
    for stem in ("regular", "acappella", "instrumental"):
        sub = m[m.gt_stem == stem]
        if sub.empty:
            continue
        print(
            f"\n  [{stem}]  n={len(sub)}  traj={sub.traj_strict.mean():.0%}  "
            f"GT-sec={sub.seconds.sum():.0f} ({sub.seconds.sum() / tot_sec:.0%} of corpus)"
        )
        g = (
            sub.groupby("span_class")
            .agg(
                n=("slot", "size"),
                traj=("traj_strict", "mean"),
                sec=("seconds", "sum"),
                sec_lost=("sec_lost", "sum"),
            )
            .sort_values("sec_lost", ascending=False)
        )
        for cls, r in g.iterrows():
            print(
                f"      {cls:9} n={int(r.n):3}  traj={r.traj:4.0%}  "
                f"sec={r.sec:6.0f}  sec_lost={r.sec_lost:6.0f}"
            )

    # ---- 2. impact ranking: where are the seconds lost --------------------
    print("\n" + "=" * 78)
    print("2. IMPACT: GT-seconds LOST, ranked by (axis x span-class)")
    print("=" * 78)
    g = (
        m.groupby(["gt_stem", "span_class"])
        .agg(
            n=("slot", "size"),
            traj=("traj_strict", "mean"),
            sec_lost=("sec_lost", "sum"),
        )
        .reset_index()
        .sort_values("sec_lost", ascending=False)
    )
    tot_lost = m.sec_lost.sum()
    print(
        f"total GT-seconds lost: {tot_lost:.0f} of {tot_sec:.0f} "
        f"({tot_lost / tot_sec:.0%})\n"
    )
    for _, r in g.head(10).iterrows():
        print(
            f"  {r.gt_stem:12} {r.span_class:9} n={int(r.n):3}  traj={r.traj:4.0%}  "
            f"sec_lost={r.sec_lost:6.0f}  ({r.sec_lost / tot_lost:4.0%} of loss)"
        )

    # ---- 3. dominant-cause attribution ------------------------------------
    print("\n" + "=" * 78)
    print("3. BINDING CAUSE per span (dependency-ordered) — share of seconds lost")
    print("=" * 78)
    m["cause"] = m.apply(_cause, axis=1)
    ca = (
        m.groupby("cause")
        .agg(
            n=("slot", "size"),
            traj=("traj_strict", "mean"),
            sec=("seconds", "sum"),
            sec_lost=("sec_lost", "sum"),
        )
        .sort_values("sec_lost", ascending=False)
    )
    print(
        f"\n  {'cause':20} {'n':>4} {'traj':>6} {'GT-sec':>8} {'sec_lost':>9} {'%loss':>7}"
    )
    for cause, r in ca.iterrows():
        print(
            f"  {cause:20} {int(r.n):4} {r.traj:6.0%} {r.sec:8.0f} "
            f"{r.sec_lost:9.0f} {r.sec_lost / tot_lost:7.0%}"
        )

    # ---- 4. cross-set consistency (transfer / low-rank test) --------------
    print("\n" + "=" * 78)
    print("4. CROSS-SET consistency (does the same cause dominate in both sets?)")
    print("=" * 78)
    piv = m.pivot_table(
        index="cause", columns="set", values="sec_lost", aggfunc="sum", fill_value=0
    )
    for col in ("BB12", "BB11"):
        if col in piv:
            piv[col + "_%"] = piv[col] / piv[col].sum()
    print("\n" + piv.round(2).to_string())

    # ---- 5. targeted diagnostics ------------------------------------------
    print("\n" + "=" * 78)
    print("5. DIAGNOSTICS")
    print("=" * 78)
    # 5a. does placement error explain trajectory failure?
    placed = m[m.set_start_err_s <= PLACEMENT_MISS_S]
    missed = m[m.set_start_err_s > PLACEMENT_MISS_S]
    print(
        f"\n  placement<=15s (n={len(placed)}): traj={placed.traj_strict.mean():.0%}  "
        f"|  placement>15s (n={len(missed)}): traj={missed.traj_strict.mean():.0%}"
    )
    # 5a2. ROUTING mis-label: pipeline routed on stale-materialize claimed_stem.
    # gt_stem (truth) vs pred_stem (what drove the feature/decoder route).
    print("\n  routing mis-label (pred_stem = stale set_track_slots vs GT truth):")
    ct = pd.crosstab(m.gt_stem, m.pred_stem)
    print("    truth(rows) x routed(cols):\n" + ct.to_string().replace("\n", "\n    "))
    for st in ("acappella", "instrumental", "regular"):
        sub = m[m.gt_stem == st]
        if not len(sub):
            continue
        mis = sub[sub.stem_mismatch == 1]
        ok = sub[sub.stem_mismatch == 0]
        traj_mis = f"{mis.traj_strict.mean():.0%}" if len(mis) else "-"
        traj_ok = f"{ok.traj_strict.mean():.0%}" if len(ok) else "-"
        print(
            f"    {st:12} n={len(sub):3}  mis-routed={len(mis):3} "
            f"({len(mis) / len(sub):.0%})  traj[correct]={traj_ok} traj[mis]={traj_mis}"
        )
    # 5b. oddratio tempo distribution (octave-fold suspects)
    odd = m[m.span_class == "oddratio"]
    if not odd.empty:
        tr = odd.tempo_ratio
        print(
            f"  oddratio spans (n={len(odd)}): tempo_ratio "
            f"min={tr.min():.2f} med={tr.median():.2f} max={tr.max():.2f}  "
            f"traj={odd.traj_strict.mean():.0%}  "
            f"(near half/double: {((tr < 0.6) | (tr > 1.7)).mean():.0%})"
        )
    # 5c. instance ambiguity: audit-covered spans, traj vs frac_distinct
    aud = m[m.frac_distinct.notna()]
    if not aud.empty:
        lo = aud[aud.frac_distinct < DISTINCT_HI]
        hi = aud[aud.frac_distinct >= DISTINCT_HI]
        print(
            f"  audit-covered (n={len(aud)}): distinct<{DISTINCT_HI} "
            f"(n={len(lo)}) traj={lo.traj_strict.mean():.0%}  |  "
            f">= (n={len(hi)}) traj={hi.traj_strict.mean():.0%}"
        )
    # 5d. ref-offset (straight clips) per set/stem
    print("\n  ref-offset MAE (straight clips only):")
    ro = df[df.ref_offset_scored == 1].copy()
    ro["ref_offset_err_s"] = pd.to_numeric(ro.ref_offset_err_s, errors="coerce")
    for st in ("regular", "acappella"):
        for setn in ("BB12", "BB11"):
            e = ro[(ro.gt_stem == st) & (ro.set == setn)].ref_offset_err_s.dropna()
            if len(e):
                print(
                    f"    {setn} {st:11} n={len(e):3} median={e.median():5.1f}s "
                    f"<2s={(e < 2).mean():.0%} <5s={(e < 5).mean():.0%}"
                )

    # 5e. coverage / upstream loss (never-matched GT recordings)
    print(
        "\n  coverage: predicted spans with NO same-recording GT anchor "
        "(pred'd a song GT never places):"
    )
    for setn in ("BB12", "BB11"):
        d = df[df.set == setn]
        nogt = d[d.span_class.isna() | (d.span_class == "")]
        print(f"    {setn}: {len(nogt)} of {len(d)} spans")

    print("\n(persisted table: eda/alignment/failure_analysis/out/span_table.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
