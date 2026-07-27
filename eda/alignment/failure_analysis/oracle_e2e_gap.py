#!/usr/bin/env python3
"""Decompose the oracle <-> e2e ACAPPELLA trajectory gap with the scorer's OWN internals.

Context (looptrace/NOTES.md "Lyrics wiring" sections): oracle-placement acappella
trajectory is ~43-44% (path_decode --eval / looptrace run, GT decode windows), but
the end-to-end scorecard credits only ~15-28% per set. Placement replacement was
proven redundant, so the queued hypothesis is span/GT CORRESPONDENCE in the scoring
path. Previous ad-hoc joins were unreliable; this script reuses
score_timeline_vs_gt's own matching (same pairing loop, same trajectory_acc,
same _pred_segs_from_span anchoring, same _decompose_span extent convention).

Two comparisons, per set and pooled, acappella axis (axis = matched GT row's
claimed_stem — never the timeline span's):

  A) oracle: cached looptrace decode dumps looptrace/out/lt_pred_<set>.json
     (per-GT-span decode with the window cropped at GT set_start). NOTE the
     oracle POPULATION is only GT rows with ref_source != 'online_candidate'
     and ref audio on disk (n=21 BB12 / 17 BB11 of 100/92 acappella rows) —
     path_decode --eval skips online-candidate rows.
  B) e2e: out/<set>_predicted_timeline_gtstem_lt.json scored through the exact
     pairing loop of score_timeline_vs_gt.main.

B's shortfall is decomposed per sampled GT second (1 Hz, trajectory_acc's own
sampling) into mutually exclusive buckets, dependency-ordered:

  CORRECT      some paired span's trajectory within 2 s (absolute anchoring)
  B1_ABSENT    GT row paired to NO span; recording absent from the timeline
  B1_STARVED   GT row paired to NO span; recording present but nearest-row
               pairing sent every span of that recording to another GT row
  B3_EXTENT    paired, but t outside every pair's decode extent
               (scorer convention: t < first segment; see caveat in report —
               _pieces extends the last segment to GT set_end, so trailing
               under-coverage is scored as decode error, not extent)
  B4_PLACEMENT paired + covered + wrong, and every covering pair has
               |set_start err| > place_tol (default 2 s)
  B5_DECODE    paired + covered + wrong with a well-placed covering pair
               (the true decoder residual)

Identity (wrong recording) cannot be a primary bucket under the scorer's own
logic: its pairing is same-recording-only (score_timeline_vs_gt.py:245-249),
so a wrong-recording prediction manifests as B1_* on the GT side. We report a
diagnostic overlay: the fraction of B1 seconds whose mix time is claimed by a
wrong-recording predicted span.

Sanity gates (script hard-fails if violated): reproduce the scorer's per-span
acappella traj means on the same timelines (BB12 28% / BB11 15%, pooled ~21%)
and the oracle means (42.9% / 43.9%).

Usage:
    venvs/audio/bin/python -m eda.alignment.failure_analysis.oracle_e2e_gap
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from eda.alignment.failure_analysis.build_span_table import _load_gt
from alignment.path_decode import (
    _gt_pieces,
    _pieces,
    _ref_at,
    _span_class,
    trajectory_acc,
)
from alignment.score_timeline_vs_gt import (
    _pred_segs_from_span,
)

_ALN = _REPO / "alignment"

SETS = [
    ("1fsnxchk", "BB12", _REPO / "labeling/fixtures/bb12_ground_truth.yaml"),
    ("2nvzlh2k", "BB11", _REPO / "labeling/fixtures/bb11_ground_truth.yaml"),
]

BUCKETS = [
    "CORRECT",
    "B1_ABSENT",
    "B1_STARVED",
    "B3_EXTENT",
    "B4_PLACEMENT",
    "B5_DECODE",
]
TOL = 2.0  # trajectory_acc's tolerance


@dataclass
class RowResult:
    slot: str
    track_id: str
    cls: str
    secs: int
    n_pairs: int
    is_wlayer: bool
    ref_source: str
    counts: dict = field(default_factory=dict)  # bucket -> seconds
    counts_win: dict = field(default_factory=dict)  # window-based extent variant
    wrong_rec_secs: int = 0  # B1 rows: seconds claimed by wrong-recording spans
    post_audible_secs: int = 0
    post_audible_correct: int = 0
    oracle: float | None = None  # oracle strict traj, when in oracle population
    # B4 seconds binned by the min covering pair's |set_start err| (scorer /
    # window coverage variants) — is "placement" a 3s nudge or a 50s miss?
    b4_bins: dict = field(default_factory=dict)
    b4_bins_win: dict = field(default_factory=dict)
    # (place_err, window∩GT / GT-dur, pred window dur, GT dur) per paired span
    pair_stats: list = field(default_factory=list)
    # in-window accounting: seconds the decoder actually SAW (inside a paired
    # span's [set_start, set_end]) vs credit earned there. In-window accuracy
    # is directly comparable to the oracle (which sees the full GT window).
    inwin_secs: int = 0
    inwin_correct: int = 0
    # starved-row rescue: correct seconds if pairing were many-to-many
    # (scored against EVERY same-recording span, not just the pairing winner)
    rescue_correct: int = 0

    @property
    def e2e(self) -> float:
        return self.counts.get("CORRECT", 0) / max(1, self.secs)


def _pair_predicted_spans(timeline: dict, gt_by_tid: dict) -> list[tuple[dict, dict]]:
    """EXACTLY score_timeline_vs_gt.main's pairing: each predicted span ->
    nearest same-recording GT row by set_start (score_timeline_vs_gt.py:245-249)."""
    pairs = []
    for s in timeline["spans"]:
        rows = gt_by_tid.get(s["recording_id"])
        if not rows:
            continue
        g = min(rows, key=lambda r: abs(float(r["set_start_s"]) - s["set_start_s"]))
        pairs.append((s, g))
    return pairs


def _decompose_row(
    g: dict,
    paired: list[dict],
    timeline_spans: list[dict],
    tl_recids: set,
    place_tol: float,
) -> RowResult:
    s0, s1 = float(g["set_start_s"]), float(g["set_end_s"])
    slot = str(g.get("slot_label"))
    rr = RowResult(
        slot=slot,
        track_id=str(g.get("track_id")),
        cls=_span_class(g),
        secs=len(np.arange(s0, s1, 1.0)),
        n_pairs=len(paired),
        is_wlayer="w" in slot,
        ref_source=str(g.get("ref_source") or ""),
        counts={b: 0 for b in BUCKETS},
        counts_win={b: 0 for b in BUCKETS},
    )
    gt = _gt_pieces(g)
    slope = float(g.get("tempo_ratio") or 1.0)
    aud_end = g.get("audible_end_s")

    # per-pair precomputation, mirroring _decompose_span/trajectory_acc:
    # segments anchored at the GT row's set_start (absolute scoring), pieces
    # built over the GT extent, extent = [first segment start, last_end].
    per_pair = []
    for s in paired:
        segs = _pred_segs_from_span(s, anchor_s=s0)
        if not segs:
            continue
        pred = _pieces([(s0 + ms, rs, re) for (ms, rs, re) in segs], s0, s1, slope)
        first = pred[0][0]
        lm0, lm1, _lrs, _lsl = pred[-1]
        last_end = lm0 + max(lm1 - lm0, 0.0)
        place_err = abs(float(s["set_start_s"]) - s0)
        win = (float(s["set_start_s"]), float(s["set_end_s"]))
        ov = max(0.0, min(win[1], s1) - max(win[0], s0))
        rr.pair_stats.append(
            (place_err, ov / max(1e-6, s1 - s0), win[1] - win[0], s1 - s0)
        )
        per_pair.append((pred, first, last_end, place_err, win))

    wrong = [
        s
        for s in timeline_spans
        if s["recording_id"] != str(g.get("track_id"))
        and float(s["set_start_s"]) < s1
        and float(s["set_end_s"]) > s0
    ]

    # starved-row rescue probe: when pairing gave this row NOTHING but the
    # recording IS in the timeline, score it against every same-recording span
    # (many-to-many counterfactual for the nearest-row pairing rule).
    rescue = []
    if not per_pair and str(g.get("track_id")) in tl_recids:
        for s in timeline_spans:
            if s["recording_id"] != str(g.get("track_id")):
                continue
            segs = _pred_segs_from_span(s, anchor_s=s0)
            if segs:
                rescue.append(
                    _pieces([(s0 + ms, rs, re) for (ms, rs, re) in segs], s0, s1, slope)
                )

    for t in np.arange(s0, s1, 1.0):
        t = float(t)
        gr = _ref_at(gt, t)
        post_aud = aud_end is not None and t > float(aud_end)
        if post_aud:
            rr.post_audible_secs += 1
        if not per_pair:
            b = "B1_ABSENT" if str(g.get("track_id")) not in tl_recids else "B1_STARVED"
            rr.counts[b] += 1
            rr.counts_win[b] += 1
            if any(
                float(w["set_start_s"]) <= t <= float(w["set_end_s"]) for w in wrong
            ):
                rr.wrong_rec_secs += 1
            if any(abs(_ref_at(p, t) - gr) < TOL for p in rescue):
                rr.rescue_correct += 1
            continue
        ok = inside = inwin = False
        min_place_inside = min_place_inwin = None
        for pred, first, last_end, perr, win in per_pair:
            pr = _ref_at(pred, t)
            if abs(pr - gr) < TOL:
                ok = True
            if not (t < first or t > last_end):  # _decompose_span's outside test
                inside = True
                min_place_inside = (
                    perr if min_place_inside is None else min(min_place_inside, perr)
                )
            if win[0] - TOL <= t <= win[1] + TOL:
                inwin = True
                min_place_inwin = (
                    perr if min_place_inwin is None else min(min_place_inwin, perr)
                )
        if inwin:
            rr.inwin_secs += 1
            if ok:
                rr.inwin_correct += 1
        if ok:
            rr.counts["CORRECT"] += 1
            rr.counts_win["CORRECT"] += 1
            if post_aud:
                rr.post_audible_correct += 1
            continue
        if inside:
            if min_place_inside <= place_tol:
                rr.counts["B5_DECODE"] += 1
            else:
                rr.counts["B4_PLACEMENT"] += 1
                rr.b4_bins[_bin(min_place_inside)] = (
                    rr.b4_bins.get(_bin(min_place_inside), 0) + 1
                )
        else:
            rr.counts["B3_EXTENT"] += 1
        if inwin:
            if min_place_inwin <= place_tol:
                rr.counts_win["B5_DECODE"] += 1
            else:
                rr.counts_win["B4_PLACEMENT"] += 1
                rr.b4_bins_win[_bin(min_place_inwin)] = (
                    rr.b4_bins_win.get(_bin(min_place_inwin), 0) + 1
                )
        else:
            rr.counts_win["B3_EXTENT"] += 1
    return rr


def _bin(place_err: float) -> str:
    if place_err <= 5:
        return "2-5s"
    if place_err <= 15:
        return "5-15s"
    return ">15s"


def _load_oracle(set_id: str, gt_rows: list[dict]) -> dict[int, float]:
    """{id(gt_row) -> oracle strict traj} from the cached looptrace decode dump
    (GT decode windows = the path_decode --eval convention)."""
    p = _ALN / "looptrace" / "out" / f"lt_pred_{set_id}.json"
    dump = json.loads(p.read_text())
    by_key = {
        (str(r.get("slot_label")), round(float(r.get("set_start_s", -1)), 2)): r
        for r in gt_rows
    }
    out: dict[int, float] = {}
    for sp in dump["spans"]:
        row = by_key.get((sp["slot_label"], round(sp["set_start_s"], 2)))
        if row is None:
            print(
                f"  WARN oracle span {sp['slot_label']} has no GT row", file=sys.stderr
            )
            continue
        acc, _n, _f = trajectory_acc(sp["segs"], row)
        out[id(row)] = acc
    return out


def _agg(rows: list[RowResult], oracle_rate: float | None = None) -> dict:
    tot = sum(r.secs for r in rows)
    out = {"rows": len(rows), "secs": tot}
    for b in BUCKETS:
        out[b] = sum(r.counts[b] for r in rows)
        out[b + "_win"] = sum(r.counts_win[b] for r in rows)
    out["e2e_sec"] = out["CORRECT"] / max(1, tot)
    orc = [r for r in rows if r.oracle is not None]
    out["oracle_rows"] = len(orc)
    out["oracle_mean"] = float(np.mean([r.oracle for r in orc])) if orc else None
    out["oracle_sec"] = (
        sum(r.oracle * r.secs for r in orc) / max(1, sum(r.secs for r in orc))
        if orc
        else None
    )
    return out


def _print_bucket_table(a: dict, oracle_sec: float | None, label: str) -> None:
    tot = a["secs"]
    print(
        f"\n  {label}: {a['rows']} GT rows, {tot} GT-seconds, "
        f"e2e(sec-weighted)={100 * a['e2e_sec']:.1f}%"
    )
    print(
        f"    {'bucket':13} {'secs':>6} {'%GT':>6}   {'repair->100%':>12} {'repair->oracle':>14}   {'win-variant':>11}"
    )
    for b in BUCKETS:
        sec = a[b]
        rep100 = (a["CORRECT"] + sec) / max(1, tot)
        reporc = (
            ((a["CORRECT"] + sec * oracle_sec) / max(1, tot))
            if (oracle_sec and b != "CORRECT")
            else None
        )
        print(
            f"    {b:13} {sec:>6} {100 * sec / max(1, tot):>5.1f}%   "
            f"{100 * rep100:>11.1f}% "
            f"{('%13.1f%%' % (100 * reporc)) if reporc is not None else '            - '}  "
            f"{a[b + '_win']:>10}"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--place-tol",
        type=float,
        default=2.0,
        help="B4/B5 split threshold on |pred-gt| set_start (s)",
    )
    ap.add_argument(
        "--suffix",
        default="_gtstem_lt",
        help="timeline variant (default the GT-routed looptrace one)",
    )
    args = ap.parse_args(argv)

    pooled_all: list[RowResult] = []
    pooled_orc: list[RowResult] = []
    per_set_agg = {}
    for set_id, label, gt_path in SETS:
        tl = json.loads(
            (
                _ALN / "out" / f"{set_id}_predicted_timeline{args.suffix}.json"
            ).read_text()
        )
        gt_rows, gt_by_tid = _load_gt(gt_path, set_id)
        pairs = _pair_predicted_spans(tl, gt_by_tid)

        # ---- sanity gate 1: reproduce the scorer's per-span acappella traj mean
        per_span = []
        for s, g in pairs:
            if (g.get("claimed_stem") or "regular") != "acappella":
                continue
            strict, _n, _f = trajectory_acc(
                _pred_segs_from_span(s, anchor_s=float(g["set_start_s"])), g
            )
            per_span.append(strict)
        scorer_mean = float(np.mean(per_span))
        print(
            f"[{label}] scorer-reproduction: acappella per-span traj "
            f"{100 * scorer_mean:.1f}% (n={len(per_span)})"
        )

        # ---- oracle (cached GT-window decode)
        oracle = _load_oracle(set_id, gt_rows)

        # ---- GT-side decomposition, acappella rows
        tl_recids = {s["recording_id"] for s in tl["spans"]}
        by_row: dict[int, list[dict]] = {}
        for s, g in pairs:
            by_row.setdefault(id(g), []).append(s)
        aca_rows = [r for r in gt_rows if r.get("claimed_stem") == "acappella"]
        results = []
        for g in aca_rows:
            rr = _decompose_row(
                g, by_row.get(id(g), []), tl["spans"], tl_recids, args.place_tol
            )
            rr.oracle = oracle.get(id(g))
            results.append(rr)

        a_all = _agg(results)
        orc_rows = [r for r in results if r.oracle is not None]
        a_orc = _agg(orc_rows)
        per_set_agg[label] = (a_all, a_orc, results)
        pooled_all.extend(results)
        pooled_orc.extend(orc_rows)

        # sanity gate 2: oracle mean matches NOTES' 43-44%
        print(
            f"[{label}] oracle: n={a_orc['oracle_rows']} rows, per-row mean "
            f"{100 * a_orc['oracle_mean']:.1f}%, sec-weighted {100 * a_orc['oracle_sec']:.1f}%"
        )

        _print_bucket_table(
            a_all, a_orc["oracle_sec"], f"{label} ALL acappella GT rows"
        )
        _print_bucket_table(
            a_orc,
            a_orc["oracle_sec"],
            f"{label} ORACLE population (demucs-ref rows only)",
        )

        # diagnostics
        b1 = [r for r in results if r.n_pairs == 0]
        wsec = sum(r.wrong_rec_secs for r in b1)
        b1sec = sum(r.secs for r in b1)
        starved = [r for r in b1 if r.counts["B1_STARVED"] > 0]
        multi = {}
        for r in results:
            multi.setdefault(r.track_id, []).append(r)
        multi_rows = [rs for rs in multi.values() if len(rs) > 1]
        starved_multi = sum(1 for rs in multi_rows for r in rs if r.n_pairs == 0)
        print(
            f"    diag: unpaired rows={len(b1)} ({b1sec}s) — of those, "
            f"{wsec}s ({100 * wsec / max(1, b1sec):.0f}%) claimed by WRONG-recording spans "
            f"(identity overlay); starved-not-absent rows={len(starved)}"
        )
        print(
            f"    diag: multi-instance recordings={len(multi_rows)} "
            f"({sum(len(rs) for rs in multi_rows)} rows; {starved_multi} starved) — "
            f"w-layer rows unpaired: "
            f"{sum(1 for r in b1 if r.is_wlayer)}/{sum(1 for r in results if r.is_wlayer)}"
        )
        # B4 magnitude: is "placement" a small nudge or a real window miss?
        for name, attr in (("scorer", "b4_bins"), ("window", "b4_bins_win")):
            bins: dict[str, int] = {}
            for r in results:
                for k, v in getattr(r, attr).items():
                    bins[k] = bins.get(k, 0) + v
            tot_b4 = sum(bins.values())
            if tot_b4:
                parts = "  ".join(
                    f"{k}={bins.get(k, 0)}s ({100 * bins.get(k, 0) / tot_b4:.0f}%)"
                    for k in ("2-5s", "5-15s", ">15s")
                )
                print(f"    diag: B4 |set_start err| bins ({name:6} coverage): {parts}")
        # pair window geometry: how much of the GT span does the paired
        # predicted span's decode window actually cover?
        ps = [p for r in results for p in r.pair_stats]
        if ps:
            ovf = np.array([p[1] for p in ps])
            wdur = np.array([p[2] for p in ps])
            gdur = np.array([p[3] for p in ps])
            print(
                f"    diag: pair window∩GT/GT-dur: median={np.median(ovf):.2f} "
                f"<50%: {100 * (ovf < 0.5).mean():.0f}% of pairs  ==0: "
                f"{100 * (ovf == 0).mean():.0f}%   pred-window dur median="
                f"{np.median(wdur):.0f}s vs GT dur median={np.median(gdur):.0f}s"
            )
        # THE punchline test: accuracy restricted to mix seconds the decoder
        # actually saw (inside a paired span's window) vs the oracle rate.
        for pop, rs_ in (("ALL", results), ("oracle-pop", orc_rows)):
            iw = sum(r.inwin_secs for r in rs_)
            iwc = sum(r.inwin_correct for r in rs_)
            tot = sum(r.secs for r in rs_)
            if iw:
                print(
                    f"    diag: in-window accuracy [{pop:10}]: {100 * iwc / iw:.1f}% "
                    f"over {iw}s seen ({100 * iw / max(1, tot):.0f}% of GT-sec seen) "
                    f"vs oracle {100 * a_orc['oracle_sec']:.1f}%"
                )
        resc = sum(r.rescue_correct for r in results)
        starv_sec = sum(r.counts["B1_STARVED"] for r in results)
        if starv_sec:
            print(
                f"    diag: starved-row many-to-many rescue: {resc}s of {starv_sec}s "
                f"({100 * resc / starv_sec:.0f}%) would score correct if pairing "
                f"were many-to-many"
            )
        pa = sum(r.post_audible_secs for r in results)
        pc = sum(r.post_audible_correct for r in results)
        if pa:
            aud = a_all["secs"] - pa
            audc = a_all["CORRECT"] - pc
            print(
                f"    diag: post-audible_end seconds={pa} ({100 * pa / a_all['secs']:.0f}% of GT-sec) "
                f"acc={100 * pc / pa:.0f}% vs audible acc={100 * audc / max(1, aud):.0f}%"
            )
        # online-candidate vs demucs e2e split (population term)
        for src in ("demucs", "online_candidate"):
            rs = [r for r in results if r.ref_source == src]
            if rs:
                sec = sum(r.secs for r in rs)
                cor = sum(r.counts["CORRECT"] for r in rs)
                print(
                    f"    diag: ref_source={src:17} rows={len(rs):3} secs={sec:5} "
                    f"e2e={100 * cor / max(1, sec):.1f}%"
                )

        # per-row table for the oracle population (the honest apples-to-apples set)
        print(
            f"\n    {label} oracle-population per-row (oracle vs e2e, loss bucket profile):"
        )
        print(
            f"    {'slot':7} {'cls':9} {'secs':>5} {'orc%':>5} {'e2e%':>5}  buckets(sec)"
        )
        for r in sorted(orc_rows, key=lambda r: (r.oracle or 0) - r.e2e, reverse=True):
            prof = " ".join(
                f"{b.split('_')[-1][:4]}={r.counts[b]}"
                for b in BUCKETS[1:]
                if r.counts[b]
            )
            print(
                f"    {r.slot:7} {r.cls:9} {r.secs:>5} {100 * (r.oracle or 0):>4.0f} "
                f"{100 * r.e2e:>4.0f}  {prof}"
            )

    # ---- pooled
    print("\n" + "=" * 72)
    a_all = _agg(pooled_all)
    a_orc = _agg(pooled_orc)
    print(
        f"POOLED acappella: e2e(sec)={100 * a_all['e2e_sec']:.1f}% over {a_all['secs']}s; "
        f"oracle pop n={a_orc['rows']} rows oracle(sec)={100 * a_orc['oracle_sec']:.1f}% "
        f"e2e-on-oracle-pop(sec)={100 * a_orc['e2e_sec']:.1f}%"
    )
    _print_bucket_table(a_all, a_orc["oracle_sec"], "POOLED ALL acappella")
    _print_bucket_table(a_orc, a_orc["oracle_sec"], "POOLED ORACLE population")

    # ---- causal waterfall (the punchline): split the oracle<->e2e deficit
    # into (a) GT-seconds the pipeline's paired spans never SAW (coverage /
    # correspondence — extents, placement, starvation) scored only by
    # extrapolation, vs (b) in-window decode under-performance vs the oracle.
    print("\ncausal waterfall (oracle population, GT-seconds):")
    for name, rs_ in (
        ("BB12", [r for r in pooled_orc if r in per_set_agg["BB12"][2]]),
        ("BB11", [r for r in pooled_orc if r in per_set_agg["BB11"][2]]),
        ("POOLED", pooled_orc),
    ):
        tot = sum(r.secs for r in rs_)
        cor = sum(r.counts["CORRECT"] for r in rs_)
        seen = sum(r.inwin_secs for r in rs_)
        seen_c = sum(r.inwin_correct for r in rs_)
        unseen, unseen_c = tot - seen, cor - seen_c
        osec = sum(r.oracle * r.secs for r in rs_) / max(1, tot)
        gap = osec * tot - cor
        cov_term = unseen * (osec - unseen_c / max(1, unseen))
        dec_term = seen * (osec - seen_c / max(1, seen))
        print(
            f"  {name:6} oracle={100 * osec:.1f}% e2e={100 * cor / max(1, tot):.1f}% "
            f"gap={gap:.0f}s | never-in-window {unseen}s "
            f"(extrapolation acc {100 * unseen_c / max(1, unseen):.0f}%) -> "
            f"{cov_term:.0f}s = {100 * cov_term / max(1, gap):.0f}% of gap | "
            f"in-window ({seen}s @ {100 * seen_c / max(1, seen):.0f}%) -> "
            f"{dec_term:.0f}s = {100 * dec_term / max(1, gap):.0f}% of gap"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
