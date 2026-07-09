#!/usr/bin/env python3
"""Span-level failure correlates on the CURRENT (post-fix) timelines.

Complements analyze.py (binding-cause attribution) with the question FINDINGS.md
has not yet answered: *what measurable properties of a span predict failure*,
framed against the north-star laws (docs/architecture_north_star.md):

  - "abstain, never lie"  -> is span `confidence` calibrated enough to abstain on?
  - B1 warp / B2 key      -> does loss concentrate in stretched / re-pitched spans?
  - crosstalk             -> does GT layer depth (concurrent audible spans) predict loss?
  - repeat structure      -> frac_distinct/clone vs decode success (acappella)
  - stem coverage         -> BB11 vocals-stem-on-disk hole, measured per span

Read-only consumer: joins span_table_postfix.csv (build_span_table --suffix
_postfix_lt) with the timelines' confidence/start_source fields, GT yaml layer
depth, and out/bb_stem_coverage.tsv (pi-storage pull, one row per spine
recording: has_vocals_stem / has_instr_stem).

Usage:
    venvs/audio/bin/python -m eda.alignment.failure_analysis.failure_correlates
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.alignment_prototype.score_timeline_vs_gt import norm_slot

OUT_DIR = Path(__file__).resolve().parent / "out"
_ALN = _REPO / "workspaces" / "alignment_prototype"

SETS = [
    ("1fsnxchk", "BB12", _REPO / "labeling/fixtures/bb12_ground_truth.yaml"),
    ("2nvzlh2k", "BB11", _REPO / "labeling/fixtures/bb11_ground_truth.yaml"),
]

_LINES: list[str] = []


def _emit(s: str = "") -> None:
    print(s)
    _LINES.append(s)


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Spearman rho + n over pairwise-finite entries (no scipy dependency)."""
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 8:
        return float("nan"), n
    rx = np.argsort(np.argsort(x[m])).astype(float)
    ry = np.argsort(np.argsort(y[m])).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    return (float((rx * ry).sum() / denom) if denom else float("nan")), n


def _f(v: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _load_rows() -> list[dict]:
    with (OUT_DIR / "span_table_postfix.csv").open() as fh:
        rows = [r for r in csv.DictReader(fh)]

    # timeline extras: confidence / start_source / ref_decoder by (set_id, slot)
    extras: dict[tuple[str, str], dict] = {}
    for set_id, _label, _gt in SETS:
        tl = json.loads(
            (_ALN / "out" / f"{set_id}_predicted_timeline_postfix_lt.json").read_text()
        )
        for s in tl["spans"]:
            extras[(set_id, norm_slot(s["slot_label"]))] = s

    # GT layer depth: mean count of OTHER audible GT rows over the span interval
    gt_rows_by_set: dict[str, list[dict]] = {}
    set_len: dict[str, float] = {}
    for set_id, _label, gt_path in SETS:
        doc = yaml.safe_load(gt_path.read_text())
        grs = [r for r in doc["tracks"] if str(r.get("slot_label")) != "mix"]
        gt_rows_by_set[set_id] = grs
        set_len[set_id] = max(float(r["set_end_s"]) for r in grs)

    def interval(r: dict) -> tuple[float, float]:
        a = r.get("audible_start_s")
        b = r.get("audible_end_s")
        if a is not None and b is not None and float(b) > float(a):
            return float(a), float(b)
        return float(r["set_start_s"]), float(r["set_end_s"])

    def layer_depth(set_id: str, lo: float, hi: float, self_slot: str) -> float:
        if hi <= lo:
            return float("nan")
        t = np.arange(lo, hi, 1.0)
        depth = np.zeros_like(t)
        for r in gt_rows_by_set[set_id]:
            if norm_slot(str(r.get("slot_label") or "")) == self_slot:
                continue
            a, b = interval(r)
            depth += ((t >= a) & (t < b)).astype(float)
        return float(depth.mean())

    unalignable = {
        (sid, norm_slot(str(r.get("slot_label") or "")))
        for sid, grs in gt_rows_by_set.items()
        for r in grs
        if r.get("unalignable")
    }

    stems: dict[tuple[str, str], tuple[int, int]] = {}
    cov_path = OUT_DIR / "bb_stem_coverage.tsv"
    if cov_path.exists():
        for line in cov_path.read_text().splitlines():
            sid, rid, hv, hi_ = line.split("\t")
            stems[(sid, rid)] = (int(hv), int(hi_))

    out = []
    for r in rows:
        if not r.get("span_class"):  # unmatched span, no GT anchor
            continue
        sid, slot = r["set_id"], r["slot"]
        e = extras.get((sid, slot), {})
        lo = _f(r["gt_set_start_s"])
        hi = lo + _f(r["gt_duration_s"])
        hv, hin = stems.get((sid, r["recording_id"]), (np.nan, np.nan))
        out.append(
            {
                **r,
                "confidence": _f(e.get("confidence")),
                "start_source": e.get("start_source") or "",
                "ref_decoder": e.get("ref_decoder") or "",
                "layer_depth": layer_depth(sid, lo, hi, slot),
                "position_frac": lo / set_len[sid],
                "abs_tempo_dev": abs(_f(r["tempo_ratio"]) - 1.0),
                "pitched": int(_f(r["pitch_shift_semi"]) != 0.0),
                "has_vocals_stem": hv,
                "has_instr_stem": hin,
                "unalignable": int((sid, slot) in unalignable),
                "traj": _f(r["traj_strict"]),
                "sserr": _f(r["set_start_err_s"]),
                "dur": _f(r["gt_duration_s"]),
            }
        )
    return out


def _col(rows: list[dict], k: str) -> np.ndarray:
    return np.array(
        [_f(str(r[k])) if not isinstance(r[k], float) else r[k] for r in rows]
    )


def _bin_table(
    rows: list[dict], key: str, edges: list[float], labels: list[str]
) -> None:
    v = _col(rows, key)
    traj = _col(rows, "traj")
    sserr = _col(rows, "sserr")
    dur = _col(rows, "dur")
    idh = _col(rows, "identity_hit")
    _emit(
        f"    {'bin':<12} {'n':>4} {'traj':>6} {'GTsec-w traj':>12} {'ss_med':>7} {'id%':>5}"
    )
    for i in range(len(edges) - 1):
        m = np.isfinite(v) & (v >= edges[i]) & (v < edges[i + 1]) & np.isfinite(traj)
        if m.sum() == 0:
            continue
        w = np.nansum(traj[m] * dur[m]) / np.nansum(dur[m])
        _emit(
            f"    {labels[i]:<12} {int(m.sum()):>4} {np.nanmean(traj[m]):>6.2f} "
            f"{w:>12.2f} {np.nanmedian(sserr[m]):>7.1f} "
            f"{100 * np.nanmean(idh[m]):>4.0f}%"
        )


def main() -> int:
    rows = _load_rows()
    n_unal = sum(r["unalignable"] for r in rows)
    rows = [r for r in rows if not r["unalignable"]]
    _emit(f"matched spans: {len(rows)}  (dropped {n_unal} GT-unalignable)")

    axes = ("acappella", "regular", "instrumental")

    # ---- 1. north-star law 3: is confidence abstain-able? -------------------
    _emit("\n== 1. CONFIDENCE CALIBRATION (north-star 'abstain, never lie') ==")
    for ax in axes + ("ALL",):
        sub = rows if ax == "ALL" else [r for r in rows if r["gt_stem"] == ax]
        conf, traj = _col(sub, "confidence"), _col(sub, "traj")
        rho, n = _spearman(conf, traj)
        rho2, _ = _spearman(conf, _col(sub, "sserr"))
        _emit(
            f"  {ax:<12} n={n:<4} spearman(conf,traj)={rho:+.2f}  (conf,ss_err)={rho2:+.2f}"
        )
    _emit("  abstention sweep (ALL): drop lowest-confidence X% -> mean traj of kept")
    conf, traj, dur = _col(rows, "confidence"), _col(rows, "traj"), _col(rows, "dur")
    m = np.isfinite(conf) & np.isfinite(traj)
    order = np.argsort(conf[m])
    ct, cd = traj[m][order], dur[m][order]
    for x in (0, 10, 20, 30, 40, 50):
        k = int(len(ct) * x / 100)
        w = np.nansum(ct[k:] * cd[k:]) / np.nansum(cd[k:])
        _emit(
            f"    drop {x:>2}%  kept n={len(ct) - k:<4} mean traj={np.nanmean(ct[k:]):.2f}  GTsec-w={w:.2f}"
        )

    # start_source is the better stratifier: which probe placed the span
    _emit("  outcome by (axis, start_source) — the usable abstention signal:")
    from collections import defaultdict

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["gt_stem"], r["start_source"] or "?")].append(r)
    for (ax, src), sub in sorted(groups.items()):
        if len(sub) < 5:
            continue
        _emit(
            f"    {ax:<13} {src:<12} n={len(sub):<4} "
            f"traj={np.nanmean(_col(sub, 'traj')):.2f}  "
            f"ss_med={np.nanmedian(_col(sub, 'sserr')):.1f}s"
        )

    # ---- 2. crosstalk ---------------------------------------------------------
    _emit("\n== 2. GT LAYER DEPTH (crosstalk) vs outcome ==")
    for ax in axes:
        sub = [r for r in rows if r["gt_stem"] == ax]
        rho, n = _spearman(_col(sub, "layer_depth"), _col(sub, "traj"))
        _emit(f"  {ax} (n={n}, spearman depth->traj {rho:+.2f}):")
        _bin_table(sub, "layer_depth", [0, 1, 2, 3, 99], ["<1", "1-2", "2-3", ">=3"])

    # ---- 3. B1 warp / B2 key --------------------------------------------------
    _emit("\n== 3. NORTH-STAR B1 (warp) & B2 (key) DIMENSIONS ==")
    tr = _col(rows, "tempo_ratio")
    _emit(
        f"  tempo_ratio: {np.isfinite(tr).sum()} spans, "
        f"|ratio-1|>2% = {100 * np.nanmean(np.abs(tr - 1) > 0.02):.0f}%, "
        f">10% = {100 * np.nanmean(np.abs(tr - 1) > 0.10):.0f}%"
    )
    _bin_table(
        rows,
        "abs_tempo_dev",
        [0, 0.02, 0.06, 0.12, 9],
        ["<2%", "2-6%", "6-12%", ">12%"],
    )
    pit = _col(rows, "pitched")
    _emit(f"  re-pitched spans: {int(np.nansum(pit))}/{int(np.isfinite(pit).sum())}")
    for ax in axes:
        sub = [r for r in rows if r["gt_stem"] == ax]
        for p in (0, 1):
            ss = [r for r in sub if r["pitched"] == p]
            if len(ss) < 5:
                continue
            t = _col(ss, "traj")
            ro = _col(ss, "ref_offset_err_s")
            _emit(
                f"    {ax:<12} pitched={p}  n={len(ss):<4} traj={np.nanmean(t):.2f}  "
                f"ref_off med={np.nanmedian(ro):.1f}s"
            )

    # ---- 4. repeat structure (acappella decode wall) ---------------------------
    _emit("\n== 4. REPEAT STRUCTURE vs decode (acappella) ==")
    sub = [r for r in rows if r["gt_stem"] == "acappella"]
    for k in ("frac_clone", "frac_distinct", "frac_unique"):
        rho, n = _spearman(_col(sub, k), _col(sub, "traj"))
        _emit(f"  spearman({k}, traj) = {rho:+.2f}  (n={n})")
    _bin_table(
        sub,
        "frac_unique",
        [0, 0.25, 0.5, 0.75, 1.01],
        ["<.25", ".25-.5", ".5-.75", ">=.75"],
    )

    # ---- 5. span geometry -------------------------------------------------------
    _emit("\n== 5. SPAN GEOMETRY ==")
    for ax in axes:
        sub = [r for r in rows if r["gt_stem"] == ax]
        _emit(f"  {ax} — duration bins:")
        _bin_table(
            sub, "dur", [0, 15, 40, 90, 9999], ["<15s", "15-40s", "40-90s", ">90s"]
        )
    _emit("  position in set (ALL):")
    _bin_table(
        rows, "position_frac", [0, 0.25, 0.5, 0.75, 1.01], ["Q1", "Q2", "Q3", "Q4"]
    )
    _emit("  audibility (ALL):")
    _bin_table(
        rows,
        "audible_frac",
        [0, 0.5, 0.8, 0.95, 1.01],
        ["<.5", ".5-.8", ".8-.95", ">=.95"],
    )

    # ---- 6. stem coverage hole ---------------------------------------------------
    _emit("\n== 6. VOCALS-STEM-ON-DISK vs acappella outcome (per set) ==")
    _emit(
        "  CAVEAT: has_vocals_stem is the CURRENT pi DB state (2026-07-09"
        "  backfill), anachronistic vs the Jul-8 timeline run — see 6b for the"
        "  run-time attribution via the local manifest."
    )
    for set_label in ("BB12", "BB11"):
        sub = [r for r in rows if r["set"] == set_label and r["gt_stem"] == "acappella"]
        for hv in (1, 0):
            ss = [r for r in sub if r["has_vocals_stem"] == hv]
            if not ss:
                _emit(f"  {set_label} has_vocals={hv}: n=0")
                continue
            t, se = _col(ss, "traj"), _col(ss, "sserr")
            ro = _col(ss, "ref_offset_err_s")
            _emit(
                f"  {set_label} has_vocals={hv}: n={len(ss):<4} traj={np.nanmean(t):.2f}  "
                f"ss_med={np.nanmedian(se):.1f}s  ref_off med={np.nanmedian(ro):.1f}s"
            )

    # ---- 6b. decoder coverage: the silent looptrace-empty fallback -----------------
    # `ref_decoder` present == looptrace produced segments; absent on an
    # acappella span == looptrace was skipped or returned EMPTY and the span
    # silently fell back to the legacy matched filter. Attribute the cause via
    # the local ~/aligning manifest the run actually used.
    _emit("\n== 6b. LOOPTRACE COVERAGE on acappella (silent-fallback attribution) ==")
    try:
        from workspaces.alignment_prototype.refine_ref_offsets import (
            find_aligning_dir,
        )

        for set_id, set_label, _gt in SETS:
            set_dir = find_aligning_dir(set_id)
            import msgspec

            from core.contracts import load_manifest
            from core.contracts.manifest import MANIFEST_FILENAME

            man = msgspec.to_builtins(
                load_manifest(Path(set_dir) / MANIFEST_FILENAME)
            )
            by_tid = {t["track_id"]: t for t in man["tracks"]}
            for t in man["tracks"]:
                if t.get("recording_id"):
                    by_tid.setdefault(t["recording_id"], t)
            sub = [
                r for r in rows if r["set_id"] == set_id and r["gt_stem"] == "acappella"
            ]
            dec = [r for r in sub if r["ref_decoder"]]
            und = [r for r in sub if not r["ref_decoder"]]
            _emit(
                f"  {set_label}: decoded n={len(dec)} traj={np.nanmean(_col(dec, 'traj')):.2f}"
                f"  |  undecoded n={len(und)} traj={np.nanmean(_col(und, 'traj')):.2f}"
            )
            cats: dict[str, list[dict]] = {}
            for r in und:
                t = by_tid.get(r["recording_id"])
                if t is None:
                    cat = "manifest-miss"
                else:
                    sp = (t.get("stems") or {}).get("vocals")
                    if sp and Path(sp).is_file():
                        cat = "decode-empty (local stem present)"
                    elif t.get("local_path") and Path(t["local_path"]).is_file():
                        cat = "no-local-vocals-stem"
                    else:
                        cat = "no-local-audio-at-all"
                cats.setdefault(cat, []).append(r)
            for c, rs in sorted(cats.items(), key=lambda kv: -len(kv[1])):
                _emit(
                    f"    {c:<38} n={len(rs):<4} traj={np.nanmean(_col(rs, 'traj')):.2f}"
                    f"  GT-sec={sum(r['dur'] for r in rs):.0f}"
                )
    except Exception as e:  # noqa: BLE001 — manifest availability is env-specific
        _emit(f"  (skipped: {e})")

    # ---- 7. compact spearman matrix ---------------------------------------------
    _emit("\n== 7. SPEARMAN vs traj_strict, per axis ==")
    feats = [
        "confidence",
        "layer_depth",
        "abs_tempo_dev",
        "dur",
        "position_frac",
        "audible_frac",
        "frac_unique",
        "n_ref_segments",
        "sserr",
    ]
    hdr = "  " + f"{'feature':<16}" + "".join(f"{ax:>14}" for ax in axes)
    _emit(hdr)
    for f in feats:
        line = f"  {f:<16}"
        for ax in axes:
            sub = [r for r in rows if r["gt_stem"] == ax]
            rho, n = _spearman(_col(sub, f), _col(sub, "traj"))
            line += f"{rho:>+9.2f}({n:>3})" if np.isfinite(rho) else f"{'--':>13} "
        _emit(line)

    (OUT_DIR / "failure_correlates.txt").write_text("\n".join(_LINES) + "\n")
    _emit(f"\nwrote {OUT_DIR / 'failure_correlates.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
