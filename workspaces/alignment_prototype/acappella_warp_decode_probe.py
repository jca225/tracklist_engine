#!/usr/bin/env python3
"""Phase 0 — disambiguate WHY acappella segment decode fails: anchor/grid vs warp.

The handoff (docs/agent_handoff_stem_axis_findings_20260629.md) says acappella
traj-acc is 10% because 55% of BB12 acappella spans are non-linear and the median
stretch is ~1.48x. That bundles two DISTINCT decode failures that need DIFFERENT
fixes:

  (B) ANCHOR/GRID failure — a single constant stretch DOES fit the span, but that
      stretch is not reachable by `path_decode._stretch_band`. The band is dense
      only in a +/-2% window around a beat-grid-anchored center `e = ref_bar /
      mix_bar`; for acappella the ref bar grid is unreliable (vocals break beat
      tracking) so `e` defaults to ~1.0 and the dense window sits in the wrong
      place. Fix = Phase 1 (acappella-routed wide/dense grid). CHEAP.

  (C) WARP failure — no single stretch fits the span: the DJ warped it
      phrase-by-phrase so the (mix,ref) slope VARIES within the span. The current
      Viterbi searches offset only at ONE fixed slope `s` (jumps/loops are free,
      slope is not), so it structurally cannot represent this regardless of grid.
      Fix = Phase 2 (varying-slope DP rewrite). EXPENSIVE.

The decode's representable class is: piecewise-linear with ALL pieces sharing ONE
slope `s` (offsets arbitrary -> loops/jumps are FREE). So a span is decode-
representable iff its GT per-segment slopes are ~equal AND that common slope is in
the grid. This probe measures exactly that, per acappella span, and partitions:

  A  already-representable (slope ~constant AND in grid)  -> failure is placement/
     feature, NOT the decode -> Phase 1/2 won't help these.
  B  Phase 1 fixes it        (slope ~constant, OUT of grid)
  C  Phase 2 required        (slope VARIES within span)

The A/B/C split is the deliverable: it says whether the Phase 2 DP rewrite is
worth it (large C) or whether the cheap grid widening captures most of it (large
B, small C).

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.acappella_warp_decode_probe \
        --als "$HOME/aligning/1fsnxchk__*/BB12 align.als" --set-dir "$HOME/aligning/1fsnxchk__*"
    # add --stems acappella,regular,instrumental to probe other axes too.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from labeling.export_als_to_gt import collect_kept_clip_rows  # noqa: E402
from workspaces.alignment_prototype.path_decode import _gt_pieces  # noqa: E402

# Decode tolerance: trajectory_acc credits a sample within 2.0 s of GT.
TOL_S = 2.0
# A grid point reaches a target stretch if within this RELATIVE distance. The fine
# grid step is 2% -> half a step ~= 1%, but allow ~3% so we don't over-count "out
# of grid" on rounding. Conservative toward calling things reachable (B vs C is the
# expensive call; reachable=A/B is the cheap-fix bucket).
GRID_REL_TOL = 0.03
# Per-segment slope spread above which a span genuinely needs varying-slope decode.
# 10% within-span slope variation already overruns the 2s tol on a ~40s span.
SLOPE_SPREAD_THR = 0.10


def _stretch_grid(e: float) -> np.ndarray:
    """Mirror path_decode._stretch_band given a center `e` (octave-folded inside)."""
    while e > 1.45:
        e *= 0.5
    while e < 0.7:
        e *= 2.0
    fine = (0.96, 0.98, 1.0, 1.02, 1.04)
    band = {round(e * oct_mult * f, 4) for oct_mult in (0.5, 1.0, 2.0) for f in fine}
    return np.array(sorted(band))


def _in_grid(s: float, grid: np.ndarray) -> bool:
    return bool(np.min(np.abs(grid - s) / s) <= GRID_REL_TOL)


def _row_dict(r) -> dict:
    """ClipRow -> the dict shape _gt_pieces expects (set/ref bounds + segments)."""
    segs = []
    for seg in getattr(r, "ref_segments", ()) or ():
        segs.append(
            {
                "mix_start_s": float(seg.mix_start_s),
                "ref_start_s": float(seg.ref_start_s),
                "ref_end_s": float(seg.ref_end_s),
            }
        )
    return {
        "set_start_s": float(r.set_start_s),
        "set_end_s": float(r.set_end_s),
        "ref_start_s": float(r.ref_start_s),
        "ref_end_s": float(r.ref_end_s),
        "tempo_ratio": float(r.tempo_ratio) if r.tempo_ratio else 1.0,
        "ref_segments": segs,
    }


def _segment_slopes(pieces) -> tuple[list[float], list[float]]:
    """(slopes, mix-durations) for the GT pieces; slope is the (ref/mix) ratio the
    decode would need on that piece."""
    slopes, durs = [], []
    for ms, me, _rs, slope in pieces:
        d = me - ms
        if d <= 0.5:  # ignore sub-window slivers (noise in slope estimate)
            continue
        slopes.append(float(slope))
        durs.append(float(d))
    return slopes, durs


def _wmedian(vals: np.ndarray, w: np.ndarray) -> float:
    order = np.argsort(vals)
    v, cw = vals[order], np.cumsum(w[order])
    return float(v[np.searchsorted(cw, 0.5 * cw[-1])])


def _classify(row: dict, grid_center: float, max_span_s: float) -> dict:
    """Per-span diagnosis. The decode's representable class = piecewise-linear with
    ONE global slope `s` and FREE per-segment offset (loops/jumps are free). So:
      - choose the best global slope s* = duration-weighted MEDIAN of GT segment
        slopes (robust to one warped phrase),
      - a GT segment is decode-fittable iff a constant slope s* drifts <2s across
        its OWN duration: |slope_i - s*| * dur_i < TOL_S (offset is free, so only
        the slope mismatch accrues within the segment),
      - const_slope_ceiling = duration-weighted fraction of the span that fits.
    Bucket C (Phase-2 varying-slope decode) iff that ceiling < 0.8 — i.e. even with
    a free slope and free offset-jumps, the GT's own segments can't share one slope.
    Otherwise A (s* in grid -> already representable; failure is placement/feature)
    or B (s* out of grid -> Phase-1 wide grid)."""
    pieces = _gt_pieces(row)
    slopes, durs = _segment_slopes(pieces)
    if not slopes:
        return {"bucket": "skip"}
    s0, s1 = row["set_start_s"], row["set_end_s"]
    set_span = s1 - s0
    durs_a, slopes_a = np.array(durs), np.array(slopes)
    s_rep = _wmedian(slopes_a, durs_a)
    spread = float(np.max(np.abs(slopes_a - s_rep)) / s_rep) if s_rep else 0.0
    # Jump-aware fit: a segment fits if its slope mismatch drifts <2s over its span.
    drift = np.abs(slopes_a - s_rep) * durs_a
    fit_frac = float(np.sum(durs_a[drift < TOL_S]) / np.sum(durs_a))
    in_grid = _in_grid(s_rep, _stretch_grid(grid_center))

    if set_span > max_span_s:
        bucket = "degenerate"  # whole-mix recurring motif, not a single warped clip
    elif fit_frac < 0.8:
        bucket = "C"  # genuine varying slope within the span -> Phase 2
    elif not in_grid:
        bucket = "B"  # constant slope, unreachable by current grid -> Phase 1
    else:
        bucket = "A"  # representable now -> failure is placement/feature
    return {
        "bucket": bucket,
        "s_rep": s_rep,
        "spread": spread,
        "fit_frac": fit_frac,
        "in_grid": in_grid,
        "n_seg": len(slopes),
        "set_span": set_span,
    }


def _ref_at(pieces, t: float) -> float:
    for ms, me, rs, slope in pieces:
        if ms <= t <= me:
            return rs + (t - ms) * slope
    ms, _me, rs, slope = pieces[0] if t < pieces[0][0] else pieces[-1]
    return rs + (t - ms) * slope


def _resolve_glob(pat: str) -> Path:
    hits = glob.glob(str(Path(pat).expanduser()))
    if not hits:
        print(f"no match for {pat}", file=sys.stderr)
        sys.exit(2)
    return Path(sorted(hits)[0])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--als", required=True)
    p.add_argument("--set-dir", required=True)
    p.add_argument(
        "--stems",
        default="acappella",
        help="comma list of claimed_stem axes to probe (default acappella)",
    )
    p.add_argument(
        "--grid-center",
        type=float,
        default=1.0,
        help="assumed _stretch_band center e (1.0 = acappella has no usable ref "
        "beat grid, the realistic case). Set to a measured ratio to test the "
        "best-case anchor.",
    )
    p.add_argument("--show", type=int, default=0, help="print N worst (C) spans")
    p.add_argument(
        "--max-span-s",
        type=float,
        default=400.0,
        help="spans longer than this are 'degenerate' (whole-mix recurring motif, "
        "not a single warped clip) and reported separately",
    )
    args = p.parse_args(argv)

    als = _resolve_glob(args.als)
    set_dir = _resolve_glob(args.set_dir)
    _set_id, rows, _ = collect_kept_clip_rows(als, set_dir)
    want = {s.strip() for s in args.stems.split(",") if s.strip()}

    for stem in sorted(want):
        spans = [r for r in rows if r.claimed_stem == stem]
        alld = [
            _classify(_row_dict(r), args.grid_center, args.max_span_s) for r in spans
        ]
        deg = sum(1 for d in alld if d["bucket"] == "degenerate")
        diags = [d for d in alld if d["bucket"] in ("A", "B", "C")]
        n = len(diags)
        if not n:
            print(f"\n=== {stem}: no scorable spans (degenerate={deg}) ===")
            continue
        counts = {b: sum(1 for d in diags if d["bucket"] == b) for b in "ABC"}
        print(
            f"\n=== {stem} (n={n} normal, +{deg} degenerate>{args.max_span_s:.0f}s, "
            f"grid-center e={args.grid_center}) ==="
        )
        print(
            f"  A already-representable (1 slope, in grid)  : "
            f"{counts['A']:3d}  {100 * counts['A'] / n:4.0f}%   -> failure is placement/feature"
        )
        print(
            f"  B Phase-1 grid fix      (1 slope, OUT grid) : "
            f"{counts['B']:3d}  {100 * counts['B'] / n:4.0f}%   -> wide acappella stretch grid"
        )
        print(
            f"  C Phase-2 DP required   (slope VARIES)      : "
            f"{counts['C']:3d}  {100 * counts['C'] / n:4.0f}%   -> varying-slope decode"
        )
        sreps = np.array([d["s_rep"] for d in diags])
        print(
            f"  median |s_rep-1| = {np.median(np.abs(sreps - 1)):.3f}  "
            f"(~0.5 = half-time vocal stretch);  s_rep in [{sreps.min():.2f},{sreps.max():.2f}]"
        )
        print(
            f"  reachable by CURRENT grid: {sum(1 for d in diags if d['in_grid'])}/{n} "
            f"({100 * sum(1 for d in diags if d['in_grid']) / n:.0f}%)"
        )
        if args.show:
            worst = sorted(
                (d for d in diags if d["bucket"] == "C"),
                key=lambda d: d["fit_frac"],
            )[: args.show]
            for d in worst:
                print(
                    f"    C: fit_frac={d['fit_frac']:.2f} spread={d['spread']:.2f} "
                    f"s_rep={d['s_rep']:.2f} n_seg={d['n_seg']} span={d['set_span']:.0f}s"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
