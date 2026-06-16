#!/usr/bin/env python3
"""v0 — does forgiving acoustically-identical sub-sections rescue ref-offset?

Tests the hypothesis that the documented ~50 s ref-offset "error" on BB12 is
mostly *repeat ambiguity* (chorus-2 chosen over chorus-1) that should not be
counted wrong. For each straight-clip GT row we localize the ref offset with a
per-bar MERT matched filter, then score two ways:

  exact     : |pred_ref_start - gt_ref_start| < tol_s
  equivalent: predicted ref window ~= GT ref window in the track's own audio
              (cosine >= --thresh) -> the two are the same bars, not an error

If equivalent >> exact, the placement task is much easier than the raw
scorecard implies, and the residual is genuinely the switch-time + which-class
problem the section HSMM targets.

Runs fully off the local MERT cache (workspaces/alignment_prototype/.cache) —
no pi-storage needed.

Usage:
    venvs/audio/bin/python -m workspaces.section_hsmm.v0_scorecard \
        --set-id 1fsnxchk [--thresh 0.90] [--tol-s 2.0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.result import Err, Ok  # noqa: E402
from workspaces.alignment_prototype.mert_store import load_bb12_mert  # noqa: E402
from workspaces.section_hsmm.equivalence import (  # noqa: E402
    l2norm,
    matched_filter,
    self_similarity_floor,
    windows_equivalent,
)


def _straight_rows(gt_path: Path) -> list[dict]:
    """GT rows that are single linear clips (loops/segments excluded — they
    are not representable by one (ref_start, stretch) and the prior work
    counts them separately too)."""
    rows = yaml.safe_load(gt_path.read_text())["tracks"]
    out = []
    for r in rows:
        if str(r.get("slot_label")) == "mix" or not r.get("track_id"):
            continue
        if r.get("is_loop") or r.get("ref_segments"):
            continue
        ratio = float(r.get("tempo_ratio") or 1.0)
        if not (0.9 <= ratio <= 1.15):
            continue
        out.append(r)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set-id", default="1fsnxchk")
    p.add_argument("--gt", type=Path,
                   default=_REPO / "labeling/fixtures/bb12_ground_truth.yaml")
    p.add_argument("--thresh", type=float, default=0.90,
                   help="window cosine for two ref offsets to count as the same audio")
    p.add_argument("--tol-s", type=float, default=2.0,
                   help="exact-match tolerance in seconds")
    args = p.parse_args(argv)

    match load_bb12_mert(args.set_id):
        case Err(msg):
            print(f"failed to load MERT cache for {args.set_id}: {msg}", file=sys.stderr)
            return 1
        case Ok((_set_audio_id, mix, refs)):
            pass

    mix_mid = 0.5 * (mix.start_s + mix.end_s)
    mix_vecs_n = l2norm(mix.vectors)

    rows = _straight_rows(args.gt)
    results: list[dict] = []
    skipped_no_ref = 0
    skipped_short = 0
    for r in rows:
        tid = str(r["track_id"])
        ref = refs.get(tid)
        if ref is None or ref.n_measures == 0:
            skipped_no_ref += 1
            continue
        set_start, set_end = float(r["set_start_s"]), float(r["set_end_s"])
        mask = (mix_mid >= set_start) & (mix_mid <= set_end)
        mix_win = mix_vecs_n[mask]
        if mix_win.shape[0] < 2:
            skipped_short += 1
            continue
        M = mix_win.shape[0]
        ref_vecs_n = l2norm(ref.vectors)
        if ref_vecs_n.shape[0] < M:
            skipped_short += 1
            continue

        p_pred, score, _ = matched_filter(mix_win, ref_vecs_n)
        pred_ref_start = float(ref.start_s[p_pred])
        gt_ref_start = float(r["ref_start_s"])
        p_gt = int(np.argmin(np.abs(ref.start_s - gt_ref_start)))

        err_s = abs(pred_ref_start - gt_ref_start)
        exact = err_s < args.tol_s
        equiv, eq_cos = windows_equivalent(ref_vecs_n, p_pred, p_gt, M, args.thresh)
        floor = self_similarity_floor(ref_vecs_n, M)
        results.append({
            "stem": r.get("claimed_stem") or "regular",
            "err_s": err_s, "exact": exact, "equiv": equiv,
            "eq_cos": eq_cos, "floor": floor, "score": score,
            "name": str(r.get("track", ""))[:40], "tid": tid,
        })

    n = len(results)
    if n == 0:
        print("no scorable straight-clip rows", file=sys.stderr)
        return 1

    def pct(key: str, sub: list[dict]) -> str:
        if not sub:
            return "   -"
        return f"{100 * np.mean([x[key] for x in sub]):3.0f}%"

    print(f"=== v0 equivalence-aware ref-offset scorecard ({args.set_id}) ===")
    print(f"straight clips scored: {n}  "
          f"(skipped {skipped_no_ref} no-ref-MERT, {skipped_short} too-short)  "
          f"thresh={args.thresh} tol={args.tol_s}s")
    print(f"{'stem':13} {'n':>3}  {'exact<tol':>9}  {'equiv':>6}  {'rescued':>7}")
    for stem in ("regular", "acappella", "instrumental", None):
        sub = results if stem is None else [x for x in results if x["stem"] == stem]
        if not sub:
            continue
        rescued = [x for x in sub if x["equiv"] and not x["exact"]]
        label = "ALL" if stem is None else stem
        print(f"{label:13} {len(sub):3}  {pct('exact', sub):>9}  "
              f"{pct('equiv', sub):>6}  {len(rescued):3} ({100*len(rescued)/len(sub):2.0f}%)")

    errs = np.array([x["err_s"] for x in results])
    print(f"\nraw |pred-gt| ref offset: median={np.median(errs):.1f}s  "
          f"p90={np.percentile(errs, 90):.1f}s")
    floors = np.array([x["floor"] for x in results if np.isfinite(x["floor"])])
    if floors.size:
        print(f"track self-similarity floor (median off-diagonal window cos): "
              f"{np.median(floors):.2f}  "
              f"-> equiv threshold {args.thresh} is {'above' if args.thresh > np.median(floors) else 'BELOW'} chance")

    rescued_all = [x for x in results if x["equiv"] and not x["exact"]]
    if rescued_all:
        print(f"\nrescued by equivalence (exact-wrong but same audio), top by gap:")
        for x in sorted(rescued_all, key=lambda x: -x["err_s"])[:8]:
            print(f"  {x['err_s']:6.1f}s off  eq_cos={x['eq_cos']:.2f}  "
                  f"{x['stem']:12} {x['name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
