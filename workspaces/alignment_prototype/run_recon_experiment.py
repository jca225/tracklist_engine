#!/usr/bin/env python3
"""Turnkey A/B harness: does reconstruction refinement improve the aligner, HELD-OUT?

Step-2 v1 experiment (docs/reconstruction_supervision_plan.md). Given a train set and an
eval set, compares two timelines against the eval set's GT:
  BASELINE  = the existing pipeline (identity model + fp/DP placement)
  TREATMENT = BASELINE + recon_rerank (reconstruction-margin placement refinement, host)

In-domain smoke: --train-set 1fsnxchk --eval-set 1fsnxchk (BB12; timeline already exists).
Held-out (turnkey once BB11 GT is exported): --train-set 1fsnxchk --eval-set 2nvzlh2k.

Nothing here reads the eval set's GT except the final scorer, so the held-out number is honest.

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.run_recon_experiment \\
        --train-set 1fsnxchk --eval-set 1fsnxchk [--gate 0.02 --band-s 30 --fibers]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_MOD = "workspaces.alignment_prototype"
OUT_DIR = Path(__file__).resolve().parent / "out"
PY = str(_REPO / "venvs/audio/bin/python")

# set_id -> GT yaml under labeling/fixtures/ (extend as sets are labeled)
GT_MAP = {
    "1fsnxchk": "bb12_ground_truth.yaml",
    "2nvzlh2k": "bb11_ground_truth.yaml",
    "w1mgcjt": "bb10_ground_truth.yaml",
}


def _run(argv: list[str], capture: bool = True) -> str:
    print(f"$ {' '.join(argv)}", file=sys.stderr)
    r = subprocess.run(
        argv,
        cwd=str(_REPO),
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    if r.returncode != 0 and capture:
        print(r.stdout, file=sys.stderr)
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(argv)}")
    return r.stdout or ""


def _headline(scorecard: str) -> list[str]:
    """Pull the comparable lines out of a score_timeline_vs_gt scorecard."""
    keep = ("identity", "set placement", "median", "p90", "<", "ref ", "traj")
    return [ln for ln in scorecard.splitlines() if any(k in ln.lower() for k in keep)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-set", default="1fsnxchk")
    ap.add_argument("--eval-set", default="1fsnxchk")
    ap.add_argument(
        "--gt", type=Path, default=None, help="eval-set GT yaml (else auto)"
    )
    ap.add_argument("--band-s", type=float, default=30.0)
    ap.add_argument("--gate", type=float, default=0.08)
    ap.add_argument("--fibers", action="store_true", help="pass to scorer (expensive)")
    ap.add_argument(
        "--infer",
        action="store_true",
        help="(re)run infer on eval-set first (needed for a fresh held-out set)",
    )
    args = ap.parse_args(argv)

    gt = args.gt or (_REPO / "labeling/fixtures" / GT_MAP.get(args.eval_set, ""))
    if not gt or not Path(gt).exists():
        print(f"\nEval GT not found: {gt}", file=sys.stderr)
        if args.eval_set == "2nvzlh2k":
            print(
                "BB11 GT not exported yet. When the .als labeling is done, run:\n"
                "  venvs/audio/bin/python -m labeling.export_als_to_gt "
                "--als ~/aligning/2nvzlh2k__*/'BB11 align.als' --set-dir ~/aligning/2nvzlh2k__*/\n"
                "then place/point --gt at labeling/fixtures/bb11_ground_truth.yaml",
                file=sys.stderr,
            )
        return 2

    base_tl = OUT_DIR / f"{args.eval_set}_predicted_timeline.json"
    if args.infer or not base_tl.exists():
        if args.eval_set == args.train_set and not args.infer:
            raise SystemExit(f"no baseline timeline {base_tl.name}; run infer first")
        _run([PY, "-m", f"{_MOD}.infer", "--set-id", args.eval_set], capture=False)

    print(
        f"\n{'=' * 64}\nRECON EXPERIMENT  train={args.train_set}  eval={args.eval_set}"
        f"  (gate={args.gate}, band={args.band_s}s)\n{'=' * 64}"
    )

    # TREATMENT timeline: recon-refine the baseline
    refined = OUT_DIR / f"{args.eval_set}_recon_refined_timeline.json"
    _run(
        [
            PY,
            "-m",
            f"{_MOD}.recon_rerank",
            "--set-id",
            args.eval_set,
            "--in",
            str(base_tl),
            "--out",
            str(refined),
            "--band-s",
            str(args.band_s),
            "--gate",
            str(args.gate),
        ],
        capture=False,
    )

    score_common = ["--set-id", args.eval_set, "--gt", str(gt)]
    if args.fibers:
        score_common.append("--fibers")
    print("\n--- scoring BASELINE ---")
    base_sc = _run(
        [
            PY,
            "-m",
            f"{_MOD}.score_timeline_vs_gt",
            *score_common,
            "--timeline",
            str(base_tl),
        ]
    )
    print("\n--- scoring TREATMENT (recon-refined) ---")
    treat_sc = _run(
        [
            PY,
            "-m",
            f"{_MOD}.score_timeline_vs_gt",
            *score_common,
            "--timeline",
            str(refined),
        ]
    )

    print(
        f"\n{'=' * 64}\nA/B  (train={args.train_set} → eval={args.eval_set})\n{'=' * 64}"
    )
    print(f"{'':<40}{'BASELINE':>12}{'+RECON':>12}")
    bl, tl_ = _headline(base_sc), _headline(treat_sc)
    for b, t in zip(bl, tl_):
        print(f"  BASE  {b}")
        print(f"  RECON {t}")
    print(
        "\n(full scorecards above; compare set-placement median / p90 / <15s and identity)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
