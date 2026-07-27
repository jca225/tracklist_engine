#!/usr/bin/env python3
"""Per-span looptrace|legacy router.

The two decoders fail on DIFFERENT spans (Phase-3 gate: looptrace wins
BB11 across the board and loops; the legacy HuBERT matched-filter wins
BB12 linear/oddratio). Route each span by looptrace's own evidence signal:
`evidence_rate` = tight-tolerance path-inlier evidence per decoded second
(from the run.py dump meta). High = the landmark cloud clearly explains
the span -> trust looptrace; low = fall back to the legacy decode.

Threshold selection is LEAVE-ONE-SET-OUT: theta is chosen on the OTHER
set and applied here, so the reported routed accuracy is never tuned on
the set it is scored on. The full theta sweep is also printed for
transparency.

Usage:
    venvs/audio/bin/python -m alignment.looptrace.router \
        --lt out/lt_pred_<A>.json --legacy out/pred_<A>_hubert.json \
        --lt-other out/lt_pred_<B>.json --legacy-other out/pred_<B>_hubert.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from alignment.path_decode import (  # noqa: E402
    _span_class,
    trajectory_acc,
)


def _load(pred_path: Path) -> tuple[dict, list[dict]]:
    """(gt_rows_by_key, spans) for one dump."""
    import yaml

    pred = json.loads(pred_path.read_text())
    raw = {
        (str(r.get("slot_label")), round(float(r.get("set_start_s", -1)), 2)): r
        for r in yaml.safe_load(Path(pred["gt"]).read_text()).get("tracks", [])
    }
    return raw, pred["spans"]


def _join(lt_path: Path, legacy_path: Path) -> list[dict]:
    """Per-span records: gt row, both decodes' traj-acc, the router signal."""
    raw, lt_spans = _load(lt_path)
    _, lg_spans = _load(legacy_path)
    lg_by = {
        (str(s["slot_label"]), round(float(s["set_start_s"]), 2)): s for s in lg_spans
    }
    out: list[dict] = []
    for sp in lt_spans:
        key = (str(sp["slot_label"]), round(float(sp["set_start_s"]), 2))
        row = raw.get(key)
        lg = lg_by.get(key)
        if row is None or lg is None:
            continue
        acc_lt, _, _ = trajectory_acc(sp["segs"], row)
        acc_lg, _, _ = trajectory_acc(lg["segs"], row)
        out.append(
            {
                "slot": sp["slot_label"],
                "cls": _span_class(row),
                "signal": float(sp["meta"].get("evidence_rate", 0.0)),
                "acc_lt": acc_lt,
                "acc_lg": acc_lg,
            }
        )
    return out


def routed_acc(rows: list[dict], theta: float) -> float:
    return float(
        np.mean([r["acc_lt"] if r["signal"] >= theta else r["acc_lg"] for r in rows])
    )


def best_theta(rows: list[dict]) -> float:
    grid = sorted({r["signal"] for r in rows} | {0.0, 1e9})
    return max(grid, key=lambda th: routed_acc(rows, th))


def report(name: str, rows: list[dict], theta: float) -> None:
    accs = {
        "looptrace-only": np.mean([r["acc_lt"] for r in rows]),
        "legacy-only": np.mean([r["acc_lg"] for r in rows]),
        f"routed (LOSO theta={theta:.1f})": routed_acc(rows, theta),
        "oracle-per-span": np.mean([max(r["acc_lt"], r["acc_lg"]) for r in rows]),
    }
    print(f"\n=== router — {name} (n={len(rows)}) ===")
    for k, v in accs.items():
        print(f"  {k:28} traj-acc {100 * v:3.0f}%")
    for cls in ("linear", "multiseg", "loop", "oddratio"):
        sel = [r for r in rows if r["cls"] == cls]
        if not sel:
            continue
        rt = routed_acc(sel, theta)
        lt = float(np.mean([r["acc_lt"] for r in sel]))
        lg = float(np.mean([r["acc_lg"] for r in sel]))
        print(
            f"    {cls:9} n={len(sel):2}  routed {100 * rt:3.0f}%  "
            f"(lt {100 * lt:3.0f}% / legacy {100 * lg:3.0f}%)"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lt", type=Path, required=True)
    p.add_argument("--legacy", type=Path, required=True)
    p.add_argument("--lt-other", type=Path, required=True)
    p.add_argument("--legacy-other", type=Path, required=True)
    args = p.parse_args(argv)

    rows_a = _join(args.lt, args.legacy)
    rows_b = _join(args.lt_other, args.legacy_other)
    theta_for_a = best_theta(rows_b)  # LOSO: tuned on the OTHER set
    theta_for_b = best_theta(rows_a)
    report(args.lt.stem, rows_a, theta_for_a)
    report(args.lt_other.stem, rows_b, theta_for_b)
    print("\ntheta sweep (transparency):")
    for name, rows in ((args.lt.stem, rows_a), (args.lt_other.stem, rows_b)):
        grid = sorted({round(r["signal"], 1) for r in rows})
        line = "  ".join(f"{th:.1f}:{100 * routed_acc(rows, th):.0f}" for th in grid)
        print(f"  {name}: {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
