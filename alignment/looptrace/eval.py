#!/usr/bin/env python3
"""Per-second segment accuracy with audit-informed equivalence (brief §9).

Scores dumped segment-list predictions (path_decode --dump, or any file in
that schema) against GT under three equivalence modes:

  strict        predicted ref within tol of GT ref,
  clone-aware   ... or of any CLONE image of GT ref (digital copies are
                content-indistinguishable — landing on one is NOT an error;
                this is the honest score, and its complement of the clone
                fraction is the unwinnable part),
  repeat-aware  ... or of any repeat image (clone or distinct take) — the
                fiber-aware analog with exact lag maps; an upper bound that
                forgives wrong-but-same-content instance picks.

Reported at ±0.25 s / ±1.0 s / ±2.0 s (the 2.0 s column is the frozen legacy
trajectory_acc tolerance and must match `path_decode --eval` strict output).

Usage:
    venvs/audio/bin/python -m alignment.looptrace.eval \
        --pred .../out/pred_<set>_hubert.json --audit .../out/audit_<set>.json \
        [--gt labeling/fixtures/bb12_ground_truth.yaml]
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

from alignment.looptrace.audit import clone_images  # noqa: E402
from alignment.looptrace.config import EVAL_V1  # noqa: E402
from alignment.path_decode import (  # noqa: E402
    _gt_pieces,
    _pieces,
    _ref_at,
)

_CLONE = frozenset({"clone"})
_REPEAT = frozenset({"clone", "distinct"})


def score_span(
    pred_segs: list[list[float]],
    row: dict,
    pairs: list[dict],
    *,
    step_s: float = 1.0,
    tolerances_s: tuple[float, ...] = (0.25, 1.0, 2.0),
) -> dict[str, dict[float, float]]:
    """{mode: {tol: fraction of sampled mix seconds correct}}."""
    s0, s1 = float(row["set_start_s"]), float(row["set_end_s"])
    out: dict[str, dict[float, float]] = {
        m: {t: 0.0 for t in tolerances_s} for m in ("strict", "clone", "repeat")
    }
    if s1 <= s0 or not pred_segs:
        return out
    gt = _gt_pieces(row)
    slope = float(row.get("tempo_ratio") or 1.0)
    pred = _pieces([(s0 + ms, rs, re) for ms, rs, re in pred_segs], s0, s1, slope)
    ts = np.arange(s0, s1, step_s)
    pr = np.array([_ref_at(pred, float(t)) for t in ts])
    gr = np.array([_ref_at(gt, float(t)) for t in ts])
    imgs_clone = [clone_images(float(g), pairs, classes=_CLONE) for g in gr]
    imgs_rep = [clone_images(float(g), pairs, classes=_REPEAT) for g in gr]
    for tol in tolerances_s:
        strict = np.abs(pr - gr) < tol
        clone = np.array(
            [
                s or any(abs(p - i) < tol for i in im)
                for s, p, im in zip(strict, pr, imgs_clone)
            ]
        )
        rep = np.array(
            [
                s or any(abs(p - i) < tol for i in im)
                for s, p, im in zip(strict, pr, imgs_rep)
            ]
        )
        out["strict"][tol] = float(strict.mean())
        out["clone"][tol] = float(clone.mean())
        out["repeat"][tol] = float(rep.mean())
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--gt", type=Path, default=None)
    args = p.parse_args(argv)
    cfg = EVAL_V1

    pred = json.loads(args.pred.read_text())
    audit = json.loads(args.audit.read_text())
    import yaml

    gt_path = args.gt or Path(pred["gt"])
    raw = {
        (str(r.get("slot_label")), round(float(r.get("set_start_s", -1)), 2)): r
        for r in yaml.safe_load(gt_path.read_text()).get("tracks", [])
    }

    rows = []
    for span in pred["spans"]:
        row = raw.get((str(span["slot_label"]), round(float(span["set_start_s"]), 2)))
        if row is None:
            print(f"  ! no GT row for slot {span['slot_label']}", file=sys.stderr)
            continue
        ta = audit["tracks"].get(span["track_id"])
        pairs = ta["pairs"] if ta else []
        sc = score_span(
            span["segs"],
            row,
            pairs,
            step_s=cfg.step_s,
            tolerances_s=cfg.tolerances_s,
        )
        rows.append((span["span_class"], sc))

    print(
        f"=== looptrace.eval ({cfg.version}) — {audit['set_id']} "
        f"pred={args.pred.name} n={len(rows)} ==="
    )
    classes = ["ALL", "linear", "multiseg", "loop", "oddratio"]
    for cls in classes:
        sel = [sc for c, sc in rows if cls == "ALL" or c == cls]
        if not sel:
            continue
        line = f"  {cls:9} n={len(sel):3} "
        for mode in ("strict", "clone", "repeat"):
            vals = " ".join(
                f"{100 * float(np.mean([s[mode][t] for s in sel])):3.0f}"
                for t in cfg.tolerances_s
            )
            line += f"  {mode}[{vals}]"
        print(line + f"   (tol {list(cfg.tolerances_s)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
