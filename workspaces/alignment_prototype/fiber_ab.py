"""Fiber A/B scorer — score a fixed prediction dump under different fiber
builders, holding the frozen metric (path_decode.trajectory_acc) constant.

Step 0 (headroom gate) + Step 2 (HuBERT vs multi A/B) of the multi-axis vocal
fibers plan. The dump (from `path_decode --dump`) already carries per-span
`segs` (mix-relative predicted segments) + `ref_audio`; we join each span to its
GT row and re-score with a chosen equivalence source:

  none   — strict only (sanity: must reproduce looptrace.eval strict)
  hubert — ref_fibers.compute_fibers (today's builder)
  multi  — ref_fibers.compute_fibers_multi (lyrics + HuBERT, added in Step 1)

Never tunes on GT: fibers only affect the *credit* rule, not the predictions.

    venvs/audio/bin/python -m workspaces.alignment_prototype.fiber_ab \
        --pred looptrace/out/pred_1fsnxchk_hubert.json \
        --gt labeling/fixtures/bb12_ground_truth.yaml --fiber hubert
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from workspaces.alignment_prototype.path_decode import (
    FPS,
    _ensure_feat,
    _span_class,
    trajectory_acc,
)

TOLS = (0.25, 1.0, 2.0)


def _load_gt_rows(gt_path: Path) -> list[dict]:
    return list(yaml.safe_load(gt_path.read_text()).get("tracks", []))


def _fiber_builder(mode: str, hubert_layer: int):
    """Return f(ref_audio) -> (labels, label_hz) | None, memoized per ref_audio."""
    if mode == "none":
        return lambda ref_audio: None

    cache: dict[str, tuple] = {}

    def build(ref_audio: str | None):
        if ref_audio is None:
            return None
        if ref_audio not in cache:
            hf = np.load(_ensure_feat(ref_audio, ref_audio, "hubert", hubert_layer))
            if mode == "hubert":
                from workspaces.alignment_prototype.ref_fibers import compute_fibers

                cache[ref_audio] = compute_fibers(hf, FPS, audio_path=ref_audio)
            elif mode == "multi":
                from workspaces.alignment_prototype.ref_fibers import (
                    compute_fibers_multi,
                )

                cache[ref_audio] = compute_fibers_multi(hf, FPS, audio_path=ref_audio)
            else:
                raise ValueError(f"unknown fiber mode {mode!r}")
        return cache[ref_audio]

    return build


def _match_gt(span: dict, gt_by_tid: dict[str, list[dict]]) -> dict | None:
    rows = gt_by_tid.get(str(span.get("track_id")))
    if not rows:
        return None
    ss = float(span["set_start_s"])
    return min(rows, key=lambda r: abs(float(r["set_start_s"]) - ss))


def score(pred_path: Path, gt_path: Path, mode: str, stems: set[str], layer: int):
    dump = json.loads(pred_path.read_text())
    spans = dump["spans"]
    gt_by_tid: dict[str, list[dict]] = defaultdict(list)
    for r in _load_gt_rows(gt_path):
        if r.get("track_id"):
            gt_by_tid[str(r["track_id"])].append(r)

    build = _fiber_builder(mode, layer)
    # accumulate per (class, tol): list of per-span accuracy
    strict_by: dict[tuple[str, float], list[float]] = defaultdict(list)
    fiber_by: dict[tuple[str, float], list[float]] = defaultdict(list)
    n_by: dict[str, int] = defaultdict(int)
    unmatched = 0

    for s in spans:
        if (s.get("claimed_stem") or "regular") not in stems:
            continue
        g = _match_gt(s, gt_by_tid)
        if g is None:
            unmatched += 1
            continue
        cls = _span_class(g)
        n_by[cls] += 1
        n_by["ALL"] += 1
        fib = build(s.get("ref_audio"))
        for tol in TOLS:
            strict, _n, facc = trajectory_acc(s["segs"], g, tol=tol, fiber=fib)
            for c in (cls, "ALL"):
                strict_by[(c, tol)].append(strict)
                fiber_by[(c, tol)].append(facc)
    return strict_by, fiber_by, n_by, unmatched


def _fmt(by, cls, n_by):
    return "  ".join(
        f"{int(round(100 * np.mean(by[(cls, t)]))):3d}" if by[(cls, t)] else "  -"
        for t in TOLS
    )


def audit_diag(
    pred_path: Path,
    gt_path: Path,
    stems: set[str],
    layer: int,
    audit_path: Path,
    tol: float = 2.0,
):
    """Step-3 guardrail: over acappella span-seconds, split multi's NEW credits
    (multi same-fiber but NOT hubert same-fiber, on strict-miss seconds) into
    audit-confirmed repeats (legit / world-3 win) vs audit-denied (false merge).
    Also headroom recall = of the distinct-take repeat-seconds hubert missed,
    what fraction multi now credits."""
    from collections import Counter

    from workspaces.alignment_prototype.path_decode import (
        _gt_pieces,
        _pieces,
        _ref_at,
    )
    from workspaces.alignment_prototype.ref_fibers import (
        compute_fibers,
        compute_fibers_multi,
        same_fiber,
    )
    from workspaces.alignment_prototype.looptrace.audit import clone_images

    REPEAT = frozenset({"clone", "distinct"})
    spans = json.loads(pred_path.read_text())["spans"]
    gt_by_tid: dict[str, list[dict]] = defaultdict(list)
    for r in _load_gt_rows(gt_path):
        if r.get("track_id"):
            gt_by_tid[str(r["track_id"])].append(r)
    audit = json.loads(audit_path.read_text())["tracks"]

    fibcache: dict[str, tuple] = {}

    def fibs(ref: str):
        if ref not in fibcache:
            hf = np.load(_ensure_feat(ref, ref, "hubert", layer))
            fibcache[ref] = (
                compute_fibers(hf, FPS, audio_path=ref),
                compute_fibers_multi(hf, FPS, audio_path=ref),
            )
        return fibcache[ref]

    tal: Counter = Counter()
    for s in spans:
        if (s.get("claimed_stem") or "regular") not in stems:
            continue
        g = _match_gt(s, gt_by_tid)
        if g is None:
            continue
        s0, s1 = float(g["set_start_s"]), float(g["set_end_s"])
        slope = float(g.get("tempo_ratio") or 1.0)
        pred = _pieces([(s0 + ms, rs, re) for (ms, rs, re) in s["segs"]], s0, s1, slope)
        gtp = _gt_pieces(g)
        (hl, hz), (ml, mz) = fibs(s["ref_audio"])
        ta = audit.get(str(s.get("track_id")))
        pairs = ta["pairs"] if ta else []
        for t in np.arange(s0, s1, 1.0):
            pr, gr = _ref_at(pred, t), _ref_at(gtp, t)
            if abs(pr - gr) < tol:
                continue  # strict hit
            hub = same_fiber(hl, hz, float(pr), float(gr))
            mul = same_fiber(ml, mz, float(pr), float(gr))
            imgs = clone_images(float(gr), pairs, classes=REPEAT)
            audit_rep = any(abs(pr - im) < tol for im in imgs)
            far = abs(pr - gr) > 8.0  # genuine instance swap, not within-tile proximity
            if audit_rep and not hub:
                tal["world3_avail"] += 1
                if mul:
                    tal["world3_captured"] += 1
            if mul and not hub:
                tal["multi_gain_legit" if audit_rep else "multi_gain_false"] += 1
                if far:
                    tal["far_legit" if audit_rep else "far_false"] += 1
    legit, false = tal["multi_gain_legit"], tal["multi_gain_false"]
    gain = legit + false
    print(f"=== audit guardrail pred={pred_path.name} stems={stems} tol={tol} ===")
    print(
        f"  multi NEW credits (multi&~hubert): {gain}  "
        f"legit(audit-repeat)={legit}  false-merge={false}"
    )
    print(f"  FALSE-MERGE RATE: {(false / gain if gain else 0):.1%}  (target ~0)")
    print(
        f"  distinct-take headroom seconds (audit-repeat & hubert-missed): "
        f"{tal['world3_avail']}"
    )
    print(
        f"  captured by multi: {tal['world3_captured']}  "
        f"RECALL: {(tal['world3_captured'] / tal['world3_avail'] if tal['world3_avail'] else 0):.1%}"
    )
    fl, ff = tal["far_legit"], tal["far_false"]
    fgain = fl + ff
    print(
        f"  FAR-only (|pred-gt|>8s, genuine instance swaps): {fgain}  "
        f"legit={fl}  false={ff}  "
        f"FAR FALSE-MERGE RATE: {(ff / fgain if fgain else 0):.1%}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred", required=True, type=Path)
    p.add_argument("--gt", required=True, type=Path)
    p.add_argument("--fiber", default="hubert", choices=("none", "hubert", "multi"))
    p.add_argument("--stems", default="acappella")
    p.add_argument("--hubert-layer", type=int, default=9)
    p.add_argument("--audit", type=Path, help="audit JSON -> run false-merge guardrail")
    args = p.parse_args()

    stems = set(args.stems.split(","))
    if args.audit:
        audit_diag(args.pred, args.gt, stems, args.hubert_layer, args.audit)
        return 0
    strict_by, fiber_by, n_by, unmatched = score(
        args.pred, args.gt, args.fiber, stems, args.hubert_layer
    )
    print(
        f"=== fiber_ab pred={args.pred.name} fiber={args.fiber} "
        f"stems={args.stems} (tol {list(TOLS)}) ==="
    )
    if unmatched:
        print(f"  [warn] {unmatched} spans had no GT track match")
    for cls in ("ALL", "linear", "multiseg", "loop", "oddratio"):
        if not n_by.get(cls):
            continue
        print(
            f"  {cls:9s} n={n_by[cls]:3d}   "
            f"strict[{_fmt(strict_by, cls, n_by)}]   "
            f"fiber[{_fmt(fiber_by, cls, n_by)}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
