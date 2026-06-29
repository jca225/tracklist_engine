"""WS1 read-only eval — does curve *precision* predict correctness, and does
precision-weighted channel selection beat raw-peak / fixed-priority?

Testbed: BB12 acappella **ref_start** (the worst, un-owned axis). Two channels —
chroma and HuBERT-L9 — run on the SAME `detect_offset_curve` harness against each
span's vocals stem, so the only variable is the feature + the arbiter. GT comes
from the exported YAML (`labeling/fixtures/bb12_ground_truth.yaml`), NOT the live
seeded `.als`.

Three questions, three answers printed:
  1. CALIBRATION — AUC of each precision proxy (peak/margin/z/prominence)
     predicting per-prediction correctness. WS1 kill gate: best proxy AUC > 0.60.
  2. ARBITER — median err + <5s% for: fixed-priority(hubert) | raw-peak |
     precision(prominence) | oracle. WS1 wins if precision >= raw-peak and is
     not worse than fixed-priority.
  3. ABSTENTION — coverage vs error-on-kept as the precision floor rises (the
     reliability collapse that the Bayesian arbiter turns into a no-decision).

Read-only: imports feature extractors + GT loader; writes nothing but a JSON
summary under neuro/out/. Touches no harness file.

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.neuro.precision_fusion_eval \
        [--features chroma,hubert] [--limit 0] [--tol-s 5] [--max-win-s 15]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from labeling.ground_truth import schema
from workspaces.alignment_prototype.neuro.precision import (
    SR,
    detect_offset_curve,
    precision_from_curve,
    select_by,
)
from workspaces.alignment_prototype.refine_ref_offsets import STRETCHES, chroma

PROXIES = ("peak", "margin", "z", "prominence")
SET_DIR = Path.home() / "aligning" / "1fsnxchk__Two Friends - Big Bootie Mix Volume 12"
GT_YAML = _REPO / "labeling" / "fixtures" / "bb12_ground_truth.yaml"


def _load(
    path: Path, *, offset: float = 0.0, duration: float | None = None
) -> np.ndarray:
    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(
            str(path), sr=SR, mono=True, offset=offset, duration=duration
        )
    return y


def _feat(name: str, y: np.ndarray, layer: int) -> np.ndarray:
    if name == "chroma":
        return chroma(y)
    from workspaces.section_hsmm.similarity_probe import _hubert

    return _hubert(y, layer)


def _auc(scores: list[float], labels: list[int]) -> float:
    """Mann-Whitney AUC of `scores` ranking the positive class (label==1)."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s = np.asarray(scores)
    for v in np.unique(s):
        idx = np.where(s == v)[0]
        if idx.size > 1:
            ranks[idx] = ranks[idx].mean()
    rsum = ranks[np.asarray(labels) == 1].sum()
    npos, nneg = len(pos), len(neg)
    return (rsum - npos * (npos + 1) / 2) / (npos * nneg)


def _pct(errs: list[float], thr: float) -> float:
    return 100.0 * sum(1 for e in errs if e <= thr) / len(errs) if errs else 0.0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", default="chroma,hubert")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--tol-s", type=float, default=5.0, help="err<=tol => correct")
    p.add_argument("--max-win-s", type=float, default=15.0)
    p.add_argument("--min-win-s", type=float, default=4.0)
    p.add_argument("--hubert-layer", type=int, default=9)
    p.add_argument("--linear-only", action="store_true", help="drop multiseg/loop GT")
    args = p.parse_args(argv)

    features = tuple(f.strip() for f in args.features.split(",") if f.strip())

    r = schema.load(GT_YAML)
    if not r.is_ok():
        print(f"GT load failed: {r.error}", file=sys.stderr)
        return 2
    gt = r.value
    manifest = json.loads((SET_DIR / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}

    mix_vocals = SET_DIR / "mix_vocals.flac"
    if not mix_vocals.is_file():
        print(f"missing {mix_vocals}", file=sys.stderr)
        return 2
    print(f"loading mix_vocals ({features=})…")
    mix: dict[str, np.ndarray] = {}  # feature -> full mix_vocals features (lazy)
    mixy = _load(mix_vocals)

    aca = [t for t in gt.tracks if t.claimed_stem == "acappella"]
    ref_cache: dict[tuple[str, str], np.ndarray] = {}

    # rows: per span, per feature -> (pred, err, Precision, nseg)
    per_err: dict[str, list[float]] = {f: [] for f in features}
    proxy_scores: dict[str, list[float]] = {k: [] for k in PROXIES}
    proxy_labels: list[int] = []  # correctness label aligned to proxy_scores
    span_rows: list[dict] = []
    n = skipped = 0

    for t in aca:
        if args.limit and n >= args.limit:
            break
        if args.linear_only and t.ref_segments:
            continue
        dur = min(args.max_win_s, t.set_end_s - t.set_start_s)
        if dur < args.min_win_s:
            skipped += 1
            continue
        mt = by_tid.get(t.track_id)
        voc = (mt.get("stems") or {}).get("vocals") if mt else None
        if not voc or not Path(voc).is_file():
            skipped += 1
            continue
        i0 = int(t.set_start_s * SR)
        gtv = t.ref_start_s
        cands: list[tuple[str, float, object]] = []
        row = {
            "slot": t.slot_label,
            "gt": gtv,
            "nseg": len(t.ref_segments),
            "feats": {},
        }
        for f in features:
            ry = ref_cache.get((t.track_id, f))
            if ry is None:
                ry = _feat(f, _load(Path(voc)), args.hubert_layer)
                ref_cache[(t.track_id, f)] = ry
            mw = mixy[i0 : i0 + int(dur * SR)]
            wf = _feat(f, mw, args.hubert_layer)
            if wf.shape[1] >= ry.shape[1]:  # window longer than ref → undefined
                continue
            pred, peak, st, curve = detect_offset_curve(wf, ry, STRETCHES)
            if curve.size < 4:
                continue
            prec = precision_from_curve(curve)
            err = abs(pred - gtv)
            per_err[f].append(err)
            cands.append((f, pred, prec))
            correct = int(err <= args.tol_s)
            for k in PROXIES:
                proxy_scores[k].append(prec.proxies[k])
            proxy_labels.append(correct)
            row["feats"][f] = {"pred": pred, "err": err, **prec.proxies}
        if cands:
            row["cands"] = cands
            span_rows.append(row)
            n += 1

    print(f"\n=== {n} acappella spans scored ({skipped} skipped) ===")
    print(f"{'feature':9}{'median':>8}{'mean':>8}{'<2s%':>7}{'<5s%':>7}{'<10s%':>7}")
    for f in features:
        e = per_err[f]
        if e:
            print(
                f"{f:9}{np.median(e):8.1f}{np.mean(e):8.1f}"
                f"{_pct(e, 2):7.0f}{_pct(e, 5):7.0f}{_pct(e, 10):7.0f}"
            )

    # (1) CALIBRATION — AUC of each proxy predicting correctness
    print(
        f"\n--- (1) precision calibration: AUC(proxy -> err<= {args.tol_s:g}s) "
        f"over {len(proxy_labels)} predictions ({sum(proxy_labels)} correct) ---"
    )
    aucs = {k: _auc(proxy_scores[k], proxy_labels) for k in PROXIES}
    for k in PROXIES:
        print(f"  {k:11} AUC={aucs[k]:.3f}")
    best_proxy = max(
        (k for k in PROXIES if not np.isnan(aucs[k])),
        key=lambda k: aucs[k],
        default="prominence",
    )
    gate = (not np.isnan(aucs[best_proxy])) and aucs[best_proxy] > 0.60
    print(
        f"  best={best_proxy} ({aucs[best_proxy]:.3f})  WS1 gate (>0.60): "
        f"{'PASS' if gate else 'FAIL'}"
    )

    # (2) ARBITER — selection strategies on per-span candidate sets
    if len(features) >= 2:
        print("\n--- (2) arbiter: per-span channel selection (err vs GT ref_start) ---")
        strat_err: dict[str, list[float]] = {
            "fixed(hubert)": [],
            "raw-peak": [],
            f"precision({best_proxy})": [],
            "oracle": [],
        }
        prio = "hubert" if "hubert" in features else features[0]
        for row in span_rows:
            cands = row["cands"]
            # fixed-priority: the routed invariant axis (hubert for acappella)
            fx = next((c for c in cands if c[0] == prio), cands[0])
            strat_err["fixed(hubert)"].append(abs(fx[1] - row["gt"]))
            rp = select_by(cands, "peak")
            strat_err["raw-peak"].append(abs(rp[1] - row["gt"]))
            pw = select_by(cands, best_proxy)
            strat_err[f"precision({best_proxy})"].append(abs(pw[1] - row["gt"]))
            strat_err["oracle"].append(min(abs(c[1] - row["gt"]) for c in cands))
        print(f"{'strategy':22}{'median':>8}{'<2s%':>7}{'<5s%':>7}{'<10s%':>7}")
        for name, e in strat_err.items():
            print(
                f"{name:22}{np.median(e):8.1f}{_pct(e, 2):7.0f}"
                f"{_pct(e, 5):7.0f}{_pct(e, 10):7.0f}"
            )

    # (3) ABSTENTION — coverage vs error-on-kept as the precision floor rises
    print(
        f"\n--- (3) abstention: keep top-precision channel per span, "
        f"floor on {best_proxy} ---"
    )
    kept = []  # (best_proxy_value, err_of_selected)
    for row in span_rows:
        sel = select_by(row["cands"], best_proxy)
        kept.append((sel[2].proxies[best_proxy], abs(sel[1] - row["gt"])))
    kept.sort(reverse=True)  # high precision first
    vals = [v for v, _ in kept]
    if vals:
        print(f"{'floor':>8}{'coverage':>10}{'kept median':>13}{'kept <5s%':>11}")
        for q in (0.0, 0.25, 0.5, 0.75):
            thr = float(np.quantile(vals, q))
            keep = [e for v, e in kept if v >= thr]
            cov = 100.0 * len(keep) / len(kept)
            print(f"{thr:8.2f}{cov:9.0f}%{np.median(keep):13.1f}{_pct(keep, 5):10.0f}%")

    out = SET_DIR.name
    summary = {
        "set": "1fsnxchk",
        "n_spans": n,
        "features": list(features),
        "tol_s": args.tol_s,
        "auc": {k: (None if np.isnan(v) else float(v)) for k, v in aucs.items()},
        "best_proxy": best_proxy,
        "gate_pass": bool(gate),
        "per_feature_median": {
            f: float(np.median(per_err[f])) if per_err[f] else None for f in features
        },
    }
    out_path = Path(__file__).resolve().parent / "out" / "ws1_precision_eval.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
