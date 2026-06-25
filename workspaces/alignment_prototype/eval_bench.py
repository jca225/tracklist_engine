#!/usr/bin/env python3
"""eval_bench — score DJ-mix alignment methods on a common metric set.

Turns "are we SOTA" into a table. Adapters yield `Sample`s (mix + candidate-track
features + ground truth); a `method` predicts per-track `(set_start_s,
tempo_ratio, score)`; metrics report the UnmixDB / André-2024 units:

  - set_start MAE  — where in the MIX each track begins (placement)
  - tempo  MAE/%   — the time-warp / speed factor (André's headline metric)
  - identity acc   — which candidate matches (closed pool; +distractors = open-set)

The synthetic adapter runs in *feature space* and needs no audio, so the whole
harness smoke-tests before UnmixDB finishes downloading.

    venvs/audio/bin/python -m workspaces.alignment_prototype.eval_bench --synthetic
    venvs/audio/bin/python -m workspaces.alignment_prototype.eval_bench \
        --unmixdb-root ~/data/unmixdb-v1.1 --max-mixes 20 --feature chroma
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.alignment_prototype.refine_ref_offsets import (  # noqa: E402
    HOP,
    SR,
    STRETCHES,
    chroma,
    correlate_window,
    detect_offset,
)


# ----------------------------------------------------------------------------- types
@dataclass(frozen=True)
class GTSpan:
    track_idx: int
    set_start_s: float
    tempo_ratio: float


@dataclass
class Sample:
    mix_id: str
    mix_feat: np.ndarray  # (D, Tm)
    track_feats: dict[int, np.ndarray]  # idx -> (D, Tk)
    gt: list[GTSpan]
    distractor_feats: dict[str, np.ndarray] = field(default_factory=dict)
    mix_path: Path | None = None  # audio (UnmixDB only) — for NMF / fingerprint
    track_paths: dict[int, Path] = field(default_factory=dict)
    distractor_paths: dict[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class Pred:
    set_start_s: float
    tempo_ratio: float
    score: float


Method = Callable[[Sample], dict[int, Pred]]


# ----------------------------------------------------------------------------- methods
def method_grid_mf(
    sample: Sample, stretches: tuple[float, ...] = STRETCHES
) -> dict[int, Pred]:
    """Our pipeline: slide each candidate track over the mix (matched filter over
    a stretch grid). detect_offset(win=track, ref=mix) -> (mix_pos, peak, stretch)."""
    out: dict[int, Pred] = {}
    for idx, tf in sample.track_feats.items():
        if tf.shape[1] < 8 or sample.mix_feat.shape[1] <= tf.shape[1]:
            out[idx] = Pred(0.0, 1.0, -1.0)
            continue
        pos, peak, st = detect_offset(tf, sample.mix_feat, stretches)
        out[idx] = Pred(pos, st, peak)
    return out


def method_grid_locked(sample: Sample) -> dict[int, Pred]:
    """Ablation: single stretch (1.0) — no warp search. Shows the cost of not
    knowing tempo, vs grid_mf which searches it."""
    return method_grid_mf(sample, stretches=(1.0,))


def method_nmf(sample: Sample) -> dict[int, Pred]:
    """André-style reference-conditioned NMF (superposition-aware). Needs audio,
    so it runs on UnmixDB samples only; returns {} in feature-only (synthetic)."""
    if sample.mix_path is None or not sample.track_paths:
        return {}
    from workspaces.alignment_prototype.nmf_baseline import recover_audio

    preds = recover_audio(sample.mix_path, sample.track_paths)
    return {
        k: Pred(p.set_start_s, p.tempo_ratio, p.gain_peak if p.present else -1.0)
        for k, p in preds.items()
    }


METHODS: dict[str, Method] = {
    "grid_mf": method_grid_mf,
    "no_warp": method_grid_locked,
    "nmf": method_nmf,
}


# ----------------------------------------------------------------------------- metrics
def score_sample(sample: Sample, preds: dict[int, Pred]) -> tuple[list[dict], float]:
    rows = []
    for sp in sample.gt:
        p = preds.get(sp.track_idx)
        if p is None:
            continue
        rows.append(
            dict(
                mix_id=sample.mix_id,
                track=sp.track_idx,
                set_start_err=abs(p.set_start_s - sp.set_start_s),
                tempo_err=abs(p.tempo_ratio - sp.tempo_ratio),
                tempo_pct=abs(p.tempo_ratio - sp.tempo_ratio)
                / max(1e-6, sp.tempo_ratio)
                * 100.0,
                peak=p.score,
            )
        )
    # identity: the true track must out-score every distractor — scored the SAME
    # way (detect_offset peak) so true vs distractor is apples-to-apples.
    id_ok = 1.0
    if sample.distractor_feats:
        dist_peaks = []
        for df in sample.distractor_feats.values():
            if df.shape[1] >= 8 and sample.mix_feat.shape[1] > df.shape[1]:
                dist_peaks.append(detect_offset(df, sample.mix_feat)[1])
        best_dist = max(dist_peaks) if dist_peaks else -2.0
        hits = sum(
            int((p := preds.get(sp.track_idx)) is not None and p.score > best_dist)
            for sp in sample.gt
        )
        id_ok = hits / max(1, len(sample.gt))
    return rows, id_ok


def run(samples: list[Sample], method: Method, label: str) -> pd.DataFrame:
    rows, ids = [], []
    for s in samples:
        preds = method(s)
        r, idok = score_sample(s, preds)
        rows.extend(r)
        ids.append(idok)
    df = pd.DataFrame(rows)
    df.attrs["label"] = label
    df.attrs["identity_acc"] = float(np.mean(ids)) if ids else float("nan")
    return df


def summary(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = []
    for label, df in dfs.items():
        if len(df) == 0:
            out.append(dict(method=label, n=0))
            continue
        out.append(
            dict(
                method=label,
                n=len(df),
                set_start_MAE_s=round(df.set_start_err.mean(), 2),
                set_start_med_s=round(df.set_start_err.median(), 2),
                set_start_exact2s_pct=round(100 * (df.set_start_err < 2).mean(), 0),
                tempo_MAE=round(df.tempo_err.mean(), 4),
                tempo_pct=round(df.tempo_pct.median(), 2),
                identity_acc=round(df.attrs.get("identity_acc", float("nan")), 3),
            )
        )
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- identity
# Method-agnostic identity: score true tracks AND distractors with the SAME
# backbone, rank@1 = true track out-scores every distractor. Fixes the earlier
# apples-to-oranges bug and lets fingerprint vs chroma be compared honestly.
def _id_scores_chroma(sample: Sample):
    cands = {f"t{k}": tf for k, tf in sample.track_feats.items()}
    cands.update(sample.distractor_feats)
    scores = {}
    for name, f in cands.items():
        ok = f.shape[1] >= 8 and sample.mix_feat.shape[1] > f.shape[1]
        scores[name] = detect_offset(f, sample.mix_feat)[1] if ok else -1.0
    return scores, {f"t{k}" for k in sample.track_feats}


_FP_CACHE: dict[str, dict] = {}


def _fp_hashes(path: Path) -> dict:
    key = str(path)
    if key not in _FP_CACHE:
        import librosa
        from workspaces.alignment_prototype.landmark_fp import constellation, hashes

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, _ = librosa.load(str(path), sr=SR, mono=True)
        _FP_CACHE[key] = hashes(*constellation(y))
    return _FP_CACHE[key]


def _id_scores_fp(sample: Sample):
    import librosa
    from workspaces.alignment_prototype.landmark_fp import (
        _vote_histogram,
        constellation,
        hashes,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        my, _ = librosa.load(str(sample.mix_path), sr=SR, mono=True)
    hm = hashes(*constellation(my))  # mix hashed ONCE
    cands = {f"t{k}": p for k, p in sample.track_paths.items()}
    cands.update(sample.distractor_paths)
    scores = {}
    for name, p in cands.items():
        votes = _vote_histogram(hm, _fp_hashes(p))
        scores[name] = float(max(votes.values())) if votes else 0.0
    return scores, {f"t{k}" for k in sample.track_paths}


def run_identity(samples: list[Sample], mode: str) -> float:
    scorer = {"chroma": _id_scores_chroma, "fingerprint": _id_scores_fp}[mode]
    hits = tot = 0
    for s in samples:
        if mode == "fingerprint" and (s.mix_path is None or not s.distractor_paths):
            continue
        if mode == "chroma" and not s.distractor_feats:
            continue
        scores, truth = scorer(s)
        best_dist = max((v for n, v in scores.items() if n not in truth), default=-1e9)
        for t in truth:
            tot += 1
            hits += int(scores.get(t, -1e9) > best_dist)
    return hits / max(1, tot)


# ----------------------------------------------------------------------------- adapters
def synthetic_samples(
    n: int = 6, D: int = 12, seed: int = 0, with_distractors: bool = True
) -> list[Sample]:
    """Feature-space mini-mixes with known (set_start, tempo). No audio/deps."""
    rng = np.random.default_rng(seed)
    out = []
    for m in range(n):
        Tm = 2400
        mix = rng.random((D, Tm)).astype(np.float32) * 0.15
        tfs, gt = {}, []
        for idx in range(3):
            tlen = int(rng.integers(350, 500))
            tf = rng.random((D, tlen)).astype(np.float32)
            stretch = float(rng.choice([0.92, 0.96, 1.0, 1.04, 1.08]))
            mm = int(round(tlen * stretch))
            start_f = int(rng.integers(0, max(1, Tm - mm - 1)))
            idxs = np.clip((np.arange(mm) / stretch).astype(int), 0, tlen - 1)
            mix[:, start_f : start_f + mm] += tf[:, idxs]  # dominant additive
            tfs[idx] = tf
            gt.append(GTSpan(idx, start_f * HOP / SR, stretch))
        dist = {}
        if with_distractors:
            for k in range(5):
                dist[f"d{k}"] = rng.random((D, int(rng.integers(350, 500)))).astype(
                    np.float32
                )
        out.append(Sample(f"synth{m}", mix, tfs, gt, dist))
    return out


def _feature_fn(feature: str) -> Callable[[Path], np.ndarray]:
    import librosa

    def fn(path: Path) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, _ = librosa.load(str(path), sr=SR, mono=True)
        if feature == "chroma":
            return chroma(y)
        if feature == "hubert":
            from workspaces.section_hsmm.similarity_probe import _hubert

            return _hubert(y, 9)
        raise ValueError(feature)

    return fn


def unmixdb_samples(
    root: Path,
    feature: str,
    max_mixes: int | None,
    good_only: bool,
    n_distractors: int = 0,
) -> list[Sample]:
    from workspaces.alignment_prototype.external.unmixdb import (
        discover_root,
        good_mix_ids,
        load_mix,
    )

    fn = _feature_fn(feature)
    # stride across the sorted label list so warp variants (none/resample/stretch)
    # all appear in a small slice — iter_mixes' head-take gives only 'none'.
    base = discover_root(root)
    labels = sorted(base.rglob("*.labels.txt"))
    good = good_mix_ids(base) if good_only else None
    if good is not None:
        labels = [p for p in labels if p.name.replace(".labels.txt", "") in good]
    if max_mixes and len(labels) > max_mixes:
        step = len(labels) / max_mixes
        labels = [labels[int(i * step)] for i in range(max_mixes)]
    mixes = []
    for lp in labels:
        try:
            mixes.append(load_mix(lp, root=base))
        except Exception:  # noqa: BLE001
            continue
    out = []
    dpool: list[np.ndarray] = []
    dpool_paths: list[Path] = []
    for mx in mixes:
        try:
            mf = fn(mx.mix_audio)
            tfs = {idx: fn(p) for idx, p in mx.track_audio.items()}
        except Exception as e:  # noqa: BLE001
            print(f"  skip {mx.mix_id}: {e}")
            continue
        gt = [GTSpan(sp.track_idx, sp.set_start_s, sp.tempo_ratio) for sp in mx.spans]
        dist, dist_paths = {}, {}
        if n_distractors and dpool:
            for k, (df, dp) in enumerate(
                zip(dpool[:n_distractors], dpool_paths[:n_distractors])
            ):
                dist[f"d{k}"] = df
                dist_paths[f"d{k}"] = dp
        out.append(
            Sample(
                mx.mix_id,
                mf,
                tfs,
                gt,
                dist,
                mix_path=mx.mix_audio,
                track_paths=dict(mx.track_audio),
                distractor_paths=dist_paths,
            )
        )
        dpool.extend(tfs.values())
        dpool_paths.extend(mx.track_audio.values())
    return out


# ----------------------------------------------------------------------------- cli
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--synthetic", action="store_true", help="feature-space smoke, no data"
    )
    p.add_argument("--unmixdb-root", type=Path)
    p.add_argument("--feature", default="chroma", choices=["chroma", "hubert"])
    p.add_argument("--max-mixes", type=int, default=None)
    p.add_argument("--good-only", action="store_true")
    p.add_argument("--n-distractors", type=int, default=0)
    p.add_argument("--methods", default="grid_mf,no_warp")
    p.add_argument(
        "--identity",
        action="store_true",
        help="also run the chroma-vs-fingerprint identity benchmark",
    )
    args = p.parse_args(argv)

    if args.synthetic:
        samples = synthetic_samples()
        print(f"synthetic: {len(samples)} mini-mixes (feature space)")
    elif args.unmixdb_root:
        print(f"loading UnmixDB from {args.unmixdb_root} (feature={args.feature}) …")
        samples = unmixdb_samples(
            args.unmixdb_root,
            args.feature,
            args.max_mixes,
            args.good_only,
            args.n_distractors,
        )
        print(
            f"loaded {len(samples)} mixes, {sum(len(s.gt) for s in samples)} GT spans"
        )
    else:
        p.error("pass --synthetic or --unmixdb-root")

    if not samples:
        print("no samples.")
        return 1

    dfs = {}
    for name in args.methods.split(","):
        name = name.strip()
        if name not in METHODS:
            print(f"unknown method {name}")
            continue
        dfs[name] = run(samples, METHODS[name], name)

    print("\n=== eval_bench (placement / warp) ===")
    print(summary(dfs).to_string(index=False))

    if args.identity:
        print("\n=== identity (rank@1: true track out-scores all distractors) ===")
        for mode in ("chroma", "fingerprint"):
            try:
                acc = run_identity(samples, mode)
                print(f"  {mode:12s}: {100 * acc:5.1f}%")
            except Exception as e:  # noqa: BLE001
                print(f"  {mode:12s}: failed ({e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
