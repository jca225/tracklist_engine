"""BB12 sanity check + learned ranker: do cleanliness features rank the
human-verified online acappella above the vocal stem we'd produce by separation,
and can a small pairwise model *learn* the right weighting?

Pairs are discovered on disk:
  WINNER (online, human-verified): tracks/NNN__<song> (Acappella) [bpm KK].m4a
  LOSER  (produced):               stems/<full original song>/vocals.flac

Two reports:
  1. per-feature rank agreement (which single signals track the human preference)
  2. a Bradley-Terry logistic over standardised feature *differences* — the
     learned weights say which features matter and in which direction, and
     leave-one-pair-out accuracy says whether the combined model generalises.
     Symmetric augmentation ((+Δ,1) and (-Δ,0)) makes this a proper pairwise fit
     with no intercept. n is small (~10): read weights as direction, LOO as a
     coarse generalisation check — not p-values.

Features are cached by (path, mtime) so refitting the ranker is instant.

Run:
  venvs/audio/bin/python workspaces/separation_qa/bb12_pair_eval.py
"""

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import numpy as np
from cleanliness_features import ORIENTATION, extract

BB = Path.home() / "aligning/1fsnxchk__Two Friends - Big Bootie Mix Volume 12"
HERE = Path(__file__).resolve().parent
OUT = HERE / "bb12_pair_eval.json"
CACHE = HERE / "_feat_cache.json"

FEATS = list(ORIENTATION)


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\[[^\]]*\]", "", s)
    s = re.sub(r"\((acappella|instrumental[^)]*|extended mix|studio acapella)\)", "", s)
    s = re.sub(r"^\d+[a-z]?\d*__", "", s)
    s = re.sub(r"\.m4a$|\.asd$", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


def discover_pairs() -> list[tuple[str, Path, Path]]:
    winners: dict[str, Path] = {}
    for p in glob.glob(str(BB / "tracks" / "*Acappella*bpm*.m4a")):
        winners[_norm(os.path.basename(p))] = Path(p)
    losers: dict[str, Path] = {}
    for d in (BB / "stems").iterdir():
        if not d.is_dir():
            continue
        n = d.name.lower()
        if "acappella" in n or "instrumental" in n:
            continue
        v = d / "vocals.flac"
        if v.exists():
            losers.setdefault(_norm(d.name), v)
    keys = sorted(set(winners) & set(losers))
    return [(k, winners[k], losers[k]) for k in keys]


def _cached_features(path: Path, cache: dict) -> dict[str, float]:
    key = str(path)
    mtime = path.stat().st_mtime
    hit = cache.get(key)
    if hit and hit.get("mtime") == mtime:
        return hit["feats"]
    feats = extract(path).as_dict()
    cache[key] = {"mtime": mtime, "feats": feats}
    return feats


def _fit_bt(deltas: np.ndarray, n_iter: int = 2000, lr: float = 0.3, l2: float = 1.0):
    """Logistic on standardised feature diffs, symmetric so no intercept.
    deltas: (n_pairs, n_feat) of (online - produced). Target is 1 for every row
    (online wins) and 0 for the mirror -deltas. Returns weight vector."""
    # Scale by std ONLY — do NOT centre. With target always "online wins" and no
    # intercept, the discriminative signal is the *mean* of the diffs; centring
    # would delete it and pin w at 0.
    mu = np.zeros(deltas.shape[1])
    sd = deltas.std(axis=0) + 1e-9
    X = deltas / sd
    w = np.zeros(X.shape[1])
    for _ in range(n_iter):
        # rows: X (y=1) and -X (y=0); gradient simplifies to using X with p at +X
        p = 1.0 / (1.0 + np.exp(-(X @ w)))
        grad = X.T @ (p - 1.0) / len(X) + l2 * w / len(X)
        w -= lr * grad
    return w, mu, sd


def _loo_accuracy(deltas: np.ndarray) -> float:
    correct = 0
    n = len(deltas)
    for i in range(n):
        train = np.delete(deltas, i, axis=0)
        w, mu, sd = _fit_bt(train)
        x = (deltas[i] - mu) / sd
        correct += int((x @ w) > 0)
    return correct / n


def main() -> int:
    pairs = discover_pairs()
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    print(f"== BB12 cleanliness pair-eval :: {len(pairs)} pairs ==\n")

    wins = {f: 0 for f in FEATS}
    deltas = []
    rows = []
    for key, win_path, lose_path in pairs:
        w = _cached_features(win_path, cache)
        l = _cached_features(lose_path, cache)
        # signed delta in the "higher == cleaner" convention via orientation
        delta = np.array([ORIENTATION[f] * (w[f] - l[f]) for f in FEATS])
        deltas.append(delta)
        votes = 0
        per_feat = {}
        for j, f in enumerate(FEATS):
            online_cleaner = bool(delta[j] > 0)
            wins[f] += int(online_cleaner)
            votes += 1 if online_cleaner else -1
            per_feat[f] = {
                "online": round(float(w[f]), 3),
                "produced": round(float(l[f]), 3),
                "online_wins": online_cleaner,
            }
        rows.append(
            {
                "song": key,
                "votes": votes,
                "online_wins_naive": bool(votes > 0),
                "features": per_feat,
            }
        )
        flag = "OK " if votes > 0 else "XX "
        print(
            f"{flag}{key:38s} naive_vote={votes:+d}  "
            f"lowendΔ={ORIENTATION['lowend_ratio_db'] * (w['lowend_ratio_db'] - l['lowend_ratio_db']):+5.1f}  "
            f"hf16kΔ={w['hf16k_ratio_db'] - l['hf16k_ratio_db']:+5.1f}"
        )

    CACHE.write_text(json.dumps(cache))
    deltas = np.array(deltas)
    n = len(pairs)

    print(f"\n-- per-feature: fraction where ONLINE (verified) is cleaner (n={n}) --")
    for f in FEATS:
        print(f"  {f:18s} {wins[f]}/{n}  ({wins[f] / n:.0%})")
    # equal vote on z-scored diffs (raw sum is dominated by Hz-scaled rolloff)
    zsum = (deltas / (deltas.std(axis=0) + 1e-9)).sum(axis=1)
    naive = int((zsum > 0).sum())
    print(f"  {'EQUAL z-vote':18s} {naive}/{n}  ({naive / n:.0%})")

    # learned ranker
    w_full, mu, sd = _fit_bt(deltas)
    train_acc = int(((deltas - mu) / sd @ w_full > 0).sum()) / n
    loo = _loo_accuracy(deltas)
    print(f"\n-- learned Bradley-Terry ranker (standardised diffs, n={n}) --")
    print(f"  train accuracy   {train_acc:.0%}   leave-one-out {loo:.0%}")
    order = np.argsort(-np.abs(w_full))
    print("  weights (|·| sorted; sign>0 => feature's orientation holds):")
    for j in order:
        prior = "+1" if ORIENTATION[FEATS[j]] > 0 else "-1"
        print(f"    {FEATS[j]:18s} {w_full[j]:+.2f}   (hand prior {prior})")

    OUT.write_text(
        json.dumps(
            {
                "n": n,
                "per_feature_online_win": {f: wins[f] for f in FEATS},
                "naive_vote_online_win": naive,
                "ranker": {
                    "train_acc": train_acc,
                    "loo_acc": loo,
                    "weights": {FEATS[j]: float(w_full[j]) for j in range(len(FEATS))},
                },
                "pairs": rows,
            },
            indent=2,
        )
    )
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
