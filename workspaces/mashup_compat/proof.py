"""BB12 mashup-compatibility proof: does separate-stem MERT carry the signal?

Core bet ([[project_hrm_text_taste_pretrain]]): if stem-MERT embeddings encode
mashup compatibility, then real (bed instrumental, payload acappella) pairs should
be separable from random non-co-occurring pairs — and we want to know AT WHICH
MERT LAYER (the empirical layer question the all-layer embed keeps open).

Two scorers per layer, both on per-layer L2-normalized vectors:
  - cosine(bed, payload)           — parameter-free, zero overfit risk
  - logistic probe on bed*payload  — GROUPED CV by payload, so it can't memorize
                                      "this acappella is always positive" (a few
                                      payloads host-dominate the positives)

A layer whose cosine OR grouped-probe AUC sits well above 0.5 = signal exists.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from workspaces.mashup_compat.embed import EMB_PATH
from workspaces.mashup_compat.pairs import MashupPair, extract_pairs


def _l2n(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _assemble(cache: dict, pairs: list[MashupPair]):
    """Return per-layer normalized (bed, payload) stacks, labels, and payload groups,
    keeping only pairs whose BOTH stems are embedded."""
    usable = [p for p in pairs
              if (p.bed.track_id, "bed") in cache and (p.payload.track_id, "payload") in cache]
    beds = np.stack([cache[(p.bed.track_id, "bed")].astype(np.float32) for p in usable])       # (N,L,D)
    pays = np.stack([cache[(p.payload.track_id, "payload")].astype(np.float32) for p in usable])
    y = np.array([1 if p.positive else 0 for p in usable])
    groups = np.array([p.payload.track_id for p in usable])
    return usable, beds, pays, y, groups


def _grouped_probe_auc(b: np.ndarray, p: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """Pooled out-of-fold AUC of a logistic probe on elementwise bed*payload,
    GroupKFold by payload so a held-out payload is never seen in training."""
    feats = b * p                                   # (N, D) per-dim agreement
    n_groups = len(np.unique(groups))
    if n_groups < 3 or y.sum() < 3 or (len(y) - y.sum()) < 3:
        return float("nan")
    oof = np.zeros(len(y))
    gkf = GroupKFold(n_splits=min(5, n_groups))
    for tr, te in gkf.split(feats, y, groups):
        if y[tr].sum() == 0 or y[tr].sum() == len(tr):
            oof[te] = y[tr].mean()
            continue
        clf = LogisticRegression(max_iter=2000, C=0.05, class_weight="balanced")
        clf.fit(feats[tr], y[tr])
        oof[te] = clf.predict_proba(feats[te])[:, 1]
    return roc_auc_score(y, oof)


def main() -> int:
    if not EMB_PATH.is_file():
        print(f"no embeddings yet at {EMB_PATH}", file=sys.stderr)
        return 1
    cache = pickle.loads(EMB_PATH.read_bytes())
    gt = sys.argv[1] if len(sys.argv) > 1 else "labeling/fixtures/bb12_ground_truth.yaml"
    pairs = extract_pairs(gt)

    usable, beds, pays, y, groups = _assemble(cache, pairs)
    n_layers = beds.shape[1]
    print(f"embedded stems: {len(cache)} | usable pairs: {len(usable)} "
          f"(pos={int(y.sum())}, neg={int((1-y).sum())}) | payload groups: {len(np.unique(groups))} "
          f"| layers: {n_layers}\n")
    if y.sum() < 5:
        print("too few usable positive pairs yet — let the embed finish.", file=sys.stderr)
        return 1

    # per-layer L2-normalize every stem vector
    bn = np.stack([[_l2n(beds[i, L]) for L in range(n_layers)] for i in range(len(beds))])
    pn = np.stack([[_l2n(pays[i, L]) for L in range(n_layers)] for i in range(len(pays))])

    print(f"{'layer':>5} {'cos_AUC':>8} {'probe_AUC':>10}")
    rows = []
    for L in range(n_layers):
        cos = np.sum(bn[:, L] * pn[:, L], axis=1)
        auc_cos = roc_auc_score(y, cos)
        auc_probe = _grouped_probe_auc(bn[:, L], pn[:, L], y, groups)
        rows.append((L, auc_cos, auc_probe))
        print(f"{L:>5} {auc_cos:>8.3f} {auc_probe:>10.3f}")

    best_cos = max(rows, key=lambda r: r[1])
    best_probe = max((r for r in rows if not np.isnan(r[2])), key=lambda r: r[2], default=None)
    print(f"\nbest cosine : layer {best_cos[0]}  AUC {best_cos[1]:.3f}")
    if best_probe:
        print(f"best probe  : layer {best_probe[0]}  AUC {best_probe[2]:.3f}")
    print("\nverdict:", "SIGNAL — stem-MERT separates real mashups"
          if max(best_cos[1], (best_probe[2] if best_probe else 0)) >= 0.62
          else "weak/none at whole-song granularity — try section-level or learned compat head")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
