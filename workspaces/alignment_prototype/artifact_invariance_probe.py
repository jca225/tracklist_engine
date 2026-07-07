"""Artifact-invariance probe: do separation artifacts break stem matching?

Question (go/no-go for a contrastive artifact adapter): given a Roformer-
SEPARATED stem and the TRUE stem (official acappella/instrumental) of the
same recording, do our existing match features (HuBERT-L9 for vocals,
chroma for instrumentals) already retrieve the right song at the right
position across the artifact gap — or is the separator noise the wall?

Data: the 2026-07 Discord canonicalization pairs (true library stem +
separated stem of the same recording's reference download), fetched to
<pairs_dir>/{true,sep}/<key>.<ext>; keys pair by basename stem.

Protocol: 4 s query windows (hop 2 s) from each SEPARATED stem; database =
4 s windows (hop 1 s) of ALL true stems in the same class. Nearest-neighbor
by cosine of the mean-pooled window feature. A hit needs the right pair AND
the right position (+-2 s after the pair's global offset, itself estimated
by matched-filter). song-acc counts right-pair-any-position (identity
survives, position lost).

Verdict guide: retrieval@1 >= ~0.9 -> artifacts are NOT the wall, skip the
adapter; <= ~0.7 -> train the contrastive head (synthetic mix->separate
pairs as training data, these real pairs as validation).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.alignment_prototype.continuity_refine import (  # noqa: E402
    _scores_at_stretch,
)
from workspaces.alignment_prototype.path_decode import FPS, _ensure_feat  # noqa: E402

WIN_S = 4.0
Q_HOP_S = 2.0
DB_HOP_S = 1.0
TOL_S = 2.0


def _feat(path: Path, feature: str) -> np.ndarray:
    return np.load(_ensure_feat(str(path), str(path), feature, 9))


def _pool_windows(f: np.ndarray, win: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    """(D, T) -> (N, D) L2-normed mean-pooled windows + start frames."""
    starts = np.arange(0, max(1, f.shape[1] - win + 1), hop)
    out = np.stack([f[:, s : s + win].mean(axis=1) for s in starts])
    n = np.linalg.norm(out, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return out / n, starts


def pair_offset(sep_f: np.ndarray, true_f: np.ndarray) -> tuple[float, float]:
    """Global offset true_t - sep_t (s) via matched-filter of a mid window."""
    win = int(round(12 * FPS))
    mid = max(0, (sep_f.shape[1] - win) // 2)
    w = np.ascontiguousarray(sep_f[:, mid : mid + win])
    curve = _scores_at_stretch(w, np.ascontiguousarray(true_f), 1.0)
    if curve.size == 0:
        return 0.0, -1.0
    best = int(curve.argmax())
    return (best - mid) / FPS, float(curve.max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs_dir", type=Path)
    args = ap.parse_args()

    true_files = {
        p.stem: p for p in (args.pairs_dir / "true").iterdir() if p.stat().st_size
    }
    sep_files = {
        p.stem: p for p in (args.pairs_dir / "sep").iterdir() if p.stat().st_size
    }
    keys = sorted(true_files.keys() & sep_files.keys())
    by_class: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        by_class["acappella" if k.startswith("acap") else "instrumental"].append(k)

    win = int(round(WIN_S * FPS))
    for cls, ks in sorted(by_class.items()):
        feature = "hubert" if cls == "acappella" else "chroma"
        print(f"\n== {cls} ({feature}), {len(ks)} pairs ==")
        feats_true = {k: _feat(true_files[k], feature) for k in ks}
        feats_sep = {k: _feat(sep_files[k], feature) for k in ks}
        offsets = {}
        for k in ks:
            off, score = pair_offset(feats_sep[k], feats_true[k])
            offsets[k] = off
            print(f"  {k}: offset {off:+.2f}s (mf peak {score:.3f})")

        db_vecs, db_meta = [], []
        for k in ks:
            v, starts = _pool_windows(feats_true[k], win, int(round(DB_HOP_S * FPS)))
            db_vecs.append(v)
            db_meta += [(k, s / FPS) for s in starts]
        db = np.concatenate(db_vecs)

        hits = pos_hits = total = 0
        for k in ks:
            q, starts = _pool_windows(feats_sep[k], win, int(round(Q_HOP_S * FPS)))
            simm = q @ db.T
            for qi, s in enumerate(starts):
                bk, bt = db_meta[int(simm[qi].argmax())]
                total += 1
                if bk == k:
                    hits += 1
                    if abs(bt - (s / FPS + offsets[k])) <= TOL_S:
                        pos_hits += 1
        print(
            f"  RETRIEVAL@1: song {hits / total:.3f} | song+position "
            f"{pos_hits / total:.3f}  (n={total} query windows)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
