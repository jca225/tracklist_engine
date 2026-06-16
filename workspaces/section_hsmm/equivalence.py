"""Audio-equivalence classes over a reference track's own bars.

Core idea (the user's "two parts sound basically identical -> allow leeway"
principle, made computable): a DJ-set aligner cannot — and should not be asked
to — distinguish chorus-1 from chorus-2 when they are acoustically the same
bars. So we define, per reference track, an *equivalence relation* over its bar
positions from the track's MERT self-similarity, and score ref-offset
placement up to that relation.

All vectors are per-bar (measure-synced) MERT probe-layer embeddings, so a bar
index is naturally tempo-invariant: bar i of the mix lines up with bar i of the
reference regardless of absolute BPM.
"""
from __future__ import annotations

import numpy as np


def l2norm(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize (last axis)."""
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-8, None)


def window_cos(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-bar cosine between two equal-length normalized bar windows."""
    m = min(len(a), len(b))
    if m == 0:
        return 0.0
    return float((a[:m] * b[:m]).sum(axis=1).mean())


def matched_filter(mix_win: np.ndarray, ref_vecs: np.ndarray) -> tuple[int, float, np.ndarray]:
    """Slide a normalized mix bar-window over a normalized ref bar-track.

    Returns (best_start_bar, best_score, full_score_curve). The window is a
    *sequence* of bars, not a single pooled vector — that is what makes it
    localize where single-bar / pooled cosine (documented ~900 s off) does not.
    """
    M, R = mix_win.shape[0], ref_vecs.shape[0]
    if R < M or M == 0:
        return 0, -1.0, np.zeros(0, dtype=np.float32)
    scores = np.empty(R - M + 1, dtype=np.float32)
    for p in range(R - M + 1):
        scores[p] = float((mix_win * ref_vecs[p : p + M]).sum() / M)
    p = int(np.argmax(scores))
    return p, float(scores[p]), scores


def windows_equivalent(
    ref_vecs: np.ndarray, p_pred: int, p_gt: int, width: int, thresh: float
) -> tuple[bool, float]:
    """Are the predicted and GT ref windows the same audio (within thresh)?

    Compares ref[p_pred : p_pred+width] against ref[p_gt : p_gt+width] in the
    track's *own* embedding space. If they are mutually near-identical (a
    chorus matching a chorus), a different bar index is not a real error.
    """
    a = ref_vecs[p_pred : p_pred + width]
    b = ref_vecs[p_gt : p_gt + width]
    c = window_cos(a, b)
    return c >= thresh, c


def self_similarity_floor(ref_vecs: np.ndarray, width: int) -> float:
    """Median off-diagonal window-cosine for a track — the chance level that a
    given equivalence threshold must beat to be meaningful (a high floor means
    the track is internally repetitive; a low floor means windows are distinct).
    """
    R = ref_vecs.shape[0]
    n = R - width + 1
    if n < 2:
        return float("nan")
    step = max(1, n // 24)  # subsample to keep it cheap
    starts = list(range(0, n, step))
    cs = []
    for i in range(len(starts)):
        for j in range(i + 1, len(starts)):
            if abs(starts[i] - starts[j]) < width:
                continue  # skip overlapping windows (trivially similar)
            cs.append(window_cos(ref_vecs[starts[i] : starts[i] + width],
                                  ref_vecs[starts[j] : starts[j] + width]))
    return float(np.median(cs)) if cs else float("nan")
