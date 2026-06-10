"""Global monotonic placement decode over a set's spans (identity-first).

BB12 GT spans are strictly monotonic in tracklist order (0/146 backward
steps, median advance 16.5 s), and the per-slot anchor prior has a
heavy-tailed error (5/30 held-out GT starts fall outside the search band
entirely). So instead of placing each span independently inside a band
around its prior, score every span's assigned recording against the WHOLE
mix and pick the jointly-best non-decreasing sequence of starts.
"""
from __future__ import annotations

import numpy as np

NEG = -1e18


def window_mean_curve(per_measure_logits: np.ndarray, k: int) -> np.ndarray:
    """Mean of logits over a k-measure window at every start -> (T,).

    Starts whose window would run past the end are masked to NEG.
    """
    t = per_measure_logits.shape[0]
    k = max(1, min(k, t))
    csum = np.concatenate(([0.0], np.cumsum(per_measure_logits, dtype=np.float64)))
    curve = np.full(t, NEG, dtype=np.float64)
    n_valid = t - k + 1
    curve[:n_valid] = (csum[k:] - csum[:-k]) / k
    return curve


def window_mean_vectors(vectors: np.ndarray, k: int) -> np.ndarray:
    """Mean of (T, D) vectors over a k-row window at every start -> (T, D).

    Tail starts (window past the end) pool whatever remains; the span curve's
    NEG mask is what invalidates them downstream.
    """
    t = vectors.shape[0]
    k = max(1, min(k, t))
    csum = np.concatenate(
        [np.zeros((1, vectors.shape[1]), dtype=np.float64), np.cumsum(vectors, axis=0, dtype=np.float64)]
    )
    out = np.empty_like(vectors, dtype=np.float64)
    n_valid = t - k + 1
    out[:n_valid] = (csum[k:] - csum[:-k]) / k
    for j in range(n_valid, t):
        out[j] = (csum[t] - csum[j]) / (t - j)
    return out.astype(np.float32)


def monotonic_decode(curves: np.ndarray, *, min_step: int = 1) -> np.ndarray:
    """Jointly-best increasing start indices for N spans.

    curves: (N, T) window-start scores (NEG = invalid start).
    Returns (N,) start indices with starts[i] >= starts[i-1] + min_step.
    min_step > 0 stops weak-curve spans from piling onto their neighbour's
    start (GT advance is never 0 — BB12 median is ~16.5 s).
    DP: best[i, w] = curves[i, w] + max_{w' <= w - min_step} best[i-1, w'] —
    the prefix max makes each span O(T), the whole decode O(N*T).
    """
    n, t = curves.shape
    ptr = np.zeros((n, t), dtype=np.int32)
    best = curves[0].astype(np.float64)

    for i in range(1, n):
        prefix = np.maximum.accumulate(best)
        # running argmax of `best` (first index achieving the prefix max)
        arg = np.zeros(t, dtype=np.int32)
        cur = 0
        for j in range(1, t):
            if best[j] > best[cur]:
                cur = j
            arg[j] = cur
        shifted_max = np.full(t, NEG, dtype=np.float64)
        shifted_arg = np.zeros(t, dtype=np.int32)
        if min_step < t:
            shifted_max[min_step:] = prefix[: t - min_step] if min_step else prefix
            shifted_arg[min_step:] = arg[: t - min_step] if min_step else arg
        ptr[i] = shifted_arg
        best = curves[i] + shifted_max

    starts = np.zeros(n, dtype=np.int32)
    starts[-1] = int(np.argmax(best))
    for i in range(n - 1, 0, -1):
        starts[i - 1] = ptr[i][starts[i]]
    return starts
