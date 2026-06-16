#!/usr/bin/env python3
"""Fibers — a track's self-repeat equivalence classes, computed a priori.

Think of playback as a map ref-time -> audio-content. A repeated chorus means
many ref positions share one content value; the *fiber* over that content is the
set of repeat instances. Every "repeat error" the placement decoder makes (e.g.
picking ref 147 s when GT is 48 s, cosine 0.99) is it landing on the wrong point
of the RIGHT fiber. Knowing the fibers lets us (1) score within-fiber picks as
correct, (2) collapse repeats so they don't compete as separate hypotheses, and
(3) disambiguate the instance by context (monotonic progression) or abstain.

Computing fibers robustly is a music-structure-analysis problem, and the naive
methods FAIL: thresholded union-find and k-NN-recurrence + connected-components
both transitively chain a repetitive track into one blob (100% of the track =
one fiber). The fix is a fixed-K spectral PARTITION (McFee-Ellis Laplacian
segmentation): a partition can't blob. Robustness to "sung differently" comes
from the feature — HuBERT phonetic frames match on the *words*, which repeat even
when melody/delivery/key don't; separation noise is rejected by path-enhancement
(a fiber needs a sustained diagonal, not frame-coincidence).

So: feed HuBERT frames for vocal stems (chroma is fine for harmonic beds but
over-similar on EDM). `compute_fibers` returns a per-(downsampled)-frame label;
`same_fiber` answers the question the decoder/scorer actually needs.
"""

from __future__ import annotations

import warnings

import numpy as np


def compute_fibers(
    feat: np.ndarray,
    fps: float,
    *,
    k: int = 6,
    ds_hz: float = 8.0,
    min_section_s: float = 4.0,
) -> tuple[np.ndarray, float]:
    """(labels, label_hz). feat=(D, T) per-column features at `fps`.

    Spectral-cluster the recurrence+sequence affinity into k section labels.
    Same label at two times => same fiber (mutually self-similar section).
    Downsamples to ~ds_hz first (structure is coarse; keeps the eigendecomp
    cheap). k is clamped so short tracks don't get over-segmented."""
    import librosa
    from sklearn.cluster import KMeans

    step = max(1, int(round(fps / ds_hz)))
    g = feat[:, ::step].astype(np.float32)
    g_hz = fps / step
    g = g / (np.linalg.norm(g, axis=0, keepdims=True) + 1e-9)
    n = g.shape[1]
    k = max(2, min(k, n // max(1, int(min_section_s * g_hz))))
    if n < 4 or k < 2:
        return np.zeros(n, dtype=int), g_hz

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = librosa.segment.recurrence_matrix(
            g, mode="affinity", sym=True, width=max(3, int(3 * g_hz))
        )
        rec = librosa.segment.path_enhance(rec, n=max(1, int(2 * g_hz)))
    # add a sequence term so sections stay temporally contiguous (McFee-Ellis)
    seq = np.zeros((n, n), dtype=np.float32)
    i = np.arange(n - 1)
    seq[i, i + 1] = seq[i + 1, i] = 1.0
    affinity = rec + seq * (rec.mean() + 1e-9)
    deg = affinity.sum(1) + 1e-9
    lap = np.eye(n) - affinity / deg[:, None]  # random-walk Laplacian
    w, v = np.linalg.eig(lap)
    order = np.argsort(w.real)
    x = v.real[:, order[:k]]
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    labels = KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(x)
    return labels.astype(int), g_hz


def fiber_at(labels: np.ndarray, label_hz: float, sec: float) -> int:
    if labels.size == 0:
        return -1
    b = int(round(sec * label_hz))
    return int(labels[min(max(b, 0), labels.size - 1)])


def same_fiber(labels: np.ndarray, label_hz: float, a_s: float, b_s: float) -> bool:
    """Do two ref times fall in the same self-repeat class?"""
    return fiber_at(labels, label_hz, a_s) == fiber_at(labels, label_hz, b_s)


def fiber_intervals(
    labels: np.ndarray, label_hz: float, min_len_s: float = 4.0
) -> list[tuple[float, float, int]]:
    """Contiguous (start_s, end_s, label) runs >= min_len_s — for inspection."""
    out: list[tuple[float, float, int]] = []
    if labels.size == 0:
        return out
    s = 0
    for i in range(1, labels.size + 1):
        if i == labels.size or labels[i] != labels[i - 1]:
            if (i - s) / label_hz >= min_len_s:
                out.append((s / label_hz, i / label_hz, int(labels[s])))
            s = i
    return out
