"""Bar-level tokenization for mix MERT sequences."""
from __future__ import annotations

import numpy as np

from .mert_vectors import l2_normalize


def fit_vq_kmeans(
    vectors: np.ndarray,
    n_clusters: int,
    *,
    seed: int = 0,
    max_iter: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd k-means on L2-normalized rows → (centroids, labels)."""
    if vectors.shape[0] < n_clusters:
        raise ValueError(
            f"need at least {n_clusters} bars for {n_clusters} clusters, got {vectors.shape[0]}"
        )
    x = l2_normalize(vectors.astype(np.float32))
    rng = np.random.default_rng(seed)
    init_idx = rng.choice(x.shape[0], size=n_clusters, replace=False)
    centroids = x[init_idx].copy()

    labels = np.zeros(x.shape[0], dtype=np.int64)
    for _ in range(max_iter):
        sims = x @ centroids.T
        new_labels = np.argmax(sims, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in range(n_clusters):
            mask = labels == k
            if not np.any(mask):
                centroids[k] = x[rng.integers(0, x.shape[0])]
            else:
                centroids[k] = l2_normalize(np.mean(x[mask], axis=0, keepdims=True))[0]

    return centroids, labels
