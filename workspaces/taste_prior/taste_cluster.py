"""Cluster listeners by liked-track + playlist overlap (sparse binary vectors)."""
from __future__ import annotations

import json
import logging
import sqlite3
from collections import Counter
from datetime import datetime, timezone

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans

from workspaces.taste_prior.persistence import clean_user_ids, replace_taste_clusters

logger = logging.getLogger(__name__)

ALGORITHM = "mbk_track_v1"


def _user_track_sets(conn: sqlite3.Connection, user_ids: list[str]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {uid: set() for uid in user_ids}
    if not user_ids:
        return out
    placeholders = ",".join("?" * len(user_ids))
    for row in conn.execute(
        f"SELECT user_id, track_id FROM sc_likes WHERE user_id IN ({placeholders})",
        user_ids,
    ):
        out[str(row["user_id"])].add(int(row["track_id"]))
    for row in conn.execute(
        f"SELECT user_id, track_ids_json FROM sc_playlists WHERE user_id IN ({placeholders})",
        user_ids,
    ):
        try:
            ids = json.loads(row["track_ids_json"])
        except json.JSONDecodeError:
            continue
        out[str(row["user_id"])].update(int(t) for t in ids if t is not None)
    return out


def cluster_mix(
    conn: sqlite3.Connection,
    mix_id: str,
    *,
    exclude_bots: bool = True,
    min_tracks: int = 15,
    max_users: int = 5000,
    vocab_size: int = 3000,
    n_clusters: int = 12,
    random_state: int = 42,
) -> dict[str, object]:
    """Cluster clean cohort users; returns summary stats."""
    user_ids = clean_user_ids(conn, mix_id, exclude_bots=exclude_bots)
    track_sets = _user_track_sets(conn, user_ids)
    eligible = [uid for uid, tracks in track_sets.items() if len(tracks) >= min_tracks]
    if len(eligible) > max_users:
        # stable subsample — sort by user_id hash
        eligible = sorted(eligible)[:max_users]

    if len(eligible) < n_clusters:
        return {"error": "too_few_users", "eligible": len(eligible), "n_clusters": n_clusters}

    freq: Counter[int] = Counter()
    for uid in eligible:
        freq.update(track_sets[uid])
    vocab = [tid for tid, _ in freq.most_common(vocab_size)]
    tid_to_col = {tid: i for i, tid in enumerate(vocab)}

    rows: list[int] = []
    cols: list[int] = []
    for r, uid in enumerate(eligible):
        for tid in track_sets[uid]:
            c = tid_to_col.get(tid)
            if c is not None:
                rows.append(r)
                cols.append(c)
    data = np.ones(len(rows), dtype=np.float32)
    matrix = csr_matrix((data, (rows, cols)), shape=(len(eligible), len(vocab)))

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=min(2048, len(eligible)),
        n_init=3,
    )
    labels = kmeans.fit_predict(matrix)

    assignments = tuple((eligible[i], int(labels[i])) for i in range(len(eligible)))
    replace_taste_clusters(conn, mix_id, ALGORITHM, assignments)

    sizes = Counter(int(x) for x in labels)
    return {
        "mix_id": mix_id,
        "algorithm": ALGORITHM,
        "users_clustered": len(eligible),
        "exclude_bots": exclude_bots,
        "min_tracks": min_tracks,
        "vocab_size": len(vocab),
        "n_clusters": n_clusters,
        "cluster_sizes": dict(sorted(sizes.items())),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
