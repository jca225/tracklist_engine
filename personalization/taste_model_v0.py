"""v0 taste model: does a user's taste predict their FUTURE likes? (ID-based CF baseline)

The cheap baseline the costly MERT version must beat. Pure collaborative filtering —
tracks are opaque IDs, signal is co-occurrence. Per user, split the like timeline by
time: prefix = earlier likes, held-out = later likes. Item-item co-occurrence learned
from TRAIN users; for held-out users, score candidates from their prefix and measure
recall@K of their actual future likes vs a popularity baseline.

If taste-CF >> popularity, taste predicts engagement (the signal audio lacked) and the
pivot is validated. ID-CF CANNOT generalize to unseen tracks (cold-start) — that gap
is exactly what MERT embeddings would fill.

  venvs/audio/bin/python -m personalization.taste_model_v0
"""
from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

DB = Path("data/taste/taste_warehouse.db")
TOPK = 5000          # track vocabulary (top by like-count)
MIN_HIST = 10        # users need enough timeline to split
K = 20               # recall@K


def main() -> int:
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "SELECT user_id, track_id FROM sc_likes ORDER BY user_id, liked_at, rowid").fetchall()

    vocab = {tid: i for i, (tid, _) in enumerate(Counter(t for _, t in rows).most_common(TOPK))}
    tl: dict[str, list[int]] = defaultdict(list)
    for u, t in rows:
        if t in vocab:
            tl[u].append(vocab[t])                       # time-ordered item indices
    users = sorted(u for u in tl if len(tl[u]) >= MIN_HIST)
    test = set(users[::5])                                # 20% held-out users
    train = [u for u in users if u not in test]
    print(f"users: {len(users)} (train {len(train)}, test {len(test)}) | vocab {TOPK}")

    # item-item co-occurrence from train users' full timelines
    ri, ci = [], []
    for i, u in enumerate(train):
        for it in set(tl[u]):
            ri.append(i); ci.append(it)
    M = csr_matrix((np.ones(len(ri)), (ri, ci)), shape=(len(train), TOPK))
    cooc = (M.T @ M).toarray().astype(np.float32)
    np.fill_diagonal(cooc, 0)
    pop = np.asarray(M.sum(0)).ravel() + 1.0
    S = cooc / np.sqrt(np.outer(pop, pop))               # cosine-normalized co-occurrence
    pop_rank = np.argsort(-pop)

    HEAD = 500                                            # top-500 = "popular"; tail = personalization zone
    head_set = set(pop_rank[:HEAD].tolist())
    rec_cf, rec_pop, tail_cf, tail_pop = [], [], [], []
    for u in test:
        seq = tl[u]
        split = int(len(seq) * 0.8)
        prefix, held = set(seq[:split]), set(seq[split:]) - set(seq[:split])
        if len(prefix) < 3 or not held:
            continue
        score = S[list(prefix)].sum(axis=0)
        score[list(prefix)] = -1e9
        cf_top = set(np.argsort(-score)[:K])
        pop_top = set([j for j in pop_rank if j not in prefix][:K])
        denom = min(K, len(held))
        rec_cf.append(len(cf_top & held) / denom)
        rec_pop.append(len(pop_top & held) / denom)
        # TAIL: held-out items outside the popular head; candidates also exclude the head
        held_t = held - head_set
        if held_t:
            score_t = score.copy(); score_t[list(head_set)] = -1e9
            cf_t = set(np.argsort(-score_t)[:K])
            pop_t = set([j for j in pop_rank if j not in prefix and j not in head_set][:K])
            dt = min(K, len(held_t))
            tail_cf.append(len(cf_t & held_t) / dt)
            tail_pop.append(len(pop_t & held_t) / dt)

    print(f"\nrecall@{K} on held-out future likes ({len(rec_cf)} test users):")
    print(f"  ALL    taste-CF {np.mean(rec_cf):.3f}  popularity {np.mean(rec_pop):.3f}  "
          f"lift {np.mean(rec_cf)/max(np.mean(rec_pop),1e-9):.2f}x")
    print(f"  TAIL   taste-CF {np.mean(tail_cf):.3f}  popularity {np.mean(tail_pop):.3f}  "
          f"lift {np.mean(tail_cf)/max(np.mean(tail_pop),1e-9):.2f}x   (personalization zone)")
    print("\nverdict:", "TASTE PREDICTS in the tail — pivot alive; MERT next for cold-start"
          if np.mean(tail_cf) >= 1.5 * np.mean(tail_pop)
          else "weak even in the tail — the BB cohort is too taste-homogeneous; reconsider the lens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
