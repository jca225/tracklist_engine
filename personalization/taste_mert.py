"""MERT-embedding taste model: does sound-proximity beat ID-CF in the tail + cold-start?

Taste = the centroid of a user's liked-track MERT embeddings (recency is implicit in
the prefix). Recommend candidates by cosine to that centroid. Compared head-to-head
with the v0 ID co-occurrence model and popularity, on the EMBEDDED-track universe,
per MERT layer (which layer carries taste?).

The decisive test is COLD-START: tracks held out of the co-occurrence graph entirely.
ID-CF scores them ~0 (never seen co-liked); MERT scores them by sound. If MERT recalls
cold tracks that ID-CF can't, that's the win the embeddings are for.

  venvs/audio/bin/python -m personalization.taste_mert
"""
from __future__ import annotations

import pickle
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

DB = Path("data/taste/taste_warehouse.db")
EMB = Path("data/taste/tail_track_embeds.pkl")
K = 20
MIN_HIST = 8


def _l2(M):
    n = np.linalg.norm(M, axis=-1, keepdims=True)
    return M / np.where(n > 0, n, 1)


def main() -> int:
    if not EMB.is_file():
        print("no embeddings yet — run personalization.embed_tail first"); return 1
    emb = pickle.loads(EMB.read_bytes())
    ids = sorted(emb)                                       # embedded sc_track_ids = the universe
    idx = {tid: i for i, tid in enumerate(ids)}
    layers = next(iter(emb.values())).shape[0]
    cube = np.stack([emb[t].astype(np.float32) for t in ids])   # (N, n_layers, dim)
    print(f"embedded universe: {len(ids)} tracks | layers: {layers}")

    conn = sqlite3.connect(DB)
    rows = conn.execute("SELECT user_id, track_id FROM sc_likes ORDER BY user_id, liked_at, rowid").fetchall()
    tl = defaultdict(list)
    for u, t in rows:
        if t in idx:
            tl[u].append(idx[t])
    users = sorted(u for u in tl if len(tl[u]) >= MIN_HIST)
    test = set(users[::5]); train = [u for u in users if u not in test]
    print(f"users w/ >= {MIN_HIST} embedded likes: {len(users)} (train {len(train)}, test {len(test)})")
    if len(test) < 20:
        print("too few — embed more tail tracks before evaluating."); return 1

    # ID-CF co-occurrence on the embedded universe (apples-to-apples baseline)
    ri, ci = [], []
    for i, u in enumerate(train):
        for it in set(tl[u]):
            ri.append(i); ci.append(it)
    M = csr_matrix((np.ones(len(ri)), (ri, ci)), shape=(len(train), len(ids)))
    cooc = (M.T @ M).toarray().astype(np.float32); np.fill_diagonal(cooc, 0)
    pop = np.asarray(M.sum(0)).ravel() + 1.0
    Scf = cooc / np.sqrt(np.outer(pop, pop))
    pop_rank = np.argsort(-pop)
    warm = pop > 2                                          # tracks ID-CF has seen co-liked enough

    print(f"\n{'layer':>5} {'cMERT':>6} {'knnMERT':>7} {'ID-CF':>6} {'pop':>6}   (recall@%d)" % K)
    best = (0, 0.0)
    for L in range(layers):
        V = _l2(cube[:, L, :])                             # (N, dim) normalized at layer L
        rc_cen, rc_knn, rc, rp = [], [], [], []
        for u in test:
            seq = tl[u]; sp = int(len(seq) * 0.8)
            prefix, held = list(dict.fromkeys(seq[:sp])), set(seq[sp:]) - set(seq[:sp])
            if len(prefix) < 3 or not held:
                continue
            centroid = V[prefix].mean(0)
            sm_c = V @ centroid; sm_c[prefix] = -1e9
            sm_k = (V @ V[prefix].T).max(1); sm_k[prefix] = -1e9    # nearest-neighbor to any liked
            sc = Scf[prefix].sum(0); sc[prefix] = -1e9
            d = min(K, len(held))
            rc_cen.append(len(set(np.argsort(-sm_c)[:K]) & held) / d)
            rc_knn.append(len(set(np.argsort(-sm_k)[:K]) & held) / d)
            rc.append(len(set(np.argsort(-sc)[:K]) & held) / d)
            rp.append(len(set([j for j in pop_rank if j not in prefix][:K]) & held) / d)
        ak = float(np.mean(rc_knn))
        print(f"{L:>5} {np.mean(rc_cen):>6.3f} {ak:>7.3f} {np.mean(rc):>6.3f} {np.mean(rp):>6.3f}")
        if ak > best[1]:
            best = (L, ak)
    print(f"\nbest taste layer: {best[0]} (knn-MERT recall@{K}={best[1]:.3f})")

    # COLD-START SIMULATION: hide 25% of tracks from the co-occurrence graph (treat as
    # unseen — like a Spotify-library track no SC user co-liked). ID-CF scores them 0;
    # MERT scores them by sound. This is MERT's actual justification.
    cold_mask = (np.arange(len(ids)) % 4 == 0)
    Scf_cold = Scf.copy(); Scf_cold[:, cold_mask] = 0.0
    V = _l2(cube[:, best[0], :])
    cm = cc = 0.0; n = 0
    for u in test:
        seq = tl[u]; sp = int(len(seq) * 0.8)
        prefix = list(dict.fromkeys(seq[:sp])); held = set(seq[sp:]) - set(seq[:sp])
        held_cold = {h for h in held if cold_mask[h]}
        if len(prefix) < 3 or not held_cold:
            continue
        sm = (V @ V[prefix].T).max(1); sm[prefix] = -1e9        # MERT knn
        sc = Scf_cold[prefix].sum(0); sc[prefix] = -1e9          # ID-CF, cold tracks blind
        d = min(K, len(held_cold))
        cm += len(set(np.argsort(-sm)[:K]) & held_cold) / d
        cc += len(set(np.argsort(-sc)[:K]) & held_cold) / d
        n += 1
    if n:
        print(f"\nCOLD-START recall@{K} on unseen tracks ({n} users, 25% of catalog hidden from CF):")
        print(f"  MERT {cm/n:.3f}  vs  ID-CF {cc/n:.3f}")
        print("  -> MERT's niche: tracks with NO co-occurrence (your Spotify library)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
