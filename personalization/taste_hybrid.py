"""Hybrid taste model: a small net that BLENDS co-occurrence (structural) + MERT (sound).

The pieces alone: ID-CF wins warm (recall@20 0.149) but is blind cold (0.000); MERT is
weak warm (0.068) but the only cold signal (0.034). This learns to USE EACH WHERE IT'S
STRONG. Per (user, candidate) it computes 4 scalar scores — CF co-occurrence, MERT
nearest-neighbor, MERT centroid, popularity — and an MLP maps them to P(engage).

Key trick: during training we randomly ZERO the CF feature (cold-dropout), so the net
learns to fall back on the MERT scores when co-occurrence is absent (the cold-start /
Spotify-library case). Frozen embeddings, CPU, no Vast.

  venvs/audio/bin/python -m personalization.taste_hybrid
"""
from __future__ import annotations

import pickle
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

DB = Path("data/taste/taste_warehouse.db")
EMB = Path("data/taste/tail_track_embeds.pkl")
K = 20
MIN_HIST = 8
LAYER = 6            # best taste layer from taste_mert
NEG_PER_POS = 5
CF_DROPOUT = 0.35    # fraction of training rows with CF feature zeroed (teach cold fallback)


def _l2(M):
    n = np.linalg.norm(M, axis=-1, keepdims=True)
    return M / np.where(n > 0, n, 1)


def main() -> int:
    emb = pickle.loads(EMB.read_bytes())
    ids = sorted(emb); idx = {t: i for i, t in enumerate(ids)}
    V = _l2(np.stack([emb[t][LAYER].astype(np.float32) for t in ids]))   # (N, dim)
    N = len(ids)

    conn = sqlite3.connect(DB)
    rows = conn.execute("SELECT user_id, track_id FROM sc_likes ORDER BY user_id, liked_at, rowid").fetchall()
    tl = defaultdict(list)
    for u, t in rows:
        if t in idx:
            tl[u].append(idx[t])
    users = sorted(u for u in tl if len(tl[u]) >= MIN_HIST)
    test = set(users[::5]); train = [u for u in users if u not in test]

    # structural co-occurrence from train users
    ri, ci = [], []
    for i, u in enumerate(train):
        for it in set(tl[u]):
            ri.append(i); ci.append(it)
    M = csr_matrix((np.ones(len(ri)), (ri, ci)), shape=(len(train), N))
    cooc = (M.T @ M).toarray().astype(np.float32); np.fill_diagonal(cooc, 0)
    pop = np.asarray(M.sum(0)).ravel() + 1.0
    Scf = cooc / np.sqrt(np.outer(pop, pop))
    logpop = np.log(pop)
    rng = np.random.default_rng(0)

    def feats(prefix):
        cf = Scf[prefix].sum(0)                       # (N,) co-occurrence with prefix
        knn = (V @ V[prefix].T).max(1)                # (N,) sound: nearest liked track
        cen = V @ V[prefix].mean(0)                   # (N,) sound: centroid
        return cf, knn, cen

    # build training rows
    X, y = [], []
    for u in train:
        seq = tl[u]; sp = int(len(seq) * 0.8)
        prefix = list(dict.fromkeys(seq[:sp])); held = list(set(seq[sp:]) - set(seq[:sp]))
        if len(prefix) < 3 or not held:
            continue
        cf, knn, cen = feats(prefix)
        negs = rng.choice([j for j in range(N) if j not in set(prefix) and j not in set(held)],
                          size=min(NEG_PER_POS * len(held), N - len(prefix) - len(held)), replace=False)
        for c in held:
            X.append([cf[c], knn[c], cen[c], logpop[c]]); y.append(1)
        for c in negs:
            X.append([cf[c], knn[c], cen[c], logpop[c]]); y.append(0)
    X = np.array(X); y = np.array(y)
    # cold-dropout: zero CF on a fraction of rows so the net learns to use MERT alone
    drop = rng.random(len(X)) < CF_DROPOUT
    X[drop, 0] = 0.0
    print(f"train rows: {len(y)} (pos {int(y.sum())}) | test users: {len(test)} | cold-dropout {CF_DROPOUT}")

    scaler = StandardScaler().fit(X)
    clf = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=400, random_state=0).fit(scaler.transform(X), y)

    cold_mask = (np.arange(N) % 4 == 0)               # 25% "unseen" for cold-start eval
    rec_h, rec_cf, rec_m = [], [], []
    cold_h, cold_cf, cold_m = 0.0, 0.0, 0.0; nc = 0
    for u in test:
        seq = tl[u]; sp = int(len(seq) * 0.8)
        prefix = list(dict.fromkeys(seq[:sp])); held = set(seq[sp:]) - set(seq[:sp])
        if len(prefix) < 3 or not held:
            continue
        cf, knn, cen = feats(prefix)
        F = np.stack([cf, knn, cen, logpop], 1)
        ph = clf.predict_proba(scaler.transform(F))[:, 1]
        for s in (ph, cf, knn):
            s[prefix] = -1e9
        d = min(K, len(held))
        rec_h.append(len(set(np.argsort(-ph)[:K]) & held) / d)
        rec_cf.append(len(set(np.argsort(-cf)[:K]) & held) / d)
        rec_m.append(len(set(np.argsort(-knn)[:K]) & held) / d)
        # cold-start: hide 25% from CF (cf=0), can the hybrid still get them via MERT?
        held_cold = {h for h in held if cold_mask[h]}
        if held_cold:
            cf_c = cf.copy(); cf_c[cold_mask] = 0.0
            Fc = np.stack([cf_c, knn, cen, logpop], 1)
            phc = clf.predict_proba(scaler.transform(Fc))[:, 1]; phc[prefix] = -1e9
            cfc = cf_c.copy(); cfc[prefix] = -1e9
            dc = min(K, len(held_cold))
            cold_h += len(set(np.argsort(-phc)[:K]) & held_cold) / dc
            cold_cf += len(set(np.argsort(-cfc)[:K]) & held_cold) / dc
            cold_m += len(set(np.argsort(-knn)[:K]) & held_cold) / dc
            nc += 1

    print(f"\nWARM recall@{K}:  hybrid {np.mean(rec_h):.3f}  |  ID-CF {np.mean(rec_cf):.3f}  |  MERT {np.mean(rec_m):.3f}")
    print(f"COLD recall@{K}:  hybrid {cold_h/nc:.3f}  |  ID-CF {cold_cf/nc:.3f}  |  MERT {cold_m/nc:.3f}   ({nc} users)")
    print("\ntargets to beat: warm 0.149 (ID-CF), cold 0.034 (MERT)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
