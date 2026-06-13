"""The taste recommender: route ID-CF (warm) + learned MERT net (cold) behind one call.

Findings (see project_hrm_text_taste_pretrain / taste_mert / taste_hybrid):
  - WARM (track has co-occurrence data): ID-CF wins, recall@20 0.149.
  - COLD (unseen track, e.g. a Spotify-library song): ID-CF=0; a small learned net
    over MERT(+pop) recovers 0.046.
Blending into one model dilutes warm, so we ROUTE per candidate and merge on a common
P(engage) scale (CF score calibrated to a probability; the net already outputs one).

  rec = TasteRecommender().fit(conn)
  rec.recommend(prefix_track_ids=[...], candidates=[...], k=20)   # -> [(track_id, p, 'warm'|'cold')]

  venvs/audio/bin/python -m personalization.taste_recommender    # fit + held-out eval + demo
"""
from __future__ import annotations

import pickle
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

DB = Path("data/taste/taste_warehouse.db")
EMB = Path("data/taste/tail_track_embeds.pkl")
MODEL = Path("data/taste/taste_recommender.pkl")
LAYER = 6
WARM_THRESH = 2.0          # candidate is "warm" if train co-like count > this
NEG_PER_POS = 5
CF_DROPOUT = 0.35


def _l2(M):
    n = np.linalg.norm(M, axis=-1, keepdims=True)
    return M / np.where(n > 0, n, 1)


class TasteRecommender:
    def __init__(self, layer: int = LAYER, warm_thresh: float = WARM_THRESH):
        self.layer = layer; self.warm_thresh = warm_thresh

    # ---- features ---------------------------------------------------------
    def _feats(self, prefix: list[int]):
        cf = self.Scf[prefix].sum(0)
        knn = (self.V @ self.V[prefix].T).max(1)
        cen = self.V @ self.V[prefix].mean(0)
        return cf, knn, cen

    # ---- fit --------------------------------------------------------------
    def fit(self, conn: sqlite3.Connection, train_users: list[str] | None = None):
        emb = pickle.loads(EMB.read_bytes())
        self.ids = sorted(emb); self.idx = {t: i for i, t in enumerate(self.ids)}
        self.V = _l2(np.stack([emb[t][self.layer].astype(np.float32) for t in self.ids]))
        N = len(self.ids)

        rows = conn.execute("SELECT user_id, track_id FROM sc_likes ORDER BY user_id, liked_at, rowid").fetchall()
        tl = defaultdict(list)
        for u, t in rows:
            if t in self.idx:
                tl[u].append(self.idx[t])
        users = train_users if train_users is not None else sorted(u for u in tl if len(tl[u]) >= 8)
        self._tl = tl

        ri, ci = [], []
        for i, u in enumerate(users):
            for it in set(tl[u]):
                ri.append(i); ci.append(it)
        M = csr_matrix((np.ones(len(ri)), (ri, ci)), shape=(len(users), N))
        cooc = (M.T @ M).toarray().astype(np.float32); np.fill_diagonal(cooc, 0)
        self.pop = np.asarray(M.sum(0)).ravel() + 1.0
        self.Scf = cooc / np.sqrt(np.outer(self.pop, self.pop))
        self.logpop = np.log(self.pop)

        rng = np.random.default_rng(0)
        X, y, cf_cal = [], [], []
        for u in users:
            seq = tl[u]; sp = int(len(seq) * 0.8)
            prefix = list(dict.fromkeys(seq[:sp])); held = list(set(seq[sp:]) - set(seq[:sp]))
            if len(prefix) < 3 or not held:
                continue
            cf, knn, cen = self._feats(prefix)
            negs = rng.choice([j for j in range(N) if j not in set(prefix) and j not in set(held)],
                              size=min(NEG_PER_POS * len(held), N - len(prefix) - len(held)), replace=False)
            for c in held:
                X.append([cf[c], knn[c], cen[c], self.logpop[c]]); y.append(1); cf_cal.append((cf[c], 1))
            for c in negs:
                X.append([cf[c], knn[c], cen[c], self.logpop[c]]); y.append(0); cf_cal.append((cf[c], 0))
        X = np.array(X); y = np.array(y)
        drop = rng.random(len(X)) < CF_DROPOUT
        X[drop, 0] = 0.0
        self.scaler = StandardScaler().fit(X)
        self.net = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=400, random_state=0).fit(
            self.scaler.transform(X), y)
        # CF-score -> probability calibrator (monotonic; preserves warm ranking)
        cc = np.array(cf_cal)
        self.cf_cal = LogisticRegression().fit(cc[:, :1], cc[:, 1].astype(int))
        return self

    # ---- inference --------------------------------------------------------
    def recommend(self, prefix_track_ids: list[int], candidates: list[int] | None = None, k: int = 20):
        prefix = [self.idx[t] for t in prefix_track_ids if t in self.idx]
        cand = [self.idx[t] for t in candidates if t in self.idx] if candidates else list(range(len(self.ids)))
        cand = [c for c in cand if c not in set(prefix)]
        cf, knn, cen = self._feats(prefix)
        out = []
        for c in cand:
            warm = self.pop[c] > self.warm_thresh and cf[c] > 0
            if warm:
                p = float(self.cf_cal.predict_proba([[cf[c]]])[0, 1]); path = "warm"
            else:
                feat = self.scaler.transform([[0.0, knn[c], cen[c], self.logpop[c]]])
                p = float(self.net.predict_proba(feat)[0, 1]); path = "cold"
            out.append((self.ids[c], p, path))
        out.sort(key=lambda r: -r[1])
        return out[:k]

    def save(self, path: Path = MODEL):
        path.write_bytes(pickle.dumps(self))
    @staticmethod
    def load(path: Path = MODEL) -> "TasteRecommender":
        return pickle.loads(path.read_bytes())


def _eval(rec: TasteRecommender, test_users: list[str], K: int = 20):
    cold_mask = (np.arange(len(rec.ids)) % 4 == 0)
    warm_r, cold_r = [], []
    for u in test_users:
        seq = rec._tl[u]; sp = int(len(seq) * 0.8)
        prefix_ix = list(dict.fromkeys(seq[:sp])); held = set(seq[sp:]) - set(seq[:sp])
        if len(prefix_ix) < 3 or not held:
            continue
        prefix = [rec.ids[i] for i in prefix_ix]
        recd = {rec.idx[t] for t, _, _ in rec.recommend(prefix, k=K)}
        warm_r.append(len(recd & held) / min(K, len(held)))
        held_cold = {h for h in held if cold_mask[h]}
        if held_cold:                                     # simulate unseen: only cold candidates
            cold_cands = [rec.ids[i] for i in range(len(rec.ids)) if cold_mask[i]]
            # force-cold by temporarily blinding CF on cold candidates
            saved = rec.pop[cold_mask].copy(); rec.pop[cold_mask] = 1.0
            recd_c = {rec.idx[t] for t, _, _ in rec.recommend(prefix, candidates=cold_cands, k=K)}
            rec.pop[cold_mask] = saved
            cold_r.append(len(recd_c & held_cold) / min(K, len(held_cold)))
    print(f"routed recall@{K}:  WARM {np.mean(warm_r):.3f} (target 0.149)   COLD {np.mean(cold_r):.3f} (target 0.046)")


def main() -> int:
    conn = sqlite3.connect(DB)
    rows = conn.execute("SELECT DISTINCT user_id FROM sc_likes").fetchall()
    emb_ids = set(pickle.loads(EMB.read_bytes()))
    tl = defaultdict(int)
    for u, t in conn.execute("SELECT user_id, track_id FROM sc_likes"):
        if t in emb_ids:
            tl[u] += 1
    users = sorted(u for u, n in tl.items() if n >= 8)
    test = set(users[::5]); train = [u for u in users if u not in test]
    rec = TasteRecommender().fit(conn, train_users=train)
    rec.save()
    print(f"fit on {len(train)} users, saved -> {MODEL}")
    _eval(rec, sorted(test))
    # demo
    demo = sorted(test)[0]
    seq = rec._tl[demo]; prefix = [rec.ids[i] for i in seq[:int(len(seq) * 0.8)]]
    print(f"\ndemo recommend for a held-out user ({len(prefix)} prefix likes):")
    id2title = {r[0]: r[1] for r in conn.execute("SELECT track_id, MAX(track_title) FROM sc_likes GROUP BY track_id")}
    for tid, p, path in rec.recommend(prefix, k=8):
        print(f"  [{path}] p={p:.2f}  {str(id2title.get(tid,''))[:48]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
