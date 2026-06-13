"""history -> which mix: does a user's music history predict WHICH DJ mix they engage?

This is the rung above taste_model_v0. taste_model_v0 asks "is taste predictable at
all" (predict a user's next song-likes). THIS asks the real question: given only the
songs a user liked BEFORE they touched any of our mixes (the causal cut, no leakage),
can we predict WHICH mix they then engaged — ranked against the other candidate mixes?

Each mix is represented by its own SoundCloud upload (the "anchor" track). Predicting
the mix = ranking its anchor among all anchors, scored by co-occurrence of the user's
prefix with each anchor's fan-taste (the validated ID-CF engine, popularity-normalized).

Baselines: RANDOM (1/n_candidates) and POPULARITY (always guess the biggest mix). CF
beating popularity = a user's *individual* history tells us which scene they belong to,
not just "everyone engages the famous mix."

  venvs/audio/bin/python -m personalization.mix_retrieval
"""
from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

DB = Path("data/taste/taste_warehouse.db")
TOPK = 8000          # song vocabulary for the co-occurrence features
MIN_PREFIX = 5       # need enough pre-engagement history to score
K = 20

# mix anchor track (its own SC upload) -> (label, scene)
ANCHORS = {
    341496235:  ("BB12",     "festival EDM"),
    317238901:  ("BB11",     "festival EDM"),
    1801876378: ("Murph",    "house/club"),
    290335129:  ("RL Grime", "bass/trap"),
    156449334:  ("Porter",   "melodic/auteur"),
}


def main() -> int:
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "SELECT user_id, liked_at, track_id FROM sc_likes ORDER BY user_id, liked_at, rowid"
    ).fetchall()

    vocab = {tid: i for i, (tid, _) in enumerate(Counter(t for _, _, t in rows).most_common(TOPK))}
    for a in ANCHORS:                                        # anchors must be in vocab
        vocab.setdefault(a, len(vocab))
    anchor_ix = {vocab[a]: a for a in ANCHORS}
    N = len(vocab)

    # per-user time-ordered timeline of (liked_at, vocab_index, raw_track_id)
    tl: dict[str, list] = defaultdict(list)
    for u, t, tid in rows:
        if tid in vocab:
            tl[u].append((t, vocab[tid], tid))

    # a labeled example: user's EARLIEST-engaged anchor = target; prefix = likes strictly before it
    examples = []  # (user, target_ix, prefix_indices)
    for u, hist in tl.items():
        first = next(((i, ix) for i, (t, ix, tid) in enumerate(hist) if tid in ANCHORS), None)
        if first is None:
            continue
        pos, target_ix = first
        prefix = list({ix for _, ix, _ in hist[:pos]} - set(anchor_ix))   # pre-engagement, non-anchor
        if len(prefix) >= MIN_PREFIX:
            examples.append((u, target_ix, prefix))

    examples.sort(key=lambda e: e[0])
    test = set(range(0, len(examples), 5))                   # 20% held out
    train_users = [examples[i][0] for i in range(len(examples)) if i not in test]

    # co-occurrence from TRAIN users' full timelines (popularity-normalized)
    uidx = {u: i for i, u in enumerate(train_users)}
    ri, ci = [], []
    for u in train_users:
        for ix in {x[1] for x in tl[u]}:
            ri.append(uidx[u]); ci.append(ix)
    M = csr_matrix((np.ones(len(ri)), (ri, ci)), shape=(len(train_users), N))
    cooc = (M.T @ M).toarray().astype(np.float32); np.fill_diagonal(cooc, 0)
    pop = np.asarray(M.sum(0)).ravel() + 1.0
    S = cooc / np.sqrt(np.outer(pop, pop))

    anchor_ixs = list(anchor_ix)                             # candidate columns
    anchor_pop = {ix: pop[ix] for ix in anchor_ixs}
    pop_order = sorted(anchor_ixs, key=lambda ix: -anchor_pop[ix])

    scene = {ix: ANCHORS[anchor_ix[ix]][1] for ix in anchor_ixs}   # vocab_ix -> scene
    cf_top1, pop_top1, rnd, cf_mrr = [], [], [], []
    cf_scene, pop_scene = [], []
    confusion = defaultdict(lambda: defaultdict(int))
    for i in sorted(test):
        u, target_ix, prefix = examples[i]
        cands = anchor_ixs                                   # rank all 5 mixes
        if target_ix not in cands:
            continue
        score = S[prefix][:, cands].sum(0)                   # CF: prefix co-occurrence w/ each anchor
        order = [cands[j] for j in np.argsort(-score)]
        cf_top1.append(int(order[0] == target_ix))
        cf_mrr.append(1.0 / (1 + order.index(target_ix)))
        pop_top1.append(int(pop_order[0] == target_ix))
        rnd.append(1.0 / len(cands))
        cf_scene.append(int(scene[order[0]] == scene[target_ix]))    # scene-level (BB11==BB12)
        pop_scene.append(int(scene[pop_order[0]] == scene[target_ix]))
        confusion[ANCHORS[anchor_ix[target_ix]][0]][ANCHORS[anchor_ix[order[0]]][0]] += 1

    n = len(cf_top1)
    print(f"history -> which-mix retrieval  ({n} held-out users, {len(ANCHORS)} candidate mixes)\n")
    print(f"  RANDOM      mix-top-1 {np.mean(rnd):.3f}")
    print(f"  POPULARITY  mix-top-1 {np.mean(pop_top1):.3f}   scene-top-1 {np.mean(pop_scene):.3f}")
    print(f"  TASTE-CF    mix-top-1 {np.mean(cf_top1):.3f}   scene-top-1 {np.mean(cf_scene):.3f}   MRR {np.mean(cf_mrr):.3f}")
    print(f"\n  CF lift over popularity:  mix {np.mean(cf_top1)/max(np.mean(pop_top1),1e-9):.2f}x"
          f"   scene {np.mean(cf_scene)/max(np.mean(pop_scene),1e-9):.2f}x")
    print("  (BB11 & BB12 are one scene — same crowd — so scene-top-1 is the honest mix-discrimination number)")
    print("\nconfusion (row=true mix, col=CF's top guess):")
    labels = [ANCHORS[a][0] for a in ANCHORS]
    print("           " + "".join(f"{l:>9}" for l in labels))
    for tl_ in labels:
        print(f"  {tl_:>8} " + "".join(f"{confusion[tl_][gl]:>9}" for gl in labels))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
