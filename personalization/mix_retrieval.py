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
    341496235:  ("BB12",     "festival EDM"),   # Big Bootie + Hardwell are empirically one
    317238901:  ("BB11",     "festival EDM"),   # scene (confusion matrix: indistinguishable crowd)
    103828635:  ("Hardwell", "festival EDM"),
    1801876378: ("Murph",    "house/club"),
    290335129:  ("RL Grime", "trap/bass"),
    218161077:  ("JAUZ",     "bass house"),
    156449334:  ("Porter",   "melodic"),
    254813618:  ("Kygo",     "tropical"),
    1833302775: ("DomDolla", "tech house"),
    2010642907: ("RUFUS",    "organic"),
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

    # fan-lift fingerprint: per anchor, how OVER-represented each song is among its fans
    # vs the global base rate. Cancels "Big Bootie gravity" (universally-liked tracks add
    # nothing distinctive). score(prefix, a) = mean_p log( P(p | fans_a) / P(p) ).
    Mcsc = M.tocsc()
    n_train = M.shape[0]
    base = pop / n_train                                      # global like-rate per song
    fp = {}                                                   # anchor_ix -> log-lift vector (N,)
    for ix in anchor_ixs:
        fans = Mcsc.getcol(ix).nonzero()[0]                  # train users who liked this anchor
        if len(fans) == 0:
            fp[ix] = np.full(N, -1e9); continue
        rate = np.asarray(M[fans].mean(0)).ravel()           # P(song | fans of a)
        fp[ix] = np.log((rate + 1e-4) / (base + 1e-4))

    scene = {ix: ANCHORS[anchor_ix[ix]][1] for ix in anchor_ixs}   # vocab_ix -> scene
    cf_top1, lift_top1, pop_top1, rnd, cf_mrr, lift_mrr = [], [], [], [], [], []
    cf_scene, lift_scene, pop_scene = [], [], []
    confusion = defaultdict(lambda: defaultdict(int))        # for the LIFT scorer
    for i in sorted(test):
        u, target_ix, prefix = examples[i]
        cands = anchor_ixs
        if target_ix not in cands:
            continue
        cf = S[prefix][:, cands].sum(0)                      # raw co-occurrence (BB-gravity prone)
        lift = np.array([fp[ix][prefix].mean() for ix in cands])   # popularity-discounted fan-lift
        cf_order = [cands[j] for j in np.argsort(-cf)]
        lift_order = [cands[j] for j in np.argsort(-lift)]
        cf_top1.append(int(cf_order[0] == target_ix)); cf_mrr.append(1.0 / (1 + cf_order.index(target_ix)))
        lift_top1.append(int(lift_order[0] == target_ix)); lift_mrr.append(1.0 / (1 + lift_order.index(target_ix)))
        pop_top1.append(int(pop_order[0] == target_ix)); rnd.append(1.0 / len(cands))
        cf_scene.append(int(scene[cf_order[0]] == scene[target_ix]))
        lift_scene.append(int(scene[lift_order[0]] == scene[target_ix]))
        pop_scene.append(int(scene[pop_order[0]] == scene[target_ix]))
        confusion[ANCHORS[anchor_ix[target_ix]][0]][ANCHORS[anchor_ix[lift_order[0]]][0]] += 1

    n = len(cf_top1)
    # macro scene-accuracy: average per-scene recall (each scene weighted equally, so the
    # huge BB cohort can't dominate). Computed from the FAN-LIFT confusion (by-mix tallies).
    scene_of = {lbl: ANCHORS[a][1] for a, (lbl, _) in ANCHORS.items()}
    sc_correct, sc_total = defaultdict(int), defaultdict(int)
    for true_lbl, guesses in confusion.items():
        s = scene_of[true_lbl]
        for guess_lbl, c in guesses.items():
            sc_total[s] += c
            if scene_of[guess_lbl] == s:
                sc_correct[s] += c
    macro = np.mean([sc_correct[s] / sc_total[s] for s in sc_total])

    print(f"history -> which-mix retrieval  ({n} held-out users, {len(ANCHORS)} candidate mixes)\n")
    print(f"  RANDOM        mix-top-1 {np.mean(rnd):.3f}")
    print(f"  POPULARITY    mix-top-1 {np.mean(pop_top1):.3f}   scene-top-1 {np.mean(pop_scene):.3f}")
    print(f"  CO-OCCUR      mix-top-1 {np.mean(cf_top1):.3f}   scene-top-1 {np.mean(cf_scene):.3f}   MRR {np.mean(cf_mrr):.3f}")
    print(f"  FAN-LIFT      mix-top-1 {np.mean(lift_top1):.3f}   scene-top-1 {np.mean(lift_scene):.3f}   MRR {np.mean(lift_mrr):.3f}")
    print(f"  FAN-LIFT macro scene-top-1 {macro:.3f}   (per-scene recall, equal-weighted — the fair metric)")
    print("\n  per-scene recall (FAN-LIFT):")
    for s in sorted(sc_total, key=lambda s: -sc_correct[s] / sc_total[s]):
        print(f"    {s:14s} {sc_correct[s]/sc_total[s]:.2f}  ({sc_correct[s]}/{sc_total[s]})")
    print("  (BB11 & BB12 are one scene — same crowd. confusion below = FAN-LIFT.)")
    print("\nconfusion (row=true mix, col=CF's top guess):")
    labels = [ANCHORS[a][0] for a in ANCHORS]
    print("           " + "".join(f"{l:>9}" for l in labels))
    for tl_ in labels:
        print(f"  {tl_:>8} " + "".join(f"{confusion[tl_][gl]:>9}" for gl in labels))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
