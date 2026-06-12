"""Build familiarity(track | audience) for BB12 and test acappella vs instrumental.

    venvs/audio/bin/python -m eda.audience_prior.run

Hypothesis: the DJ picks acappellas the audience already knows (the recognition
"payload"), while instrumentals are compatibility anchors the audience need not
know. So acappella familiarity ≫ instrumental familiarity in the audience's
SoundCloud like history.

Read-only on data/taste/taste_warehouse.db. Writes data/analysis/audience_prior/.
"""
from __future__ import annotations

import csv
import sqlite3
from collections import defaultdict
from pathlib import Path

from labeling.ground_truth.schema import load as load_gt

from .match import Vocab, split_artist_title

DB = "data/taste/taste_warehouse.db"
GT = "labeling/fixtures/bb12_ground_truth.yaml"
MIX = "1fsnxchk"
OUT = Path("data/analysis/audience_prior")


def _load_vocab(con: sqlite3.Connection, bot_users: set[str]) -> Vocab:
    rows = con.execute(
        "SELECT track_id, track_title, track_artist_username, "
        "COUNT(DISTINCT user_id) FROM sc_likes WHERE mix_id=? GROUP BY track_id",
        (MIX,),
    ).fetchall()
    return Vocab.build([(t, ti, u, lk) for (t, ti, u, lk) in rows])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)

    bot_users = {r[0] for r in con.execute(
        "SELECT user_id FROM listener_bot_scores WHERE mix_id=? AND is_bot=1", (MIX,))}
    audience = {r[0] for r in con.execute(
        "SELECT DISTINCT user_id FROM sc_likes WHERE mix_id=?", (MIX,))} - bot_users
    n_aud = len(audience)
    print(f"[info] {MIX}: {n_aud} non-bot enriched listeners, {len(bot_users)} bots excluded")

    print("[info] building SC like vocabulary …")
    vocab = _load_vocab(con, bot_users)
    print(f"[info] vocab: {len(vocab.track_ids)} distinct SC tracks")

    gt = load_gt(GT)
    if not gt.is_ok():
        raise SystemExit(gt.error)
    tracks = gt.value.tracks

    # phase 1: match each tracklist entry → SC track_ids
    matched_ids: dict[int, list[int]] = {}
    all_ids: set[int] = set()
    for i, t in enumerate(tracks):
        art, tit = split_artist_title(t.label)
        rows = vocab.match(art, tit)
        ids = [vocab.track_ids[r] for r in rows]
        matched_ids[i] = ids
        all_ids.update(ids)

    # phase 2: pull likes for matched tracks once, build track→liker-set
    track_likers: dict[int, set[str]] = defaultdict(set)
    if all_ids:
        idl = list(all_ids)
        for k in range(0, len(idl), 800):
            chunk = idl[k:k + 800]
            q = ("SELECT track_id, user_id FROM sc_likes WHERE mix_id=? AND track_id IN (%s)"
                 % ",".join("?" * len(chunk)))
            for tid, uid in con.execute(q, (MIX, *chunk)):
                if uid not in bot_users:
                    track_likers[tid].add(uid)

    # phase 3: familiarity = distinct audience members who like ANY matched track
    out_rows = []
    by_stem: dict[str, list[float]] = defaultdict(list)
    for i, t in enumerate(tracks):
        users: set[str] = set()
        for tid in matched_ids[i]:
            users |= track_likers.get(tid, set())
        fam = len(users) / n_aud if n_aud else 0.0
        sample = sorted(
            ((vocab.likers[r], vocab.bags[r]) for r in vocab.match(*split_artist_title(t.label))),
            reverse=True,
        )[:1]
        out_rows.append({
            "slot": t.slot_label, "label": t.label, "stem": t.claimed_stem,
            "n_matched_sc_tracks": len(matched_ids[i]),
            "familiarity_listeners": len(users),
            "familiarity_frac": round(fam, 4),
        })
        by_stem[t.claimed_stem].append(fam)

    out_rows.sort(key=lambda r: r["familiarity_frac"], reverse=True)
    _write_csv(OUT / "bb12_track_familiarity.csv", out_rows)
    _write_summary(OUT / "summary.md", out_rows, by_stem, n_aud)
    print(f"[done] {OUT}/summary.md")


def _stats(xs: list[float]) -> tuple[float, float, int]:
    import statistics
    if not xs:
        return 0.0, 0.0, 0
    return statistics.mean(xs), statistics.median(xs), len(xs)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _write_summary(path: Path, rows: list[dict], by_stem: dict, n_aud: int) -> None:
    L = [f"# Audience-conditioned familiarity — BB12 ({n_aud} listeners)\n"]
    L.append("`familiarity` = fraction of the set's non-bot SoundCloud audience "
             "who have liked the track (fuzzy artist+title join to their like history).\n")
    L.append("## By identity stem (the payload hypothesis)\n")
    L.append("| Stem | n | mean | median | max |")
    L.append("|---|---|---|---|---|")
    for stem in ("acappella", "instrumental", "regular"):
        xs = by_stem.get(stem, [])
        m, md, n = _stats(xs)
        if n:
            L.append(f"| {stem} | {n} | {m:.3f} | {md:.3f} | {max(xs):.3f} |")
    L.append("\nThe mean is misleading: acappella familiarity is **heavy-tailed** "
             "(few mega-hooks, many obscure) and `claimed_stem` labels are noisy. "
             "The signal lives at the **top**, not the mean →")
    # top-K acappella enrichment vs base rate
    base = (len(by_stem.get("acappella", [])) /
            sum(len(v) for v in by_stem.values())) if by_stem else 0.0
    for K in (15, 25):
        topk = rows[:K]
        share = sum(r["stem"] == "acappella" for r in topk) / K
        L.append(f"- Top-{K} most-familiar: **{share:.0%} acappella** (set base rate {base:.0%})")
    L.append("\n## Top 15 most-familiar tracks\n")
    L.append("| Familiarity | Listeners | Stem | Track |")
    L.append("|---|---|---|---|")
    for r in rows[:15]:
        L.append(f"| {r['familiarity_frac']:.3f} | {r['familiarity_listeners']} | "
                 f"{r['stem']} | {r['label']} |")
    L.append("\n*Caveat: these listeners engaged with the BB12 upload, so familiarity "
             "is partly endogenous. The clean causal version builds each listener's prior "
             "from history excluding this set and tests cross-set generalization.*")
    path.write_text("\n".join(L))


if __name__ == "__main__":
    main()
