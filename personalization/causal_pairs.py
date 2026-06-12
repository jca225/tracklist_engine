"""Mine per-user causal (taste-prior -> engaged-set) pairs from the SC warehouse.

The mashup-compat sweep showed audio doesn't determine what a curator picks — the
signal is TASTE. This builds the taste-conditioned training data: for each user and
each DJ-set they engaged (a Big Bootie volume in their likes, with a real liked_at),
the PREFIX is everything they liked strictly BEFORE that engagement (the causal cut,
no future leakage), the TARGET is the set they then engaged.

One user generates several pairs (liked BB11, then BB12, then BB13 -> three examples
with growing prefixes) = the response diversity the conditional objective needs.

  venvs/audio/bin/python -m personalization.causal_pairs
Output: data/taste/causal_pairs.jsonl  (one {user, target, cut_at, prefix:[...]} per line)
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

DB = Path("data/taste/taste_warehouse.db")
OUT = Path("data/taste/causal_pairs.jsonl")
MIN_PREFIX = 5            # skip cold-start engagements
PREFIX_CAP = 200          # keep most-recent N likes before the cut (recency-weighted taste)
TARGET_LIKE = "%Big Bootie Mix%"


def main() -> int:
    conn = sqlite3.connect(DB)
    targets = {r[0]: (r[1] or "")[:60] for r in conn.execute(
        f"SELECT track_id, MAX(track_title) FROM sc_likes WHERE track_title LIKE '{TARGET_LIKE}' GROUP BY track_id")}
    print(f"target sets (BB volumes): {len(targets)}")

    rows = conn.execute(
        "SELECT user_id, liked_at, track_id FROM sc_likes ORDER BY user_id, liked_at, rowid")

    n_pairs, prefix_sizes = 0, []
    per_target: dict[str, int] = {}
    users = set()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    def flush(user, hist, out):
        nonlocal n_pairs
        for i, (t, tid) in enumerate(hist):
            if tid in targets:
                prefix = [h[1] for h in hist[:i]]          # liked strictly before the cut
                if len(prefix) >= MIN_PREFIX:
                    out.write(json.dumps({
                        "user_id": user, "target": tid, "target_title": targets[tid],
                        "cut_at": t, "n_prefix": len(prefix), "prefix": prefix[-PREFIX_CAP:],
                    }) + "\n")
                    n_pairs += 1
                    prefix_sizes.append(len(prefix))
                    per_target[targets[tid]] = per_target.get(targets[tid], 0) + 1
                    users.add(user)

    with OUT.open("w") as out:
        cur, hist = None, []
        for user, t, tid in rows:
            if user != cur:
                if cur is not None:
                    flush(cur, hist, out)
                cur, hist = user, []
            hist.append((t, tid))
        if cur is not None:
            flush(cur, hist, out)

    ps = np.array(prefix_sizes)
    print(f"\ncausal pairs: {n_pairs}  | distinct users: {len(users)}  -> {OUT}")
    print(f"prefix size: median={int(np.median(ps))}  p25={int(np.percentile(ps,25))} "
          f"p75={int(np.percentile(ps,75))}  max={int(ps.max())}")
    print("\npairs per target volume (top):")
    for title, c in sorted(per_target.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {c:>5}  {title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
