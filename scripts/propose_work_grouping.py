#!/usr/bin/env python3
"""Propose sibling grouping of recordings under a shared ``work`` (DRY-RUN).

Today every recording is its own singleton work (``work_id == recording_id`` for
all 18,812), so nothing links "Roses" ↔ "Roses (Acappella)" or "Emily" ↔ "Emily
(Remix)". This proposes a conservative grouping so version/stem/variant siblings
share a work — which (a) fixes the data-model gap and (b) lets the aligner scorer
credit "right song, wrong version" as a near-miss instead of a full identity miss.

**Conservative by design.** Two recordings are proposed as siblings iff their work
rows share an EXACT normalized ``(title, artist-set)`` key. The `work.title` /
`work.artist` are already the clean song title + primary artist (version qualifiers
live on the ``version`` / ``stem`` / ``variant`` axes, not the title). Exact-key
matching therefore UNDER-merges (a slightly different title spelling stays split) —
the safe failure mode — rather than fusing two genuinely different songs.

**DRY-RUN ONLY.** Writes a proposal JSON + prints stats + a review sample. It does
NOT touch the canonical DB. Apply is a separate, reviewed step (see --emit-apply-sql
which only PRINTS the UPDATEs it would run).

Usage:
  python scripts/propose_work_grouping.py --db /mnt/storage/data/db/music_database.db \
      --out /tmp/work_grouping_proposal.json [--sample 25] [--emit-apply-sql]
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict

_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def _norm_title(t: str | None) -> str:
    t = (t or "").lower().strip()
    t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t).strip()


def _norm_artist(artist_json: str | None) -> str:
    """Sorted, normalized artist set from the work.artist JSON array."""
    try:
        arr = json.loads(artist_json) if artist_json else []
    except (json.JSONDecodeError, TypeError):
        arr = [artist_json] if artist_json else []
    names = sorted(_WS.sub(" ", _PUNCT.sub(" ", str(a).lower())).strip() for a in arr)
    return "|".join(n for n in names if n)


def build_proposal(conn: sqlite3.Connection) -> dict:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT r.recording_id, r.work_id, r.version, r.version_artist,
               w.title AS title, w.artists_json AS artist
        FROM recording r JOIN work w ON w.work_id = r.work_id
        """
    ).fetchall()
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        key = (_norm_title(r["title"]), _norm_artist(r["artist"]))
        if not key[0]:  # never group on an empty title
            key = (f"__EMPTY__{r['recording_id']}", "")
        by_key[key].append(
            dict(
                rec=r["recording_id"],
                work=r["work_id"],
                version=r["version"],
                title=r["title"],
            )
        )
    clusters = {}
    for (title, artist), recs in by_key.items():
        if len(recs) < 2:
            continue
        canonical = min(r["work"] for r in recs)  # deterministic
        clusters[canonical] = dict(
            title=title,
            artist=artist,
            recordings=sorted(r["rec"] for r in recs),
            versions=[r["version"] for r in recs],
        )
    n_recs = len(rows)
    affected = sum(len(c["recordings"]) for c in clusters.values())
    return dict(
        n_recordings=n_recs,
        n_singleton_works_now=n_recs,
        n_clusters_merging=len(clusters),
        n_recordings_affected=affected,
        n_works_after=n_recs - (affected - len(clusters)),
        clusters=clusters,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True, help="path or file:...?immutable=1 URI")
    ap.add_argument("--out", default="/tmp/work_grouping_proposal.json")
    ap.add_argument("--sample", type=int, default=25)
    ap.add_argument(
        "--emit-apply-sql",
        action="store_true",
        help="PRINT (do not run) the UPDATE statements an apply would use",
    )
    args = ap.parse_args(argv)

    conn = sqlite3.connect(args.db, uri=args.db.startswith("file:"))
    try:
        p = build_proposal(conn)
    finally:
        conn.close()

    clusters = p.pop("clusters")
    with open(args.out, "w") as fh:
        json.dump({**p, "clusters": clusters}, fh, indent=0)

    sizes = defaultdict(int)
    for c in clusters.values():
        sizes[len(c["recordings"])] += 1
    print("=== work-grouping proposal (DRY-RUN) ===")
    for k, v in p.items():
        print(f"  {k}: {v}")
    print(f"  cluster-size distribution: {dict(sorted(sizes.items()))}")
    print(
        f"\n  sample of {args.sample} proposed sibling clusters "
        "(eyeball for false merges - versions should differ):"
    )
    for canon, c in list(clusters.items())[: args.sample]:
        vs = ",".join(str(v) for v in c["versions"])
        print(
            f'   work {canon}: "{c["title"]}" [{c["artist"]}] '
            f"{len(c['recordings'])} recs (versions: {vs})"
        )

    if args.emit_apply_sql:
        print("\n=== APPLY SQL (NOT executed — review before running) ===")
        for canon, c in clusters.items():
            ids = ",".join(f'"{r}"' for r in c["recordings"] if r != canon)
            if ids:
                print(
                    f'UPDATE recording SET work_id="{canon}" '
                    f"WHERE recording_id IN ({ids});"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
