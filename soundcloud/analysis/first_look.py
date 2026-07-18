"""First-look descriptive report over the SoundCloud lake — validation, not research."""

from __future__ import annotations

import argparse
from pathlib import Path

from soundcloud import store
from soundcloud.config import load_settings


def report(db_path: Path) -> dict:
    with store.connect(db_path) as conn:
        top = conn.execute(
            """
            SELECT t.owner_sc_user_id AS owner, COUNT(*) AS n
            FROM sc_likes l JOIN sc_tracks t ON t.sc_track_id = l.sc_track_id
            GROUP BY t.owner_sc_user_id ORDER BY n DESC, owner ASC
            """
        ).fetchall()
        genres = conn.execute(
            """
            SELECT t.genre AS g, COUNT(*) AS n
            FROM sc_likes l JOIN sc_tracks t ON t.sc_track_id = l.sc_track_id
            WHERE t.genre IS NOT NULL GROUP BY t.genre
            """
        ).fetchall()
        overlap = conn.execute(
            """
            SELECT COUNT(DISTINCT t.owner_sc_user_id)
            FROM sc_likes l JOIN sc_tracks t ON t.sc_track_id = l.sc_track_id
            JOIN sc_follows f ON f.followee_sc_user_id = t.owner_sc_user_id
            """
        ).fetchone()[0]
        sizes = conn.execute("SELECT track_count FROM sc_playlists").fetchall()
    return {
        "top_liked_owners": [(r["owner"], r["n"]) for r in top],
        "genre_distribution": {r["g"]: r["n"] for r in genres},
        "following_like_overlap": overlap,
        "playlist_sizes": [r["track_count"] for r in sizes],
    }


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="SoundCloud lake first-look").parse_args(argv)
    r = report(load_settings().db_path)
    print(f"top liked owners: {r['top_liked_owners'][:10]}")
    print(f"genre distribution: {r['genre_distribution']}")
    print(f"following↔like overlap (owners): {r['following_like_overlap']}")
    print(f"playlists: {len(r['playlist_sizes'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
