# tests/test_soundcloud_crawl.py
"""Tests for soundcloud.crawl depth-1 frontier via an injected fake fetch."""

from __future__ import annotations

from soundcloud import crawl, store
from soundcloud.config import load_settings
from soundcloud.records import CrawlPolicy


class FakeFetch:
    """Stand-in for soundcloud.fetch with canned data for user 7."""

    def user(self, c, rl, cid, uid):
        return {"id": uid, "permalink": f"u{uid}", "username": f"U{uid}"}

    def user_likes(self, c, rl, cid, uid):
        yield {
            "id": 100,
            "title": "L",
            "user": {"id": 42},
            "created_at": "2020/01/01 00:00:00 +0000",
        }

    def user_reposts(self, c, rl, cid, uid):
        yield {
            "type": "track-repost",
            "track": {"id": 200, "title": "R", "user": {"id": 43}},
            "created_at": "2020/02/02 00:00:00 +0000",
        }

    def user_playlists(self, c, rl, cid, uid):
        yield {
            "id": 300,
            "title": "P",
            "user": {"id": uid},
            "track_count": 1,
            "tracks": [{"id": 301, "title": "PT", "user": {"id": 44}}],
        }

    def user_tracks(self, c, rl, cid, uid):
        yield {"id": 400, "title": "Own", "user": {"id": uid}}

    def user_followings(self, c, rl, cid, uid):
        yield {"id": 500, "permalink": "u500", "username": "U500"}

    def user_followers(self, c, rl, cid, uid):
        yield {"id": 600, "permalink": "u600", "username": "U600"}


def test_depth1_pulls_seed_collections_no_recurse(tmp_path, monkeypatch):
    monkeypatch.setenv("SC_LAKE_ROOT", str(tmp_path))
    settings = load_settings()
    store.init_db(settings.db_path)
    policy = CrawlPolicy(seed_user_ids=(7,), depth=1)
    with store.connect(settings.db_path) as conn:
        counts = crawl.crawl(
            conn,
            settings,
            policy,
            object(),
            object(),
            "CID",
            "2026-07-18T00:00:00Z",
            fetch_mod=FakeFetch(),
        )
        conn.commit()
        n_users = conn.execute("SELECT COUNT(*) FROM sc_users").fetchone()[0]
        n_likes = conn.execute("SELECT COUNT(*) FROM sc_likes").fetchone()[0]
        n_follows = conn.execute("SELECT COUNT(*) FROM sc_follows").fetchone()[0]
        # neighbor 500 is a node but was NOT itself crawled (no likes for 500)
        n_likes_of_500 = conn.execute(
            "SELECT COUNT(*) FROM sc_likes WHERE sc_user_id=500"
        ).fetchone()[0]
    assert n_likes == 1
    assert n_follows == 2  # following 500 + follower 600
    assert n_users >= 3  # seed 7 + 500 + 600 (+ owners)
    assert n_likes_of_500 == 0
    assert counts["likes"] == 1


def test_resume_skips_completed_phase(tmp_path, monkeypatch):
    monkeypatch.setenv("SC_LAKE_ROOT", str(tmp_path))
    settings = load_settings()
    store.init_db(settings.db_path)
    policy = CrawlPolicy(seed_user_ids=(7,), depth=1)
    with store.connect(settings.db_path) as conn:
        crawl.crawl(
            conn,
            settings,
            policy,
            object(),
            object(),
            "CID",
            "2026-07-18T00:00:00Z",
            fetch_mod=FakeFetch(),
        )
        conn.commit()

        class Boom(FakeFetch):
            def user_likes(self, c, rl, cid, uid):
                raise AssertionError("likes phase should have been skipped on resume")

        # Should not raise — likes already checkpointed done.
        crawl.crawl(
            conn,
            settings,
            policy,
            object(),
            object(),
            "CID",
            "2026-07-18T00:00:01Z",
            fetch_mod=Boom(),
        )
        conn.commit()
