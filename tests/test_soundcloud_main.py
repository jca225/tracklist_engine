# tests/test_soundcloud_main.py
"""Tests for soundcloud.main CLI dispatch (crawl injected via monkeypatch)."""

from __future__ import annotations

from soundcloud import main as sc_main
from soundcloud import store


def test_stats_runs_on_empty_db(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SC_LAKE_ROOT", str(tmp_path))
    rc = sc_main.main(["stats"])
    assert rc == 0
    assert "sc_users" in capsys.readouterr().out


def test_crawl_subcommand_invokes_driver(tmp_path, monkeypatch):
    monkeypatch.setenv("SC_LAKE_ROOT", str(tmp_path))
    seen = {}

    def fake_crawl(conn, settings, policy, client, rl, cid, now, fetch_mod=None):
        seen["seed"] = policy.seed_user_ids
        seen["depth"] = policy.depth
        return {
            "users": 1,
            "tracks": 0,
            "likes": 0,
            "reposts": 0,
            "playlists": 0,
            "follows": 0,
        }

    monkeypatch.setattr(sc_main.crawl, "crawl", fake_crawl)
    monkeypatch.setattr(
        sc_main, "_bootstrap_client", lambda settings: (object(), object(), "CID")
    )
    rc = sc_main.main(
        ["crawl", "--seed", "7", "--depth", "1", "--now", "2026-07-18T00:00:00Z"]
    )
    assert rc == 0
    assert seen["seed"] == (7,)
    assert seen["depth"] == 1


def test_sync_user_resolves_profile_url(tmp_path, monkeypatch):
    monkeypatch.setenv("SC_LAKE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        sc_main, "_bootstrap_client", lambda settings: (object(), object(), "CID")
    )
    monkeypatch.setattr(
        sc_main.client,
        "resolve",
        lambda c, rl, cid, url: {"kind": "user", "id": 327506308},
    )
    captured = {}

    def fake_crawl(conn, settings, policy, *a, **k):
        captured["seed"] = policy.seed_user_ids
        return {
            "users": 1,
            "tracks": 0,
            "likes": 0,
            "reposts": 0,
            "playlists": 0,
            "follows": 0,
        }

    monkeypatch.setattr(sc_main.crawl, "crawl", fake_crawl)
    rc = sc_main.main(
        [
            "sync-user",
            "https://soundcloud.com/user-327506308",
            "--now",
            "2026-07-18T00:00:00Z",
        ]
    )
    assert rc == 0
    assert captured["seed"] == (327506308,)
