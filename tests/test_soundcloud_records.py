# tests/test_soundcloud_records.py
"""Tests for soundcloud.records dataclasses."""

from __future__ import annotations

import dataclasses

import pytest

from soundcloud.records import CrawlPolicy, DEFAULT_ENTITIES, ScTrack, ScUser


def test_records_are_frozen():
    u = ScUser(
        sc_user_id=1,
        permalink="p",
        username="u",
        followers_count=None,
        followings_count=None,
        verified=False,
        city=None,
        country=None,
        description=None,
        fetched_at="2026-07-18T00:00:00Z",
        raw_ref="raw/users/1/x.jsonl",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        u.username = "x"  # type: ignore[misc]


def test_crawl_policy_defaults():
    p = CrawlPolicy(seed_user_ids=(327506308,))
    assert p.depth == 1
    assert p.entity_types == DEFAULT_ENTITIES
    assert "followers" in p.entity_types


def test_track_holds_owner_id():
    t = ScTrack(
        sc_track_id=9,
        title="T",
        owner_sc_user_id=1,
        genre="house",
        tag_list=None,
        duration_ms=1000,
        playback_count=None,
        likes_count=None,
        created_at=None,
        permalink=None,
        fetched_at="2026-07-18T00:00:00Z",
        raw_ref="r",
    )
    assert t.owner_sc_user_id == 1
