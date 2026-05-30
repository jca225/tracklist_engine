"""Tests for scripts/ingest_stem_url.py remote command builder."""
from __future__ import annotations

import argparse

from scripts import ingest_stem_url as isu


def _args(**kwargs: object) -> argparse.Namespace:
    base = dict(
        url=None,
        file=None,
        track_audio_id=None,
        track_id=None,
        role=None,
        promote=False,
        set_id="set1",
        position="022",
        reason="quality:good|identity:OK",
        player_id=None,
        pull=False,
        aligning_dest="~/aligning",
        fail_on="",
        dry_run=False,
        no_log=False,
        skip_preflight=True,
    )
    base.update(kwargs)
    return argparse.Namespace(**base)


def test_add_url_no_promote_by_default():
    a = _args(url="https://www.youtube.com/watch?v=abc", track_id="tid1", role="acappella")
    cmd = isu.build_remote_command(a)
    assert cmd[0] == "scripts/acquire_variant.py"
    assert "https://www.youtube.com/watch?v=abc" in cmd
    assert "--no-promote-reference" in cmd
    assert "--promote" not in cmd
    assert "--role" in cmd and "acappella" in cmd


def test_add_url_with_promote():
    a = _args(
        url="https://youtu.be/abc",
        track_id="tid1",
        role="instrumental",
        promote=True,
    )
    cmd = isu.build_remote_command(a)
    assert "--no-promote-reference" not in cmd


def test_replace_uses_replace_stem_audio():
    a = _args(
        url="https://www.youtube.com/watch?v=xyz",
        track_audio_id=4011,
    )
    cmd = isu.build_remote_command(a)
    assert cmd[0] == "scripts/replace_stem_audio.py"
    assert "--track-audio-id" in cmd
    assert "4011" in cmd
    assert "--no-promote-reference" not in cmd


def test_add_file_remote_path():
    a = _args(track_id="tid1", role="acappella")
    cmd = isu.build_remote_command(a, remote_file="/tmp/ingest_stem_x.m4a")
    assert "--file" in cmd
    assert "/tmp/ingest_stem_x.m4a" in cmd


def test_remote_shell_wraps_repo():
    parts = ["scripts/acquire_variant.py", "https://x", "--role", "acappella"]
    shell = isu._remote_shell(parts)
    assert "cd ~/tracklist_engine" in shell
    assert "venvs/audio/bin/python" in shell


def test_fail_on_mapping():
    blocked = isu._fail_on_set("fallback,wrong_song")
    assert "FALLBACK_TO_ORIGINAL" in blocked
    assert "WRONG_SONG" in blocked


def test_parse_verdict():
    out = "identity-check [WRONG_SONG]: too low\n"
    assert isu._parse_verdict(out) == "WRONG_SONG"
