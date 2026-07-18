# tests/test_soundcloud_config.py
"""Tests for soundcloud.config env + default resolution."""

from __future__ import annotations

from pathlib import Path

from soundcloud.config import load_settings


def test_defaults_point_at_pi_storage(monkeypatch):
    monkeypatch.delenv("SC_LAKE_ROOT", raising=False)
    monkeypatch.delenv("SC_LAKE_RPM", raising=False)
    s = load_settings()
    assert s.data_root == Path("/mnt/storage/data/soundcloud")
    assert s.db_path == Path("/mnt/storage/data/soundcloud/sc_lake.db")
    assert s.raw_root == Path("/mnt/storage/data/soundcloud/raw")
    assert s.rpm == 60


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("SC_LAKE_ROOT", str(tmp_path))
    monkeypatch.setenv("SC_LAKE_RPM", "20")
    s = load_settings()
    assert s.data_root == tmp_path
    assert s.db_path == tmp_path / "sc_lake.db"
    assert s.rpm == 20


def test_explicit_args_beat_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SC_LAKE_RPM", "20")
    s = load_settings(data_root=tmp_path, rpm=5)
    assert s.data_root == tmp_path
    assert s.rpm == 5
