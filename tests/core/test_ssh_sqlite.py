"""Signature + command-construction test for the ssh_sqlite primitive
(core/ssh_sqlite.py) — extracted out of labeling/acquire/pull_set_for_alignment.py
and labeling/enrich_gt_track_ids.py's duplicate `_ssh_sql` (2026-07 identity-stage
refactor). No network/SSH is exercised — subprocess.run is monkeypatched.
"""

from __future__ import annotations

from typing import Any

import pytest

from core import ssh_sqlite as ssh_sqlite_mod
from core.ssh_sqlite import PI_DB, PI_HOST, ssh_sqlite


class _FakeCompletedProcess:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout


def test_default_host_db_and_json_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(cmd, *, input, capture_output, text, check):  # noqa: ANN001
        captured["cmd"] = cmd
        captured["input"] = input
        return _FakeCompletedProcess('[{"a": 1, "b": "x"}]')

    monkeypatch.setattr(ssh_sqlite_mod.subprocess, "run", fake_run)

    rows = ssh_sqlite("SELECT 1;")

    assert rows == [{"a": 1, "b": "x"}]
    assert captured["cmd"] == [
        "ssh",
        "-o",
        "ConnectTimeout=15",
        PI_HOST,
        f"sqlite3 {PI_DB}",
    ]
    assert ".mode json" in captured["input"]
    assert "SELECT 1;" in captured["input"]


def test_custom_host_db_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(cmd, *, input, capture_output, text, check):  # noqa: ANN001
        captured["cmd"] = cmd
        return _FakeCompletedProcess("")

    monkeypatch.setattr(ssh_sqlite_mod.subprocess, "run", fake_run)

    rows = ssh_sqlite("SELECT 1;", host="pi-worker", db="/tmp/other.db")

    assert rows == []
    assert captured["cmd"] == [
        "ssh",
        "-o",
        "ConnectTimeout=15",
        "pi-worker",
        "sqlite3 /tmp/other.db",
    ]
