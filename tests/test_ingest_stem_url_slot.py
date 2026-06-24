"""_resolve_slot_on_pi must never *guess* a slot key.

It backs batch acquisition-case logging (ingest_candidate_winners /
apply_stem_matches pass --set-id + --track-id but no --position). The safety
property: a unique slot resolves; anything ambiguous, absent, or failing returns
None so the caller skips logging rather than mis-keying the case.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

import scripts.ingest_stem_url as isu


@dataclass
class _FakeProc:
    returncode: int
    stdout: str
    stderr: str = ""


def _patch(monkeypatch, *, returncode: int, stdout: str):
    def fake_run(*a, **k):
        return _FakeProc(returncode=returncode, stdout=stdout)

    monkeypatch.setattr(subprocess, "run", fake_run)


def test_unique_slot_resolves(monkeypatch):
    _patch(monkeypatch, returncode=0, stdout="081\n")
    assert isu._resolve_slot_on_pi("bb11", "rec1") == "081"


def test_ambiguous_multi_slot_returns_none(monkeypatch):
    # Same recording fills two slots -> don't guess a key.
    _patch(monkeypatch, returncode=0, stdout="081\n112\n")
    assert isu._resolve_slot_on_pi("bb11", "rec1") is None


def test_no_slot_returns_none(monkeypatch):
    _patch(monkeypatch, returncode=0, stdout="\n")
    assert isu._resolve_slot_on_pi("bb11", "rec1") is None


def test_query_failure_returns_none(monkeypatch):
    _patch(monkeypatch, returncode=1, stdout="")
    assert isu._resolve_slot_on_pi("bb11", "rec1") is None


def test_ssh_exception_returns_none(monkeypatch):
    def boom(*a, **k):
        raise OSError("ssh down")

    monkeypatch.setattr(subprocess, "run", boom)
    assert isu._resolve_slot_on_pi("bb11", "rec1") is None
