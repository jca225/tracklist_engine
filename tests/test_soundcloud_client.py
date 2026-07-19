# tests/test_soundcloud_client.py
"""Tests for soundcloud.client transport helpers (no live network)."""

from __future__ import annotations

import json
from pathlib import Path

from soundcloud import client as sc

FIX = Path(__file__).resolve().parent / "fixtures"


class FakeResp:
    def __init__(self, payload, status=200, text=None):
        self._payload = payload
        self.status_code = status
        self.text = text if text is not None else json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError("err", request=None, response=None)


def test_next_url_joins_client_id_with_amp():
    assert sc.next_url("https://x/y?a=1", "CID") == "https://x/y?a=1&client_id=CID"
    assert sc.next_url("https://x/y", "CID") == "https://x/y?client_id=CID"


def test_extract_client_id_finds_in_script(monkeypatch):
    home = '<script src="https://a-cdn.sndcdn.com/assets/app.js"></script>'
    js = 'foo,client_id:"abcdef0123456789abcd",bar'
    calls = {"n": 0}

    def fake_rl_get(client, rl, url, **kw):
        calls["n"] += 1
        return FakeResp(None, text=home if url == sc.SC_HOME else js)

    monkeypatch.setattr(sc, "rl_get", fake_rl_get)
    cid = sc.extract_client_id(object(), object())
    assert cid == "abcdef0123456789abcd"


def test_resolve_returns_any_kind(monkeypatch):
    payload = json.loads((FIX / "sc_resolve_user.json").read_text())
    monkeypatch.setattr(sc, "rl_get", lambda *a, **k: FakeResp(payload))
    out = sc.resolve(object(), object(), "CID", "https://soundcloud.com/user-327506308")
    assert out["kind"] == "user"
    assert out["id"] == 327506308


def test_shim_reexports_and_resolve_track_asserts():
    from personalization import soundcloud_client as shim

    assert shim.RateLimiter is sc.RateLimiter
    assert hasattr(shim, "resolve_track")
