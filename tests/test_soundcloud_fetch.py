# tests/test_soundcloud_fetch.py
"""Tests for soundcloud.fetch pagination + endpoint routing (fake client)."""

from __future__ import annotations

from soundcloud import fetch


class FakeResp:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


class FakeClient:
    """Serves queued responses; records requested URLs."""

    def __init__(self, pages):
        self._pages = list(pages)
        self.urls: list[str] = []

    def get(self, url, params=None, **kw):
        self.urls.append(url)
        return FakeResp(self._pages.pop(0))


class NullRL:
    def wait(self):
        return None


def test_paged_follows_next_href_until_empty():
    pages = [
        {
            "collection": [{"id": 1}, {"id": 2}],
            "next_href": "https://api-v2.soundcloud.com/users/7/likes?cursor=abc",
        },
        {"collection": [{"id": 3}], "next_href": None},
    ]
    client = FakeClient(pages)
    items = list(fetch.user_likes(client, NullRL(), "CID", 7))
    assert [i["id"] for i in items] == [1, 2, 3]
    # first call hits the likes path; second call follows next_href with client_id appended
    assert "/users/7/likes" in client.urls[0]
    assert "client_id=CID" in client.urls[1]


def test_user_single_object():
    client = FakeClient([{"id": 7, "username": "John"}])
    out = fetch.user(client, NullRL(), "CID", 7)
    assert out["username"] == "John"


def test_paged_stops_on_skip_status():
    import httpx

    class Boom(FakeClient):
        def get(self, url, params=None, **kw):
            self.urls.append(url)
            resp = FakeResp({})
            resp.status_code = 404

            def raise_for_status():
                raise httpx.HTTPStatusError(
                    "nf",
                    request=httpx.Request("GET", url),
                    response=httpx.Response(404, request=httpx.Request("GET", url)),
                )

            resp.raise_for_status = raise_for_status
            return resp

    items = list(fetch.user_tracks(Boom([{}]), NullRL(), "CID", 7))
    assert items == []
