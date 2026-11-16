"""Embedded /engine routes (mounted inside Streamlit's starlette app).

The dashboard-pinned start command can't be changed through the API, so
these run inside the app's own port and relay to the engine. They are a
thin proxy — these tests pin the forwarding contract and the guards.
"""
import os
import subprocess
import sys
import uuid

import pytest
from starlette.routing import Route
from starlette.applications import Starlette
from starlette.testclient import TestClient

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from backend import embedded_engine  # noqa: E402


@pytest.fixture()
def client(monkeypatch):
    calls = []

    class FakeResp:
        def __init__(self, status, payload):
            self.status_code = status
            self._payload = payload
            self.headers = {"content-type": "application/json"}

        @property
        def content(self):
            import json as _json
            return _json.dumps(self._payload).encode()

    def fake_get(url, timeout=0, **kw):
        calls.append(("GET", url))
        if url.endswith("/media/ping"):
            return FakeResp(200, {"ok": True})
        if "/media/" in url and url.endswith("/status"):
            return FakeResp(200, {"id": url.split("/media/")[1].split("/")[0],
                                  "size": 5, "total": 10, "complete": False})
        return FakeResp(404, {"detail": "nope"})

    def fake_post(url, data=None, headers=None, timeout=0, **kw):
        calls.append(("POST", url, len(data or b"")))
        return FakeResp(200, {"received": len(data or b""), "complete": False})

    monkeypatch.setattr(embedded_engine.requests, "get", fake_get)
    monkeypatch.setattr(embedded_engine.requests, "post", fake_post)

    app = Starlette(routes=[Route("/engine/media/ping", embedded_engine._ping, methods=["GET"]),
                            Route("/engine/media/{media_id}/upload", embedded_engine._upload, methods=["POST"]),
                            Route("/engine/media/{media_id}/status", embedded_engine._status, methods=["GET"])])
    c = TestClient(app)
    c.calls = calls
    return c


class TestEmbeddedRoutes:
    def test_ping_relays_engine(self, client):
        r = client.get("/engine/media/ping")
        assert r.status_code == 200
        assert r.json() == {"ok": True}

    def test_upload_forwards_slice_and_query(self, client):
        r = client.post("/engine/media/abc123/upload?offset=8&total=20&ext=.mp4",
                        content=b"12345678",
                        headers={"Content-Type": "application/octet-stream"})
        assert r.status_code == 200
        assert r.json()["received"] == 8
        assert ("POST", embedded_engine.ENGINE_URL +
                "/media/abc123/upload?offset=8&total=20&ext=.mp4", 8) in client.calls

    def test_upload_rejects_bad_id_without_hitting_engine(self, client):
        r = client.post("/engine/media/bad%20id/upload?offset=0&total=8&ext=.mp4",
                        content=b"12345678")
        assert r.status_code == 400
        assert not any(c[0] == "POST" for c in client.calls)

    def test_status_relays_engine(self, client):
        r = client.get("/engine/media/abc123/status")
        assert r.status_code == 200
        assert r.json()["total"] == 10

    def test_extra_routes_shape(self):
        def has(path, methods):
            return any(r.path == path and methods <= set(r.methods or set())
                       for r in embedded_engine.extra_routes())
        assert has("/engine/media/ping", {"GET"})
        assert has("/engine/media/{media_id}/upload", {"POST"})
        assert has("/engine/media/{media_id}/status", {"GET"})


def test_sitecustomize_imports_cleanly():
    # The hook runs at every interpreter start from the repo root; it must
    # never raise and must never print to stdout/stderr.
    out = subprocess.run(
        [sys.executable, "-c", "import sitecustomize; print('ok')"],
        capture_output=True, text=True, cwd=ROOT,
        env={**os.environ, "PYTHONPATH": ROOT})
    assert out.returncode == 0
    assert out.stdout.strip() == "ok"
    assert out.stderr.strip() == ""
