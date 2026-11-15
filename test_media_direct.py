"""Direct-to-engine uploads: the browser streams the picked clip straight to
the engine (through the reverse proxy), addressed by absolute offset.

These tests pin the contract the JS relies on: byte-exact slice assembly,
retry-safe absolute offsets, completeness reporting, the size cap, and
LRU eviction on small disks.
"""
import os
import sys
import uuid

import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

from fastapi.testclient import TestClient  # noqa: E402

from backend import main  # noqa: E402

client = TestClient(main.app)


@pytest.fixture(autouse=True)
def _isolated_temp_dir(tmp_path, monkeypatch):
    # All media reads/writes in this module see a sandboxed dir — never the
    # real temp/ (which holds the dev's clips).
    monkeypatch.setattr(main, "TEMP_DIR", str(tmp_path))
    yield


def _mid() -> str:
    # Unique per test; matches the engine's own id alphabet.
    return "t" + uuid.uuid4().hex[:19]


def _upload(media_id: str, body: bytes, offset: int = 0, total: int = 0,
            ext: str = ".mp4"):
    return client.post(
        f"/media/{media_id}/upload?offset={offset}&total={total}&ext={ext}",
        content=body,
        headers={"Content-Type": "application/octet-stream"},
    )


class TestRawUpload:
    def test_ping(self):
        r = client.get("/media/ping")
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_slices_assemble_byte_exact(self):
        mid = _mid()
        payload = bytes(range(256)) * 20000  # 5,120,000 bytes
        half = len(payload) // 2
        assert _upload(mid, payload[:half], offset=0, total=len(payload)).status_code == 200
        r = _upload(mid, payload[half:], offset=half, total=len(payload))
        assert r.status_code == 200
        assert r.json()["complete"] is True
        with open(main._stored_media_path(mid), "rb") as fh:
            assert fh.read() == payload
        # The app polls this to unlock the render.
        st = client.get(f"/media/{mid}/status").json()
        assert st["complete"] is True
        assert st["size"] == len(payload)
        assert st["total"] == len(payload)

    def test_offset_must_match_stored_bytes(self):
        mid = _mid()
        _upload(mid, b"x" * 100, offset=0, total=1000)
        r = _upload(mid, b"y" * 100, offset=200, total=1000)
        assert r.status_code == 409

    def test_restart_from_zero_after_gap(self):
        mid = _mid()
        _upload(mid, b"garbage-" * 10, offset=0, total=80)
        # offset=0 always overwrites — no stale bytes survive a restart.
        assert _upload(mid, b"fresh" * 16, offset=0, total=80).status_code == 200
        r = _upload(mid, b"", offset=80, total=80)
        assert r.json()["complete"] is True
        with open(main._stored_media_path(mid), "rb") as fh:
            assert fh.read()[:5] == b"fresh"

    def test_oversize_rejected(self, monkeypatch):
        mid = _mid()
        monkeypatch.setattr(main, "MEDIA_MAX_FILE_MB", 1)
        r = _upload(mid, b"x", offset=0, total=2 * 1024 * 1024)
        assert r.status_code == 413

    def test_bad_media_id(self):
        assert client.get("/media/bad id with spaces!/status").status_code == 400
        assert client.post(
            "/media/bad%20id/upload?offset=0&total=10&ext=.mp4",
            content=b"1234567890").status_code == 400

    def test_extension_is_sanitized(self):
        mid = _mid()
        _upload(mid, b"data", offset=0, total=4, ext=".mp4\"; DROP TABLE")
        path = main._stored_media_path(mid)
        assert path.endswith(".mp4")

    def test_lru_sweep_evicts_oldest(self, monkeypatch, tmp_path):
        monkeypatch.setattr(main, "MEDIA_DISK_CAP_MB", 1)  # ~1MB sweep cap
        old = os.path.join(str(tmp_path), "media_" + uuid.uuid4().hex[:12] + ".mp4")
        new = os.path.join(str(tmp_path), "media_" + uuid.uuid4().hex[:12] + ".mp4")
        with open(old, "wb") as fh:
            fh.write(b"o" * (768 * 1024))
        os.utime(old, (1000, 1000))
        with open(new, "wb") as fh:
            fh.write(b"n" * (768 * 1024))
        main._sweep_media_dir()
        assert not os.path.exists(old), "oldest copy must be evicted"
        assert os.path.exists(new), "newest copy must survive"

    def test_sweep_never_evicts_fresh_files(self, monkeypatch, tmp_path):
        monkeypatch.setattr(main, "MEDIA_DISK_CAP_MB", 1)  # ~1MB sweep cap
        fresh = os.path.join(str(tmp_path), "media_" + uuid.uuid4().hex[:12] + ".mp4")
        stale = os.path.join(str(tmp_path), "media_" + uuid.uuid4().hex[:12] + ".mp4")
        with open(fresh, "wb") as fh:
            fh.write(b"f" * (800 * 1024))
        with open(stale, "wb") as fh:
            fh.write(b"s" * (800 * 1024))
        os.utime(stale, (1000, 1000))  # ancient
        main._sweep_media_dir()
        assert os.path.exists(fresh), "files younger than 10 minutes must survive the sweep"
        assert not os.path.exists(stale), "ancient files are evicted first"


class TestCarrier:
    def test_parse_carrier_roundtrip(self):
        import json
        from frontend import media_store

        doc = {"name": "recitation.mp4", "type": "video/mp4", "size": 40000000,
               "is_video": True, "media_id": "abc123", "video_media_id": "def456"}
        parsed = media_store.parse_carrier(json.dumps(doc).encode())
        assert parsed["media_id"] == "abc123"
        assert parsed["video_media_id"] == "def456"
        assert parsed["size"] == 40000000
        assert parsed["is_video"] is True
        assert media_store.is_carrier("caption_carrier.qcap")

    def test_parse_carrier_rejects_junk(self):
        from frontend import media_store

        assert media_store.parse_carrier(b"") is None
        assert media_store.parse_carrier(b"not json") is None
        assert media_store.parse_carrier(b'{"media_id": "bad id!!", "name": "x"}') is None
        assert media_store.parse_carrier(b'{"name": "x"}') is None
        assert media_store.parse_carrier(b"[" + b"1" * 5000 + b"]") is None

    def test_engine_media_status_shape(self, monkeypatch):
        from frontend import media_store

        class R:
            status_code = 200

            def json(self):
                return {"size": 5, "total": 10, "complete": False}

        monkeypatch.setattr(media_store.requests, "get", lambda *a, **k: R())
        st = media_store.engine_media_status("http://x", "mid")
        assert st == {"size": 5, "total": 10, "complete": False}

        def boom(*a, **k):
            raise RuntimeError("down")

        monkeypatch.setattr(media_store.requests, "get", boom)
        assert media_store.engine_media_status("http://x", "mid") is None
