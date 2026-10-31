"""Large uploads must not OOM the 512MB host.

Reproduced on the live Render service (2026-10-31): the 40MB burn-in video was
sent as ONE multipart body, so Streamlit's copy, the `getvalue()` copy and the
encoded body all sat in memory at once — Render's event log shows
`server_failed: oomKilled (512Mi)` seconds into that upload, and every request
afterwards got 503/429 until the instance came back. The preview that followed
was worse: a server-side 320p transcode inlined into session_state as base64,
which is re-sent over the websocket on every rerun.

Locked in here:
  1. clips upload in 4MB slices and the engine assembles them on disk
  2. the ext from the client filename cannot escape the media directory
  3. the deployed app publishes previews through Streamlit's static dir
     (HTTP range requests from disk) instead of fetching/inlining them
  4. the app never sends a whole clip as a single multipart body again
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend"))

import pytest
from fastapi.testclient import TestClient

import backend.main as main
import media_store


@pytest.fixture()
def media_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "TEMP_DIR", str(tmp_path))
    monkeypatch.setattr(main, "MEDIA_FILES", {})
    return tmp_path


def test_chunked_upload_assembles_and_registers(media_dir):
    client = TestClient(main.app)
    a = client.post("/media/abc/chunk",
                    data={"offset": 0, "total": 8, "ext": ".ogg"},
                    files={"file": ("slice", b"AAAA", "application/octet-stream")})
    assert a.status_code == 200 and a.json()["received"] == 4
    b = client.post("/media/abc/chunk",
                    data={"offset": 4, "total": 8, "ext": ".ogg"},
                    files={"file": ("slice", b"BBBB", "application/octet-stream")})
    assert b.status_code == 200 and b.json()["received"] == 8

    stored = media_dir / "media_abc.ogg"
    assert stored.read_bytes() == b"AAAABBBB"
    assert main.MEDIA_FILES["abc"] == str(stored)

    got = client.get("/media/abc")
    assert got.status_code == 200
    assert got.content == b"AAAABBBB"
    assert got.headers["content-type"].startswith("audio/ogg")


def test_chunk_ext_cannot_escape_media_dir(media_dir):
    client = TestClient(main.app)
    r = client.post("/media/xyz/chunk",
                    data={"offset": 0, "total": 2, "ext": "/../../evil"},
                    files={"file": ("slice", b"hi", "application/octet-stream")})
    assert r.status_code == 200
    assert (media_dir / "media_xyz.mp4").read_bytes() == b"hi"


def test_whole_file_upload_still_works(media_dir):
    client = TestClient(main.app)
    r = client.post("/media/whole",
                    files={"file": ("clip.mp3", b"ID3DATA", "audio/mpeg")})
    assert r.status_code == 200
    assert (media_dir / "media_whole.mp3").read_bytes() == b"ID3DATA"


def test_post_media_slices_bytes(monkeypatch):
    calls = []

    class Resp:
        status_code = 200

    def fake_post(url, data=None, files=None, timeout=None):
        calls.append((url, dict(data), len(files["file"][1])))
        return Resp()

    monkeypatch.setattr(media_store.requests, "post", fake_post)
    payload = b"x" * (9 * 1024 * 1024 + 7)
    r = media_store.post_media("http://api", "mid", "recitation.mp4", payload, "video/mp4")
    assert r.status_code == 200
    assert [c[0] for c in calls] == ["http://api/media/mid/chunk"] * 3
    assert [c[1]["offset"] for c in calls] == [0, media_store.CHUNK, 2 * media_store.CHUNK]
    assert [c[2] for c in calls] == [media_store.CHUNK, media_store.CHUNK,
                                     len(payload) - 2 * media_store.CHUNK]
    assert all(c[1]["total"] == len(payload) for c in calls)
    assert all(c[1]["ext"] == ".mp4" for c in calls)


def test_post_media_reads_a_file_object_in_slices(monkeypatch):
    calls = []

    class Resp:
        status_code = 200

    monkeypatch.setattr(
        media_store.requests, "post",
        lambda url, data=None, files=None, timeout=None: (
            calls.append((data, len(files["file"][1]))) or Resp()))

    class Uploaded(io.BytesIO):
        size = 5 * 1024 * 1024
        name = "clip.mov"

    media_store.post_media("http://api", "mid", "clip.mov", Uploaded(b"y" * Uploaded.size))
    assert [c[1] for c in calls] == [media_store.CHUNK, media_store.CHUNK - 3 * 1024 * 1024]
    assert [c[0]["offset"] for c in calls] == [0, media_store.CHUNK]


def test_post_media_stops_on_error(monkeypatch):
    calls = []

    class Resp:
        def __init__(self, code):
            self.status_code = code

    def fake_post(url, data=None, files=None, timeout=None):
        calls.append(data["offset"])
        return Resp(413 if len(calls) == 2 else 200)

    monkeypatch.setattr(media_store.requests, "post", fake_post)
    r = media_store.post_media("http://api", "mid", "a.mp4", b"z" * (10 * 1024 * 1024))
    assert r.status_code == 413
    assert calls == [0, media_store.CHUNK]  # never sent the third slice


def test_publish_media_copies_and_serves_from_static(tmp_path):
    temp, static = tmp_path / "temp", tmp_path / "static"
    temp.mkdir()
    src = temp / "media_abc123.mp4"
    src.write_bytes(b"video-bytes")

    url = media_store.publish_media(str(temp), str(static), "abc123")
    assert url.startswith("/app/static/media_abc123.mp4?v=")
    assert (static / "media_abc123.mp4").read_bytes() == b"video-bytes"


def test_publish_media_missing_copy_returns_none(tmp_path):
    assert media_store.publish_media(str(tmp_path), str(tmp_path / "s"), "nope") is None


def test_prune_static_drops_only_this_sessions_media(tmp_path):
    static = tmp_path / "static"
    static.mkdir()
    (static / "media_mine.mp4").write_bytes(b"x")
    (static / "media_theirs.mp4").write_bytes(b"y")
    (static / ".gitkeep").write_bytes(b"")
    media_store.prune_static(str(static), ["mine", None])
    assert not (static / "media_mine.mp4").exists()
    assert (static / "media_theirs.mp4").exists()
    assert (static / ".gitkeep").exists()


def test_app_ships_the_chunked_and_static_paths():
    root = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(root, "frontend", "app.py"), encoding="utf-8-sig") as f:
        src = f.read()
    assert "import media_store" in src
    # The 40MB single-body upload is what got the instance OOM-killed.
    assert "_vfile.getvalue()" not in src
    assert '/preview", timeout=1200' not in src

    with open(os.path.join(root, ".streamlit", "config.toml"), encoding="utf-8") as f:
        cfg = f.read()
    assert "enableStaticServing = true" in cfg
    assert os.path.isdir(os.path.join(root, "frontend", "static"))
