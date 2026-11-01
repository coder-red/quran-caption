"""Render must fit the free host's 512MB — and never buffer the MP4 in RAM.

Render OOM-killed the live instance twice (2026-11-01, 13:18Z and 13:27Z):
a 1080p x264 encode peaks at 327MB RSS (measured on the real 8-min Al-Hashr
clip) before Streamlit and the API are counted, and the finished MP4 was then
read into the Streamlit process (`r.content` + st.download_button copies).
720p ultrafast peaks at 88MB, and the file now goes straight to the static
dir where the browser streams it from disk.
"""
import ast
import os
import types

import pytest

from backend import renderer
from frontend import media_store

ROOT = os.path.dirname(os.path.abspath(__file__))


def _app_source():
    with open(os.path.join(ROOT, "frontend", "app.py"), encoding="utf-8-sig") as f:
        return f.read()


# ---------------------------------------------------------------- resolution
def test_1080p_is_clamped_to_720p_on_a_fractional_core_host(monkeypatch):
    monkeypatch.setattr(renderer, "_cpu_quota", lambda: 0.1)
    assert renderer._clamp_resolution(1920, 1080, None) == (1280, 720)


def test_explicit_720p_and_480p_are_kept(monkeypatch):
    monkeypatch.setattr(renderer, "_cpu_quota", lambda: 0.1)
    assert renderer._clamp_resolution(1280, 720, None) == (1280, 720)
    assert renderer._clamp_resolution(854, 480, None) == (854, 480)


def test_original_is_clamped_when_the_source_is_1080p(monkeypatch):
    """'Original' means the source's own size — often 1080p once decoded."""
    monkeypatch.setattr(renderer, "_cpu_quota", lambda: 0.1)
    monkeypatch.setattr(renderer, "probe_video_size", lambda p: (1920, 1080))
    assert renderer._clamp_resolution(None, None, "clip.mp4") == (1280, 720)


def test_original_720p_source_is_left_alone(monkeypatch):
    monkeypatch.setattr(renderer, "_cpu_quota", lambda: 0.1)
    monkeypatch.setattr(renderer, "probe_video_size", lambda p: (1280, 720))
    assert renderer._clamp_resolution(None, None, "clip.mp4") == (None, None)


def test_a_full_core_host_is_untouched(monkeypatch):
    monkeypatch.setattr(renderer, "_cpu_quota", lambda: 4.0)
    monkeypatch.setattr(renderer, "probe_video_size", lambda p: (1920, 1080))
    assert renderer._clamp_resolution(1920, 1080, "clip.mp4") == (1920, 1080)
    assert renderer._clamp_resolution(None, None, "clip.mp4") == (None, None)


def test_probe_video_size_reads_ffprobe_output(monkeypatch):
    monkeypatch.setattr(renderer, "_ffprobe_path", lambda: "ffprobe")
    monkeypatch.setattr(
        renderer.subprocess, "run",
        lambda *a, **k: types.SimpleNamespace(stdout="1920x1080\n"),
    )
    assert renderer.probe_video_size("clip.mp4") == (1920, 1080)


def test_probe_video_size_survives_garbage(monkeypatch):
    monkeypatch.setattr(renderer, "_ffprobe_path", lambda: "ffprobe")
    monkeypatch.setattr(
        renderer.subprocess, "run",
        lambda *a, **k: types.SimpleNamespace(stdout="N/A\n"),
    )
    assert renderer.probe_video_size("clip.mp4") == (0, 0)


# ---------------------------------------------------------- publish / prune
class _FakeResponse:
    status_code = 200

    def __init__(self, chunks):
        self._chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_content(self, size):
        yield from self._chunks


def test_publish_render_streams_into_static_and_returns_a_url(tmp_path, monkeypatch):
    static = tmp_path / "static"
    calls = {}

    def fake_get(url, **kw):
        calls["url"], calls["kw"] = url, kw
        return _FakeResponse([b"one", b"", b"two"])

    monkeypatch.setattr(media_store.requests, "get", fake_get)
    url = media_store.publish_render("http://api", str(static), "job1")
    assert url.startswith("/app/static/render_job1.mp4?v=")
    assert (static / "render_job1.mp4").read_bytes() == b"onetwo"
    assert calls["kw"].get("stream") is True


def test_publish_render_failed_fetch_returns_none(tmp_path, monkeypatch):
    resp = _FakeResponse([b"x"])
    resp.status_code = 404
    monkeypatch.setattr(media_store.requests, "get", lambda *a, **k: resp)
    assert media_store.publish_render("http://api", str(tmp_path), "job2") is None


def test_publish_render_connection_error_returns_none(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise media_store.requests.RequestException("down")

    monkeypatch.setattr(media_store.requests, "get", boom)
    assert media_store.publish_render("http://api", str(tmp_path), "job3") is None


def test_prune_renders_drops_only_this_sessions_files(tmp_path):
    static = tmp_path / "static"
    static.mkdir()
    (static / "render_mine.mp4").write_bytes(b"x")
    (static / "render_theirs.mp4").write_bytes(b"y")
    (static / ".gitkeep").write_bytes(b"")
    media_store.prune_renders(str(static), ["mine", None])
    assert not (static / "render_mine.mp4").exists()
    assert (static / "render_theirs.mp4").exists()
    assert (static / ".gitkeep").exists()


# ------------------------------------------------------------------- guards
def test_render_bytes_never_flow_through_the_app_process():
    src = _app_source()
    # The finished MP4 lives in the static dir and is streamed from disk.
    assert "render_bytes" not in src
    assert "publish_render(" in src
    assert 'st.video(ss["render_url"])' in src
    # Only SRT/VTT go through download_button (small, encoded in the script).
    assert src.count("st.download_button") == 2
    assert "Download MP4" in src


def test_publish_render_streams_instead_of_buffering():
    path = os.path.join(ROOT, "frontend", "media_store.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    assert "stream=True" in src
    assert "r.content" not in src


def _lift_app_function(name):
    path = os.path.join(ROOT, "frontend", "app.py")
    tree = ast.parse(_app_source())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            mod = ast.Module(body=[node], type_ignores=[])
            ns = {"os": os}
            exec(compile(mod, path, "exec"), ns)
            return ns[name]
    raise AssertionError(f"{name} not found in frontend/app.py")


class _FakeFile:
    def __init__(self, text):
        self._text = text

    def read(self):
        return self._text

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_the_ui_reads_the_container_quota():
    """On the 0.1-CPU host the export card must default to 720p, not 1080p."""
    fn = _lift_app_function("_host_cpu_quota")
    ns_open = getattr(fn, "__globals__", {})
    ns_open["open"] = lambda path, *a, **k: (
        _FakeFile("20000 100000\n")
        if str(path) == "/sys/fs/cgroup/cpu.max"
        else (_ for _ in ()).throw(OSError(path))
    )
    assert fn() == pytest.approx(0.2)
