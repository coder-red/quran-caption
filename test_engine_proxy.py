"""Reverse-proxy route table: only /engine/media/* reaches the engine.

A wrong pattern here does not fail loudly — the request silently falls
through to Streamlit (which is how the first live check returned the app
page for /engine/media/ping). These tests pin the table.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from deploy.render import proxy  # noqa: E402


class TestEngineUpstream:
    def test_ping_routes_to_engine(self):
        assert proxy.engine_upstream("GET", "/engine/media/ping") == "/media/ping"

    def test_upload_routes_to_engine(self):
        assert (proxy.engine_upstream("POST", "/engine/media/abc123/upload")
                == "/media/abc123/upload")

    def test_chunk_routes_to_engine(self):
        assert (proxy.engine_upstream("POST", "/engine/media/abc123/chunk")
                == "/media/abc123/chunk")

    def test_status_routes_to_engine(self):
        assert (proxy.engine_upstream("GET", "/engine/media/abc123/status")
                == "/media/abc123/status")

    def test_query_is_not_part_of_the_route(self):
        # engine_upstream must see only the path; query handling is in handle_http
        assert proxy.engine_upstream("POST", "/engine/media/a-b_c1/upload") == "/media/a-b_c1/upload"

    def test_everything_else_goes_to_streamlit(self):
        assert proxy.engine_upstream("GET", "/") is None
        assert proxy.engine_upstream("GET", "/_stcore/health") is None
        assert proxy.engine_upstream("GET", "/_stcore/stream") is None
        assert proxy.engine_upstream("GET", "/app/static/media_x.mp4") is None
        # align and render must never be public through the proxy
        assert proxy.engine_upstream("POST", "/engine/align") is None
        assert proxy.engine_upstream("POST", "/engine/render") is None
        assert proxy.engine_upstream("GET", "/engine/align/result/abc") is None
        assert proxy.engine_upstream("GET", "/engine/media/abc") is None
        assert proxy.engine_upstream("DELETE", "/engine/media/abc123/upload") is None
        # id alphabet: no slashes, no traversal, bounded length
        assert proxy.engine_upstream("POST", "/engine/media/../upload") is None
        assert proxy.engine_upstream("POST", "/engine/media/" + "a" * 65 + "/upload") is None
        assert proxy.engine_upstream("POST", "/engine/media/a b/upload") is None

    def test_wrong_method_does_not_match(self):
        assert proxy.engine_upstream("POST", "/engine/media/ping") is None
        assert proxy.engine_upstream("GET", "/engine/media/abc123/upload") is None
