"""Repository interpreter hook (imported at Python startup when the repo
root is on sys.path — the Render service sets PYTHONPATH=. for this).

Why: the free Render service runs `streamlit run frontend/app.py` with a
dashboard-owned start command, and only Streamlit's own port is reachable
from the browser — so every uploaded byte used to cross Streamlit's file
uploader (whole body in RAM, starlette upload.read()) and OOM-killed the
512MB instance mid-upload.

Streamlit 1.58 builds its starlette app in
streamlit.web.server.starlette.starlette_app.create_starlette_app();
this hook wraps that factory and APPENDS the three /engine/media routes
(backend/embedded_engine.py — a thin in-process reverse proxy to the
caption engine on 127.0.0.1:8000) to the route table. Pages, /app/static,
/_stcore/* and the websocket are untouched; everything /engine/* does is
bounded exactly like the standalone proxy (deploy/render/proxy.py).

Anything unexpected (Streamlit internals change, import failure, engine
down) is swallowed silently: the app then behaves exactly as before, and
the browser's JSON preflight falls back to Streamlit's own uploader.
"""
import os


def _qcap_patch_streamlit():
    try:
        if os.environ.get("QCAP_NO_EMBEDDED_ENGINE"):
            return
        import streamlit.web.server.starlette.starlette_app as sa

        flag = "__qcap_engine_patched__"
        if getattr(sa, flag, False):
            return
        original = sa.create_starlette_app

        def patched(runtime):
            app = original(runtime)
            try:
                from backend.embedded_engine import extra_routes
                # Streamlit matches routes in order and its table ends with a
                # catch-all static handler — appended routes would never be
                # reached, so ours go first.
                app.routes[:0] = extra_routes()
            except Exception:
                pass  # never break the app; the JS preflight falls back
            return app

        sa.create_starlette_app = patched
        # Belt and braces for any module that already from-imported it.
        import sys
        server = sys.modules.get(
            "streamlit.web.server.starlette.starlette_server")
        if server is not None:
            try:
                server.create_starlette_app = patched
            except Exception:
                pass
        setattr(sa, flag, True)
    except Exception:
        pass


try:
    _qcap_patch_streamlit()
except Exception:
    pass
