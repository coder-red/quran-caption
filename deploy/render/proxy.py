#!/usr/bin/env python3
"""Public entry point for the free Render host: Streamlit + engine behind one port.

Why this exists: the free instance exposes only one port, and the caption
engine runs on 127.0.0.1:8000 inside the same container. Routing everything
publicly in the obvious way (Streamlit directly on $PORT) leaves the browser
no way to reach the engine — so every byte the user uploads had to cross
Streamlit's own uploader, which buffers the whole body in RAM (starlette
read() + the uploaded-file record). That is what OOM-killed the 512MB
instance mid-upload.

This proxy listens on $PORT and forwards:

  /engine/media/ping            -> caption engine (reachability probe)
  /engine/media/<id>/upload     -> caption engine (browser streams raw slices)
  /engine/media/<id>/chunk      -> caption engine (multipart slices, compat)
  /engine/media/<id>/status     -> caption engine (size/complete polling)
  everything else (pages, /app/static, /_stcore/*, the websocket) -> Streamlit

Both hops stream the body chunk by chunk; nothing is buffered whole in RAM,
so a 100MB recitation crossing this proxy costs ~zero memory either way.
Only the three media routes reach the engine — the align/render API surface
stays unreachable from the public internet.
"""

import asyncio
import os
import re
import subprocess
import sys

from aiohttp import ClientSession, ClientTimeout, WSMsgType, web

PORT = int(os.environ.get("PORT", "8080"))
STREAMLIT_PORT = int(os.environ.get("STREAMLIT_PORT", "8501"))
ENGINE_PORT = int(os.environ.get("ENGINE_PORT", "8000"))
STREAMLIT_URL = f"http://127.0.0.1:{STREAMLIT_PORT}"
ENGINE_URL = f"http://127.0.0.1:{ENGINE_PORT}"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade",
}
_MEDIA_ID = r"[A-Za-z0-9_-]{1,64}"
_ENGINE_ROUTES = [
    ("GET", re.compile(r"/media/ping")),
    ("POST", re.compile(r"/media/(" + _MEDIA_ID + r")/upload")),
    ("POST", re.compile(r"/media/(" + _MEDIA_ID + r")/chunk")),
    ("GET", re.compile(r"/media/(" + _MEDIA_ID + r")/status")),
]


def engine_upstream(method: str, path: str):
    """Upstream /media/... path for an allowlisted /engine/... route, else None."""
    if not path.startswith("/engine/"):
        return None
    rest = path[len("/engine"):]
    for verb, pattern in _ENGINE_ROUTES:
        if verb == method and pattern.fullmatch(rest):
            return rest
    return None


def filtered_headers(headers):
    out = {}
    for k, v in headers.items():
        lk = k.lower()
        if lk in HOP_BY_HOP:
            continue
        out[k] = v
    return out


async def handle_http(request: web.Request) -> web.StreamResponse:
    upstream_path = engine_upstream(request.method, request.path)
    if upstream_path is not None:
        base, upstream = ENGINE_URL, upstream_path
    else:
        base, upstream = STREAMLIT_URL, request.path
    url = base + upstream
    if request.query_string:
        url += "?" + request.query_string

    timeout = ClientTimeout(total=1800, connect=30, sock_read=1800)
    try:
        upstream = await request.app["client"].request(
            request.method, url,
            headers=filtered_headers(request.headers),
            data=request.content if request.can_read_body else None,
            timeout=timeout, allow_redirects=False,
        )
    except Exception as e:
        return web.Response(status=502, text=f"caption service unreachable: {e}")

    resp = web.StreamResponse(
        status=upstream.status, headers=filtered_headers(upstream.headers))
    await resp.prepare(request)
    async for chunk in upstream.content.iter_any():
        await resp.write(chunk)
    await resp.write_eof()
    upstream.release()
    return resp


async def handle_websocket(request: web.Request) -> web.StreamResponse:
    query = request.query_string
    upstream_url = STREAMLIT_URL + request.path + (f"?{query}" if query else "")

    offered = request.headers.get("Sec-WebSocket-Protocol", "")
    protocols = [p.strip() for p in offered.split(",") if p.strip()] or None

    ws_server = web.WebSocketResponse(protocols=protocols, autoping=True)
    await ws_server.prepare(request)
    try:
        ws_upstream = await request.app["client"].ws_connect(
            upstream_url, protocols=protocols, autoping=True, heartbeat=15)
    except Exception as e:
        await ws_server.close(code=1013, message=b"upstream down")
        return ws_server

    async def client_to_upstream():
        async for msg in ws_server:
            if msg.type == WSMsgType.TEXT:
                await ws_upstream.send_str(msg.data)
            elif msg.type == WSMsgType.BINARY:
                await ws_upstream.send_bytes(msg.data)
            elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING):
                break
            elif msg.type == WSMsgType.ERROR:
                break
        await ws_upstream.close()

    async def upstream_to_client():
        async for msg in ws_upstream:
            if msg.type == WSMsgType.TEXT:
                await ws_server.send_str(msg.data)
            elif msg.type == WSMsgType.BINARY:
                await ws_server.send_bytes(msg.data)
            elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING):
                break
            elif msg.type == WSMsgType.ERROR:
                break
        try:
            await ws_server.close()
        except Exception:
            pass

    tasks = [asyncio.ensure_future(client_to_upstream()),
             asyncio.ensure_future(upstream_to_client())]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    await ws_upstream.close()
    return ws_server


async def healthz(request: web.Request):
    return web.json_response({"ok": True})


async def websocket_dispatch(request: web.Request):
    if request.headers.get("Upgrade", "").lower() == "websocket":
        return await handle_websocket(request)
    return await handle_http(request)


async def on_startup(app):
    app["client"] = ClientSession(
        auto_decompress=False,
        connector_owner=True,
        headers={"Connection": "keep-alive"},
    )


async def on_cleanup(app):
    await app["client"].close()


def spawn_streamlit() -> subprocess.Popen | None:
    if os.environ.get("QCAP_NO_STREAMLIT"):
        return None
    cmd = [
        sys.executable, "-m", "streamlit", "run", "frontend/app.py",
        "--server.port", str(STREAMLIT_PORT),
        "--server.address", "127.0.0.1",
        "--server.fileWatcherType", "none",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "true",
    ]
    # Streamlit must not see PORT, or it binds the same port as this proxy.
    return subprocess.Popen(cmd, cwd=REPO_ROOT)


def make_app() -> web.Application:
    app = web.Application(client_max_size=0)
    app.router.add_get("/healthz", healthz)
    app.router.add_route("*", "/{tail:.*}", websocket_dispatch)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def main():
    child = spawn_streamlit()
    try:
        web.run_app(make_app(), host="0.0.0.0", port=PORT,
                    print=None, access_log=None)
    finally:
        if child is not None:
            try:
                child.terminate()
                child.wait(timeout=10)
            except Exception:
                try:
                    child.kill()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
