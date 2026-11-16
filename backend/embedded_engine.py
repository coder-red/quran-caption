"""In-process /engine routes for the dashboard-pinned start command.

The free Render service runs `streamlit run frontend/app.py` with a command
the workspace owner cannot change through the API, and only Streamlit's own
port is reachable from the browser — so every uploaded byte used to cross
Streamlit's file uploader (whole body in RAM) and OOM-kill the 512MB host.

`sitecustomize.py` mounts these routes INSIDE Streamlit's starlette app
(additive only; everything else is untouched). They forward the same paths
as the standalone proxy (deploy/render/proxy.py) to the caption engine on
127.0.0.1:8000, which does the disk writes:

  GET  /engine/media/ping            -> engine reachability probe
  POST /engine/media/<id>/upload    -> raw slice (query: offset, total, ext)
  GET  /engine/media/<id>/status    -> size/complete for the render unlock

Everything else (pages, /app/static, /_stcore/*, the websocket) stays with
Streamlit. If this module or Streamlit's internals change, the mount fails
silently and the browser's JSON preflight falls back to the legacy path —
the app keeps working exactly as before, just slower.
"""
import os
import re

import requests
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

ENGINE_URL = os.environ.get("QCAP_ENGINE_URL", "http://127.0.0.1:8000")
_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


async def _ping(request: Request) -> Response:
    try:
        r = requests.get(f"{ENGINE_URL}/media/ping", timeout=3)
        return Response(content=r.content, status_code=r.status_code,
                        media_type=r.headers.get("content-type", "application/json"))
    except Exception:
        return JSONResponse({"ok": False}, status_code=502)


async def _upload(request: Request) -> Response:
    media_id = request.path_params.get("media_id", "")
    if not _ID.fullmatch(media_id):
        return JSONResponse({"detail": "bad media id"}, status_code=400)
    offset = request.query_params.get("offset", "0")
    total = request.query_params.get("total", "0")
    ext = request.query_params.get("ext", ".mp4")
    # The browser slices the file itself (4MB), so the whole body here is
    # bounded by one slice; reading it once is fine.
    body = await request.body()
    try:
        r = requests.post(
            f"{ENGINE_URL}/media/{media_id}/upload"
            f"?offset={int(offset)}&total={int(total)}&ext={ext}",
            data=body,
            headers={"Content-Type": "application/octet-stream"},
            timeout=(10, 600),
        )
    except Exception as e:
        return JSONResponse({"detail": f"caption engine unreachable: {e}"},
                            status_code=502)
    return Response(content=r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type", "application/json"))


async def _status(request: Request) -> Response:
    media_id = request.path_params.get("media_id", "")
    if not _ID.fullmatch(media_id):
        return JSONResponse({"detail": "bad media id"}, status_code=400)
    try:
        r = requests.get(f"{ENGINE_URL}/media/{media_id}/status", timeout=5)
    except Exception as e:
        return JSONResponse({"detail": f"caption engine unreachable: {e}"},
                            status_code=502)
    return Response(content=r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type", "application/json"))


def extra_routes():
    return [
        Route("/engine/media/ping", _ping, methods=["GET"]),
        Route("/engine/media/{media_id}/upload", _upload, methods=["POST"]),
        Route("/engine/media/{media_id}/status", _status, methods=["GET"]),
    ]
