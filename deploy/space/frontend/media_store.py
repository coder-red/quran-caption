"""Media transfer between the Streamlit app and the caption engine.

On the deployed host the browser can only reach Streamlit, so every byte of a
recitation crosses this process. Two rules keep a 512MB instance alive:

* uploads go up in 4MB slices — one 40MB multipart body (plus the copies of it
  Streamlit and requests keep in memory) OOM-killed the free instance
  mid-upload, which looked like "the upload never finishes";
* previews are published into Streamlit's static directory, which serves them
  from disk with HTTP range support, instead of being transcoded and inlined
  as a base64 data URI in session_state.
"""
import io
import os
import shutil

import requests

CHUNK = 4 * 1024 * 1024


def post_media(api_url: str, media_id: str, name: str, source, ctype: str = ""):
    """Upload `source` (bytes or a readable file object) in CHUNK slices.

    Returns the response of the last slice (or of the failing one)."""
    ext = os.path.splitext(name or "")[1] or ".mp4"
    if isinstance(source, (bytes, bytearray)):
        reader, size = io.BytesIO(source), len(source)
    else:
        try:
            source.seek(0)
        except Exception:
            pass
        reader, size = source, getattr(source, "size", None)
    offset, resp = 0, None
    while True:
        buf = reader.read(CHUNK)
        if not buf:
            break
        resp = requests.post(
            f"{api_url}/media/{media_id}/chunk",
            data={"offset": offset, "total": size or 0, "ext": ext},
            files={"file": ("slice", buf, ctype or "application/octet-stream")},
            timeout=300,
        )
        if resp.status_code != 200:
            return resp
        offset += len(buf)
    if resp is None:
        # An empty file still has to create the stored copy.
        resp = requests.post(
            f"{api_url}/media/{media_id}/chunk",
            data={"offset": 0, "total": 0, "ext": ext},
            files={"file": ("slice", b"", ctype or "application/octet-stream")},
            timeout=60,
        )
    return resp


def _stored(temp_dir: str, media_id: str):
    try:
        names = sorted(n for n in os.listdir(temp_dir)
                       if n.startswith(f"media_{media_id}."))
    except OSError:
        return None
    return os.path.join(temp_dir, names[0]) if names else None


def publish_media(temp_dir: str, static_dir: str, media_id: str):
    """Expose the engine's copy of a clip through Streamlit's static server.

    Returns a URL for the <video> src, or None when the engine's copy is gone
    (e.g. it restarted) or the copy failed."""
    src = _stored(temp_dir, media_id)
    if not src:
        return None
    name = os.path.basename(src)
    dst = os.path.join(static_dir, name)
    try:
        os.makedirs(static_dir, exist_ok=True)
        if (not os.path.exists(dst)
                or os.path.getmtime(dst) < os.path.getmtime(src)
                or os.path.getsize(dst) != os.path.getsize(src)):
            shutil.copyfile(src, dst)
        # Mtime as a query keeps a re-upload of the same id from being served
        # from the browser's cache.
        return f"/app/static/{name}?v={int(os.path.getmtime(src))}"
    except OSError:
        return None


def prune_static(static_dir: str, media_ids):
    """Drop a session's published copies once it picks a new clip.

    Only the given ids are touched: other sessions on the same instance keep
    their own files."""
    ids = [i for i in media_ids if i]
    if not ids:
        return
    try:
        names = os.listdir(static_dir)
    except OSError:
        return
    for name in names:
        if any(name.startswith(f"media_{i}.") for i in ids):
            try:
                os.remove(os.path.join(static_dir, name))
            except OSError:
                pass
