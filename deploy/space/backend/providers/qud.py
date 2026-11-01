"""QUD³ Quranic Universal Aligner provider (https://aligner.qud.dev).

Unlike the ASR providers this one does not return a flat transcript: it aligns
the recitation to the mushaf and returns ayah-referenced segments with
per-word times (`ref_from`/`ref_to` as ``surah:ayah:word``). Repeats are
separate segments, and a held madd is inside its word's duration, which is
why the caption path can trust these times as-is (see segmenter
`segments_from_qud_alignment`).

The public API needs no key. GPU is a daily per-caller quota with an automatic
CPU fallback; CPU is fair-use limited (429 with `retry_after_s`). Any failure
raises `QudUnavailable` so the caller can fall back to Groq.
"""
import json
import os
import time
from typing import Callable, Optional

from backend.config import (
    QUD_DEVICE,
    QUD_MODEL,
    QUD_RIWAYAH,
    QUD_TIMEOUT,
    QUD_URL,
)
from backend.providers.base import ASRProvider

try:
    import requests
except Exception:  # pragma: no cover - requests ships with streamlit
    requests = None

MAX_RETRIES = 2
MAX_RETRY_AFTER = 10.0

# Pipeline stage -> progress percent, mapped into the 18-80 band by main.py.
_STAGE_PCT = {
    "queued_gpu": 20.0,
    "queued_cpu": 21.0,
    "segmenting": 30.0,
    "transcribing": 35.0,
    "matching": 68.0,
    "recovering": 72.0,
    "timing_words": 76.0,
    "building": 79.0,
}


class QudUnavailable(RuntimeError):
    """QUD could not produce a usable alignment; the caller should fall back."""

    def __init__(self, message: str, transient: bool = False,
                 retry_after: Optional[float] = None):
        super().__init__(message)
        self.transient = transient
        self.retry_after = retry_after


class QudProvider(ASRProvider):
    name = "qud"

    def __init__(self, url: Optional[str] = None, device: Optional[str] = None,
                 model: Optional[str] = None, riwayah: Optional[str] = None,
                 timeout: Optional[float] = None):
        self.url = (url or QUD_URL).rstrip("/")
        self.device = (device or QUD_DEVICE).strip().upper() or "GPU"
        self.model = model or QUD_MODEL
        self.riwayah = (riwayah or QUD_RIWAYAH).strip().lower() or "hafs"
        self.timeout = float(timeout or QUD_TIMEOUT)

    def transcribe(self, audio_path: str, slices=None, **kwargs):
        raise QudUnavailable("QUD provider only supports align()")

    def align(self, audio_path: str,
              progress_cb: Optional[Callable[[float, str], None]] = None) -> dict:
        """Align one audio file and return the parsed QUD response.

        progress_cb(percent, label) receives the server's pipeline stage.
        Retries transient failures (connection, 5xx, short 429 waits)."""
        if requests is None:
            raise QudUnavailable("requests is not installed (pip install requests)")
        last: Optional[QudUnavailable] = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                return self._stream(audio_path, progress_cb)
            except QudUnavailable as exc:
                last = exc
                if not exc.transient or attempt >= MAX_RETRIES:
                    raise
                delay = exc.retry_after
                if delay is None:
                    delay = 3.0 * (2 ** attempt)
                if delay > MAX_RETRY_AFTER:
                    raise
                if progress_cb:
                    progress_cb(20.0, f"QUD busy — retrying in {delay:.0f}s…")
                time.sleep(delay)
        raise last  # pragma: no cover

    # ------------------------------------------------------------------ HTTP

    def _stream(self, audio_path: str, progress_cb) -> dict:
        data = {
            "device": self.device,
            "model_name": self.model,
            "riwayah": self.riwayah,
        }
        name = os.path.basename(audio_path) or "audio.wav"
        try:
            with open(audio_path, "rb") as fh:
                resp = requests.post(
                    self.url + "/align/audio/stream",
                    files={"audio": (name, fh, "audio/wav")},
                    data=data,
                    stream=True,
                    # The connect value doubles as urllib3's write timeout while
                    # the body is sent; 10 s aborted a 15.6 MB WAV upload on a
                    # slow uplink before the server had read it.
                    timeout=(60, self.timeout),
                )
        except requests.RequestException as exc:
            raise QudUnavailable(f"QUD request failed: {exc}", transient=True) from exc

        if resp.status_code != 200:
            raise self._http_error(resp)

        event = None
        try:
            for raw in resp.iter_lines(decode_unicode=True):
                if raw is None:
                    continue
                line = raw.strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event = (line[6:].strip() or None)
                    continue
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if event == "progress":
                    self._progress(payload, progress_cb)
                    event = None
                    continue
                if event == "error":
                    raise self._error_from(payload)
                try:
                    obj = json.loads(payload)
                except ValueError:
                    event = None
                    continue
                if isinstance(obj, dict) and "segments" in obj:
                    return obj
                if isinstance(obj, dict) and ("code" in obj or "status" in obj):
                    raise self._error_from(payload)
                event = None
        except requests.RequestException as exc:
            raise QudUnavailable(f"QUD stream interrupted: {exc}", transient=True) from exc
        raise QudUnavailable("QUD stream ended without a result", transient=True)

    def _progress(self, payload: str, progress_cb) -> None:
        if not progress_cb:
            return
        try:
            obj = json.loads(payload)
        except ValueError:
            return
        if not isinstance(obj, dict):
            return
        stage = str(obj.get("stage") or "").strip().lower()
        if not stage:
            return
        pct = _STAGE_PCT.get(stage)
        if pct is None:
            return
        if stage == "transcribing":
            try:
                step = float(obj.get("step") or 0)
                steps = float(obj.get("steps") or 0)
            except (TypeError, ValueError):
                step = steps = 0.0
            if steps > 0:
                pct += 30.0 * min(1.0, step / steps)
        label = f"aligning with QUD ({stage.replace('_', ' ')})…"
        progress_cb(pct, label)

    def _http_error(self, resp) -> QudUnavailable:
        status = int(getattr(resp, "status_code", 0) or 0)
        detail = {}
        try:
            body = resp.json()
            if isinstance(body, dict):
                detail = body
        except Exception:
            pass
        if not detail:
            try:
                detail = {"message": (resp.text or "")[:300]}
            except Exception:
                detail = {}
        if not isinstance(detail, dict):
            detail = {"message": str(detail)[:300]}
        code = detail.get("code") or f"http_{status}"
        message = detail.get("message") or code
        inner = detail.get("detail")
        retry_after = None
        if isinstance(inner, dict):
            try:
                retry_after = float(inner.get("retry_after_s"))
            except (TypeError, ValueError):
                retry_after = None
            if inner.get("retry_after_s") is not None and retry_after is None:
                message = f"{message} ({inner})"
        if status == 402:
            return QudUnavailable(f"QUD quota exhausted: {message}", transient=False)
        if status == 429:
            return QudUnavailable(f"QUD is rate limited: {message}",
                                  transient=True, retry_after=retry_after)
        if status >= 500:
            return QudUnavailable(f"QUD is unavailable ({status}): {message}",
                                  transient=True)
        return QudUnavailable(f"QUD rejected the request ({status}): {message}",
                              transient=False)

    def _error_from(self, payload: str) -> QudUnavailable:
        try:
            obj = json.loads(payload)
        except ValueError:
            obj = {"message": payload[:300]}
        if not isinstance(obj, dict):
            obj = {"message": str(obj)[:300]}
        status = int(obj.get("status") or 0
                     or (500 if obj.get("code") else 0))
        code = obj.get("code") or "error"
        message = obj.get("message") or code
        retry_after = None
        inner = obj.get("detail")
        if isinstance(inner, dict):
            try:
                retry_after = float(inner.get("retry_after_s"))
            except (TypeError, ValueError):
                retry_after = None
        transient = status >= 500 or status == 429
        return QudUnavailable(f"QUD failed: {message}", transient=transient,
                              retry_after=retry_after)


def validate_response(result: dict) -> dict:
    """Raise QudUnavailable unless the response carries at least one Quran
    segment. Some failures come back as a 200 with an empty segment list."""
    segments = (result or {}).get("segments") or []
    if not segments:
        raise QudUnavailable("QUD returned no segments")
    if not any((s or {}).get("kind") == "quran" for s in segments):
        raise QudUnavailable("QUD returned no Quran segments")
    return result
