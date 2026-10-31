"""What proxy formats can THIS host's ffmpeg actually decode?

The browser can build a tiny Ogg/Opus audio proxy (see
frontend/opus_proxy.js) that makes the upload ~20x smaller. But an ffmpeg
build is not guaranteed to have an Opus decoder: the slim public deploy
falls back to imageio-ffmpeg's binary, and hosts may ship their own
system ffmpeg. A build without Opus decodes the .ogg as an error, and the
align dies with "ffmpeg extract_audio failed".

So the host PROBES itself once with a real Ogg/Opus sample
(backend/ogg_probe.ogg, produced by the very same muxer the browser uses)
and reports the result. The frontend asks the engine which formats are
safe and picks accordingly — Ogg/Opus when it works, plain WAV (PCM, which
every ffmpeg build supports) when it does not.

Nothing here guesses: the answer comes from running the real decode.
"""
from __future__ import annotations

import os
import subprocess
import threading

from backend.config import FFMPEG_PATH

_PROBE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "ogg_probe.ogg")

_lock = threading.Lock()
_result: bool | None = None


def supports_ogg_opus(force: bool = False) -> bool:
    """True when this host's ffmpeg can decode an Ogg/Opus file.

    Cached after the first probe (one ffmpeg run, ~50 ms).
    """
    global _result
    with _lock:
        if _result is not None and not force:
            return _result
        ok = False
        if os.path.isfile(_PROBE):
            try:
                r = subprocess.run(
                    [FFMPEG_PATH, "-v", "error", "-i", _PROBE,
                     "-f", "null", "-"],
                    capture_output=True, timeout=60,
                )
                ok = r.returncode == 0
            except Exception:
                ok = False
        _result = ok
        return ok


def proxy_formats() -> list[str]:
    """Formats the browser may use for a fast-upload proxy, best first.

    'ogg_opus' is ~25x smaller than the source; 'wav' is PCM (universally
    decodable) at ~1.3x smaller than a typical video but far smaller than
    the source for a long clip.
    """
    fmts = []
    if supports_ogg_opus():
        fmts.append("ogg_opus")
    fmts.append("wav")
    return fmts
