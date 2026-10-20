"""Base contract for hosted ASR providers.

`transcribe(audio_path, slices)` returns:
  {
    "transcript": str,      # space-joined <words[].text> (openers kept; detector strips them)
    "words": [{"text", "start", "end"}],   # absolute seconds, sanitized
    "bounds": [(s, e), ...]  # audio segments sent (debug)
  }
Word timestamps are the source of truth for timing (no forced alignment).
"""
import re
import time
from typing import Any

_ARABIC_RE = re.compile(r"[^\u0621-\u064a\u0671\u0620]")

# Network-level failures are common on a free tier with a shared, flaky
# uplink: a single dropped connection used to abort a whole multi-minute
# align ("groq transcription failed: Connection error") even though the
# very next request would have succeeded. These helpers classify a failure
# as transient (worth retrying) or permanent (worth failing fast on).
_TRANSIENT_WORDS = (
    "connection", "timeout", "timed out", "temporarily", "unavailable",
    "reset by peer", "broken pipe", "eof occurred", "server error",
    "bad gateway", "gateway timeout", "service unavailable",
    "remote end closed", "incomplete", "rate limit", "too many requests",
)


def is_transient_error(exc: BaseException) -> bool:
    """True when the failure looks like a blip worth retrying.

    Retried: connection resets, timeouts, DNS hiccups, 5xx, 429.
    NOT retried: other 4xx (bad request, bad key) — retrying those just
    burns time and quota.
    """
    name = type(exc).__name__.lower()
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int):
        if status == 429:
            return True
        if status >= 500:
            return True
        if 400 <= status < 500:
            return False
    if any(k in name for k in ("connection", "timeout", "serviceunavailable",
                               "internalserver")):
        return True
    msg = str(exc).lower()
    return any(k in msg for k in _TRANSIENT_WORDS)


def retry_with_backoff(fn, attempts: int = 3, base_delay: float = 3.0):
    """Call `fn()`, retrying transient failures with exponential backoff.

    `fn` receives the 0-based attempt number. Raises the last error once
    the attempts are exhausted.
    """
    last = None
    for attempt in range(attempts + 1):
        try:
            return fn(attempt)
        except Exception as exc:
            last = exc
            if not is_transient_error(exc) or attempt >= attempts:
                raise
            time.sleep(base_delay * (2 ** attempt))
    raise last  # pragma: no cover


def clean_arabic_word(text: str) -> str:
    """Keep only Arabic letter characters (drops punctuation, Latin, etc.)."""
    return _ARABIC_RE.sub("", text or "")


class ASRProvider:
    name = "base"

    def transcribe(self, audio_path: str, slices=None, **kwargs) -> dict[str, Any]:
        raise NotImplementedError