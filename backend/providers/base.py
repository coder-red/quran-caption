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
from typing import Any

_ARABIC_RE = re.compile(r"[^\u0621-\u064a\u0671\u0620]")


def clean_arabic_word(text: str) -> str:
    """Keep only Arabic letter characters (drops punctuation, Latin, etc.)."""
    return _ARABIC_RE.sub("", text or "")


class ASRProvider:
    name = "base"

    def transcribe(self, audio_path: str, slices=None, **kwargs) -> dict[str, Any]:
        raise NotImplementedError