"""Factory for the active ASR provider (env ASR_BACKEND or request override)."""
from typing import Optional

from backend.config import ASR_BACKEND


def get_asr_provider(name: Optional[str] = None):
    name = (name or ASR_BACKEND).strip().lower()
    if name == "groq":
        from backend.providers.groq import GroqProvider
        return GroqProvider()
    if name == "gemini":
        try:
            from backend.providers.gemini import GeminiProvider
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "gemini backend is not installed - pip install google-genai "
                "(or use ASR_BACKEND=groq / local)"
            ) from exc
        return GeminiProvider()
    return None  # "local" -> in-process whisper + forced alignment


__all__ = ["get_asr_provider"]