"""Gemini hosted ASR provider: free tier handles LONG audio without chunking
limits or audio-quota walls.

Free tier (2026): gemini-2.5-flash ~15 rpm / 1,500 req/day / 1M TPM, no credit
card. Audio tokenizes at a fixed 32 tokens/sec, so the total input for a 2h
clip is ~230K tokens — trivial for the free tier. Unlike Groq there is no
per-request size cap once audio is chunked locally.

Strategy (matches the official long-audio recipes: slice, transcribe, offset):
- Convert to 16kHz mono WAV once (like the groq provider).
- Pack VAD slices into chunks of <= MAX_REQUEST_SECONDS (10 min) with a small
  overlap so no word is cut at a boundary.
- Each chunk is reduced to 16k mono FLAC (~8MB/10min, under the 20MB inline
  cap) and sent INLINE with JSON-mode output requesting word-level
  timestamps. Gemini returns estimated word times (not true per-word CTC
  frames), which `_anchor_words_to_vad` corrects against the VAD slice bounds.
- Sequential calls stay well under every free-tier quota; a full 2h05m surah
  is ~13 small requests and finishes in minutes instead of an hour.
"""
import json
import math
import os
import subprocess
import tempfile
import time
from typing import Optional

import numpy as np
import soundfile as sf

from backend.config import FFMPEG_PATH, TEMP_DIR, GEMINI_API_KEY, GEMINI_MODEL
from backend.providers.base import ASRProvider, clean_arabic_word

try:
    from google.genai import types
except Exception:  # pragma: no cover
    types = None

try:
    from google import genai as _genai_mod
except Exception:  # pragma: no cover
    _genai_mod = None

MAX_REQUEST_SECONDS = 600.0
OVERLAP_SECONDS = 8.0
SAMPLE_RATE = 16000
MAX_RETRIES = 5

_RETRY_WAIT = (10.0, 20.0, 40.0, 80.0, 160.0)


class _ParseError(RuntimeError):
    """Gemini answered, but the response wasn't the expected word JSON."""


class GeminiProvider(ASRProvider):
    name = "gemini"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or GEMINI_API_KEY
        self.model = model or GEMINI_MODEL
        self._client = None

    def _api(self):
        if self._client is None:
            if _genai_mod is None:
                raise RuntimeError("google-genai not installed (pip install google-genai)")
            if not self.api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set - get a free key (no credit card) at "
                    "https://aistudio.google.com/app/apikey and set GEMINI_API_KEY "
                    "(or ASR_BACKEND=groq for the free hosted path)"
                )
            self._client = _genai_mod.Client(api_key=self.api_key)
        return self._client

    def transcribe(self, audio_path: str, slices=None, language: str = "ar") -> dict:
        wav_path = self._ensure_16k_mono(audio_path)
        try:
            audio, sr = sf.read(wav_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                audio = np.interp(
                    np.arange(len(audio) * SAMPLE_RATE / sr) * sr / SAMPLE_RATE,
                    np.arange(len(audio)), audio,
                ).astype(np.float32)
            dur = len(audio) / SAMPLE_RATE
            if dur < 0.5:
                raise RuntimeError("audio too short to transcribe")
            bounds = self._plan_chunks(slices, dur)
            words = []
            for i, (s, e) in enumerate(bounds):
                chunk_words = self._call_or_shrink(audio, s, e)
                for w in chunk_words:
                    w["_chunk"] = i
                words.extend(chunk_words)
                time.sleep(1.0)  # gentle pacing; free tier is ~15 rpm
            words = self._dedupe_overlaps(words, bounds)
            words.sort(key=lambda w: w["start"])
            _anchor_words_to_vad(words, slices)
            transcript = " ".join(w["text"] for w in words).strip()
            return {
                "transcript": transcript,
                "words": words,
                "bounds": bounds,
            }
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

    def _ensure_16k_mono(self, audio_path: str) -> str:
        os.makedirs(TEMP_DIR, exist_ok=True)
        fd, out = tempfile.mkstemp(suffix=".wav", dir=TEMP_DIR)
        os.close(fd)
        r = subprocess.run(
            [
                FFMPEG_PATH, "-y", "-i", audio_path,
                "-ac", "1", "-ar", str(SAMPLE_RATE),
                "-acodec", "pcm_s16le", out,
            ],
            capture_output=True,
        )
        if r.returncode != 0 or not os.path.exists(out):
            msg = r.stderr.decode("utf-8", errors="replace")[-500:]
            raise RuntimeError(f"gemini: ffmpeg conversion failed: {msg}")
        return out

    @staticmethod
    def _plan_chunks(slices, dur: float) -> list[tuple[float, float]]:
        """Pack VAD slices into <= MAX_REQUEST_SECONDS chunks with OVERLAP on
        each side so boundary words are never cut. Falls back to uniform
        splits when no slice info is available."""
        def _expand(s, e):
            return (max(0.0, s - OVERLAP_SECONDS), min(dur, e + OVERLAP_SECONDS))

        if slices:
            plan = []
            for s, e in slices:
                s, e = max(0.0, float(s)), min(dur, float(e))
                if e <= s:
                    continue
                cs = s
                while e - cs > MAX_REQUEST_SECONDS:
                    plan.append(_expand(cs, cs + MAX_REQUEST_SECONDS))
                    cs += MAX_REQUEST_SECONDS
                if e > cs:
                    plan.append(_expand(cs, e))
            if plan:
                return plan
        n = max(1, math.ceil(dur / MAX_REQUEST_SECONDS))
        step = dur / n
        return [_expand(i * step, min(dur, (i + 1) * step)) for i in range(n)]

    @staticmethod
    def _dedupe_overlaps(words: list[dict], bounds: list[tuple[float, float]]) -> list[dict]:
        """Later chunks re-transcribe audio their predecessor already covered
        (the OVERLAP expansion). Drop any later-chunk word that starts before
        the previous chunk's real end; keep the earlier chunk's copy."""
        if len(bounds) < 2 or not words:
            return words
        prev_end = bounds[0][1]
        drops = set()
        for i in range(1, len(bounds)):
            for k, w in enumerate(words):
                if w.get("_chunk") == i and w["start"] < prev_end:
                    drops.add(k)
            prev_end = bounds[i][1]
        out = []
        for k, w in enumerate(words):
            if k in drops:
                continue
            w.pop("_chunk", None)
            out.append(w)
        return out

    def _call(self, audio: np.ndarray, s: float, e: float) -> list[dict]:
        lo = max(0, int(s * SAMPLE_RATE))
        hi = min(len(audio), int(e * SAMPLE_RATE))
        seg = audio[lo:hi]
        if len(seg) < SAMPLE_RATE * 0.2:
            return []

        os.makedirs(TEMP_DIR, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=".flac", dir=TEMP_DIR)
        os.close(fd)
        try:
            sf.write(tmp, seg, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
            with open(tmp, "rb") as f:
                flac_bytes = f.read()
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

        res = self._generate(flac_bytes)
        return self._parse_words(res, s)

    def _generate(self, flac_bytes: bytes):
        last_err = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                part = types.Part.from_bytes(data=flac_bytes, mime_type="audio/flac")
                return self._api().models.generate_content(
                    model=self.model,
                    contents=[part],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "words": types.Schema(
                                    type=types.Type.ARRAY,
                                    items=types.Schema(
                                        type=types.Type.OBJECT,
                                        properties={
                                            "word": types.Schema(type=types.Type.STRING),
                                            "start": types.Schema(type=types.Type.NUMBER),
                                            "end": types.Schema(type=types.Type.NUMBER),
                                        },
                                        required=["word", "start", "end"],
                                    ),
                                ),
                            },
                            required=["words"],
                        ),
                        temperature=0.0,
                    ),
                )
            except Exception as exc:  # 429 / 500 / 503 overload / transient
                last_err = exc
                msg = str(exc).lower()
                overload = "503" in msg or "unavailable" in msg or "high demand" in msg
                if attempt < MAX_RETRIES:
                    wait = _RETRY_WAIT[attempt]
                    if overload:
                        wait = max(wait, 30.0 * (2 ** attempt))
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"gemini transcription failed: {exc}") from exc
        raise RuntimeError(f"gemini transcription failed: {last_err}")

    def _parse_words(self, res, offset: float) -> list[dict]:
        text = getattr(res, "text", "") or ""
        try:
            payload = json.loads(text)
            items = payload.get("words", [])
        except Exception:
            raise _ParseError(f"gemini: could not parse word JSON: {text[:200]!r}")
        out = []
        for it in items:
            if isinstance(it, dict):
                raw = it.get("word", "")
                start = float(it.get("start", 0.0))
                end = float(it.get("end", start))
            else:
                raw = getattr(it, "word", "")
                start = float(getattr(it, "start", 0.0))
                end = float(getattr(it, "end", start))
            txt = clean_arabic_word(raw)
            if not txt:
                continue
            out.append({"text": txt, "start": round(start + offset, 3), "end": round(end + offset, 3)})
        return out

    def _call_or_shrink(self, audio: np.ndarray, s: float, e: float,
                        depth: int = 0) -> list[dict]:
        """Call _call for [s,e); if Gemini answers but the words are
        unparseable (likely an output-cap issue), bisect and retry each
        half. Transport errors (429/503 overload) are NOT bisected — they
        are model-side and retried with backoff inside _generate."""
        try:
            return self._call(audio, s, e)
        except _ParseError:
            if depth >= 3 or e - s < 30.0:
                raise
            mid = (s + e) / 2
            out = self._call_or_shrink(audio, s, mid, depth + 1)
            out.extend(self._call_or_shrink(audio, mid, e, depth + 1))
            return out


def _anchor_words_to_vad(words: list[dict], slices: list[tuple[float, float]]):
    """Same drift correction as the groq provider: linearly snap the first word
    of each VAD slice to the slice onset and the last word to its end."""
    from backend.providers.groq import _anchor_words_to_vad as _anchor
    _anchor(words, slices)