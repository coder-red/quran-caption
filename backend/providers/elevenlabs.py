"""ElevenLabs Scribe hosted ASR provider: scribe_v2 with word timestamps.

Scribe v2 benchmarks ~51ms mean word-boundary error (vs Whisper's
±200-500ms) with word-level timestamps in 90+ languages including Arabic.
Free plan: 4.5h/month recurring, no card; then ~$0.22/hr audio.

Zero local models/compute: audio is transcribed server-side. Words flow
through the same pipeline as the groq provider (VAD anchoring, snapping,
phrase splitting, clamping), so only the timestamp source changes.
"""
import os
import subprocess
import tempfile
import time
from typing import Optional

import numpy as np
import soundfile as sf

from backend.config import (
    FFMPEG_PATH, TEMP_DIR, ELEVENLABS_API_KEY, ELEVENLABS_MODEL,
)
from backend.audio_util import ensure_16k_mono
from backend.providers.base import ASRProvider, clean_arabic_word, is_transient_error
from backend.providers.groq import (
    _anchor_words_to_vad, _plan_bounds_request, MAX_ANCHOR_SHIFT,
)

# Scribe v2 word boundaries benchmark ~51ms (vs Whisper's ±200-500ms), so
# the linear VAD-anchor correction must stay a light touch: small VAD-pad
# level nudges only, never a large drag of good stamps toward a dubious
# slice bound. Phrase-boundary truth still comes from the segmenter VAD
# clamp downstream.
SCRIBE_ANCHOR_SHIFT = 0.25

try:
    from elevenlabs.client import ElevenLabs
except Exception:  # pragma: no cover
    ElevenLabs = None

SAMPLE_RATE = 16000


class ElevenLabsProvider(ASRProvider):
    name = "elevenlabs"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 language: Optional[str] = None):
        self.api_key = api_key or ELEVENLABS_API_KEY
        self.model = model or ELEVENLABS_MODEL
        self.language = language
        self._client = None

    def _api(self) -> ElevenLabs:
        if self._client is None:
            if ElevenLabs is None:
                raise RuntimeError(
                    "elevenlabs package not installed (pip install elevenlabs)")
            if not self.api_key:
                raise RuntimeError(
                    "ELEVENLABS_API_KEY is not set - create a free key at "
                    "https://elevenlabs.io (4.5h/month included) and set "
                    "ELEVENLABS_API_KEY (or ASR_BACKEND=groq for the free "
                    "hosted path)"
                )
            self._client = ElevenLabs(api_key=self.api_key)
        return self._client

    def transcribe(self, audio_path: str, slices=None, language: str = "ar",
                   progress_cb: Optional[callable] = None,
                   anchor_slices=None,
                   speech_segs=None) -> dict:
        wav_path, _owned = self._ensure_16k_mono(audio_path)
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
            bounds = _plan_bounds_request(slices, dur, speech_segs)
            words = []
            n_bounds = len(bounds)
            for i, (s, e) in enumerate(bounds):
                words.extend(self._call(audio, s, e))
                if progress_cb:
                    progress_cb(round(100.0 * (i + 1) / n_bounds))
            words.sort(key=lambda w: w["start"])
            _anchor_words_to_vad(words, anchor_slices or slices,
                                 max_shift=SCRIBE_ANCHOR_SHIFT)
            transcript = " ".join(w["text"] for w in words).strip()
            return {
                "transcript": transcript,
                "words": words,
                "bounds": bounds,
            }
        finally:
            # Only delete a conversion THIS call created.
            if _owned:
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

    def _ensure_16k_mono(self, audio_path: str) -> tuple[str, bool]:
        """Return (wav_path, owned). `owned` False means the input was already
        16 kHz mono WAV and was reused as-is — no second ffmpeg transcode."""
        return ensure_16k_mono(audio_path)

    def _call(self, audio: np.ndarray, s: float, e: float) -> list[dict]:
        lo = max(0, int(s * SAMPLE_RATE))
        hi = min(len(audio), int(e * SAMPLE_RATE))
        seg = audio[lo:hi]
        if len(seg) < SAMPLE_RATE * 0.2:
            return []

        os.makedirs(TEMP_DIR, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=".wav", dir=TEMP_DIR)
        os.close(fd)
        sf.write(tmp, seg, SAMPLE_RATE)
        res = None
        try:
            for attempt in range(4):
                try:
                    with open(tmp, "rb") as f:
                        kwargs = {
                            "model_id": self.model,
                            "file": f,
                            "timestamps_granularity": "word",
                            "diarize": False,
                        }
                        if self.language:
                            kwargs["language_code"] = self.language
                        res = self._api().speech_to_text.convert(**kwargs)
                    break
                except Exception as exc:
                    # Retry transient network failures (dropped connection,
                    # timeout) with backoff instead of aborting a
                    # multi-minute align over one blip.
                    if is_transient_error(exc) and attempt < 3:
                        time.sleep(3 * (2 ** attempt))
                        continue
                    raise RuntimeError(
                        f"elevenlabs transcription failed: {exc}") from exc
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

        words = []
        for item in getattr(res, "words", []) or []:
            if isinstance(item, dict):
                raw_text = item.get("text", item.get("word", ""))
                start = float(item.get("start", 0.0))
                end = float(item.get("end", start))
            else:
                raw_text = getattr(item, "text", None)
                if raw_text is None:
                    raw_text = getattr(item, "word", "")
                start = float(getattr(item, "start", 0.0))
                end = float(getattr(item, "end", start))
            txt = clean_arabic_word(raw_text)
            if not txt:
                continue
            words.append({"text": txt, "start": round(start + s, 3),
                          "end": round(end + s, 3)})
        return words
