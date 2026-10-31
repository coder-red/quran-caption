"""Groq hosted ASR provider: whisper-large-v3-turbo with word timestamps.

Free tier: ~20 req/min, 2,000 clips/day. One API call per <= MAX_REQUEST_SECONDS
of audio (slices are packed on VAD-silence boundaries), so a typical short
recitation is a single request. Real word timestamps replace the local
wav2vec2 forced-alignment stage.
"""
import math
import os
import subprocess
import tempfile
import time
from typing import Optional

import numpy as np
import soundfile as sf

from backend.config import FFMPEG_PATH, TEMP_DIR, GROQ_API_KEY, GROQ_MODEL
from backend.audio_util import ensure_16k_mono
from backend.providers.base import ASRProvider, clean_arabic_word, is_transient_error

try:
    from groq import Groq, RateLimitError
except Exception:  # pragma: no cover
    Groq = None
    RateLimitError = None

MAX_REQUEST_SECONDS = 240.0
SAMPLE_RATE = 16000
MAX_RETRIES = 3

# Safety cap on the per-word VAD-anchor correction: Whisper stamp drift
# rarely exceeds ~0.5s, so a bigger implied shift means the word was
# mis-associated with the slice (e.g. hallucinated words in silence) and
# dragging it by seconds would be wrong. Phrase-boundary truth comes from
# the VAD clamp in segmenter._phrase_split instead.
MAX_ANCHOR_SHIFT = 0.6


class GroqProvider(ASRProvider):
    name = "groq"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or GROQ_API_KEY
        self.model = model or GROQ_MODEL
        self._client = None

    def _api(self) -> Groq:
        if self._client is None:
            if Groq is None:
                raise RuntimeError("groq package not installed (pip install groq)")
            if not self.api_key:
                raise RuntimeError(
                    "GROQ_API_KEY is not set - create a free key at "
                    "https://console.groq.com and set GROQ_API_KEY (or "
                    "ASR_BACKEND=local for the in-process engine)"
                )
            self._client = Groq(api_key=self.api_key)
        return self._client

    def transcribe(self, audio_path: str, slices=None, language: str = "ar",
                   progress_cb: Optional[callable] = None,
                   anchor_slices=None,
                   speech_segs=None) -> dict:
        """slices: merged VAD spans used to pack API requests. anchor_slices:
        raw (unmerged) VAD segments used for the drift correction — anchoring
        per breath-delimited segment keeps the correction local instead of
        smearing non-uniform Whisper drift across minutes (measured: 5-min
        merged slices left per-word residuals up to ~1s). speech_segs: raw
        VAD segments used to cut over-long requests on silence instead of an
        arbitrary time mark. Both default to slices (legacy behavior)."""
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
            bounds = self._plan_bounds(slices, dur, speech_segs)
            words = []
            n_bounds = len(bounds)
            for i, (s, e) in enumerate(bounds):
                words.extend(self._call_or_shrink(audio, s, e, language))
                if progress_cb:
                    progress_cb(round(100.0 * (i + 1) / n_bounds))
            words.sort(key=lambda w: w["start"])
            _anchor_words_to_vad(words, anchor_slices or slices)
            transcript = " ".join(w["text"] for w in words).strip()
            return {
                "transcript": transcript,
                "words": words,
                "bounds": bounds,
            }
        finally:
            # Only delete a conversion THIS call created. When the input was
            # already 16 kHz mono we reused it and must leave it alone.
            if _owned:
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

    def _ensure_16k_mono(self, audio_path: str) -> tuple[str, bool]:
        """Return (wav_path, owned). `owned` False means the input was already
        16 kHz mono WAV and was reused as-is — no second ffmpeg transcode."""
        return ensure_16k_mono(audio_path)

    @staticmethod
    def _plan_bounds(slices, dur: float, speech=None) -> list[tuple[float, float]]:
        """Pack VAD slices into requests of <= MAX_REQUEST_SECONDS. speech:
        raw (unmerged) VAD segments — when a span exceeds the cap it is cut
        at the latest speech-segment end inside the window instead of an
        arbitrary time mark, so requests never split mid-word/mid-ayah.
        Without speech data the cut falls back to the exact cap mark."""
        if slices:
            plan = []
            cur_s = cur_e = None
            for s, e in slices:
                s, e = max(0.0, float(s)), min(dur, float(e))
                if e <= s:
                    continue
                if cur_s is None:
                    cur_s, cur_e = s, e
                elif e - cur_s <= MAX_REQUEST_SECONDS:
                    cur_e = e
                else:
                    plan.append((cur_s, cur_e))
                    cur_s, cur_e = s, e
                while cur_e - cur_s > MAX_REQUEST_SECONDS:
                    cut = _silence_cut(speech, cur_s,
                                       cur_s + MAX_REQUEST_SECONDS)
                    plan.append((cur_s, cut))
                    cur_s = cut
            if cur_s is not None:
                plan.append((cur_s, cur_e))
            if plan:
                return plan
        n = max(1, math.ceil(dur / MAX_REQUEST_SECONDS))
        step = dur / n
        return [(i * step, min(dur, (i + 1) * step)) for i in range(n)]

    def _call(self, audio: np.ndarray, s: float, e: float, language: str) -> list[dict]:
        lo = max(0, int(s * SAMPLE_RATE))
        hi = min(len(audio), int(e * SAMPLE_RATE))
        seg = audio[lo:hi]
        if len(seg) < SAMPLE_RATE * 0.2:
            return []

        os.makedirs(TEMP_DIR, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=".wav", dir=TEMP_DIR)
        os.close(fd)
        sf.write(tmp, seg, SAMPLE_RATE)
        try:
            res = self._create_transcription(tmp, language)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

        words = []
        for item in getattr(res, "words", []) or []:
            if isinstance(item, dict):
                raw_text = item.get("word", "")
                start = float(item.get("start", 0.0))
                end = float(item.get("end", start))
            else:
                raw_text = getattr(item, "word", "")
                start = float(getattr(item, "start", 0.0))
                end = float(getattr(item, "end", start))
            txt = clean_arabic_word(raw_text)
            if not txt:
                continue
            words.append({"text": txt, "start": round(start + s, 3), "end": round(end + s, 3)})
        return words

    def _call_or_shrink(self, audio: np.ndarray, s: float, e: float,
                        language: str, depth: int = 0) -> list[dict]:
        """Call _call for [s,e); if Groq rejects the slice as too large
        (413 — file-size or duration cap), bisect it and retry each half."""
        try:
            return self._call(audio, s, e, language)
        except RuntimeError as exc:
            msg = str(exc).lower()
            too_large = ("413" in msg or "too large" in msg
                         or "content size" in msg or "entity" in msg)
            if not too_large or depth >= 3 or e - s < 15.0:
                raise
            mid = (s + e) / 2
            out = self._call_or_shrink(audio, s, mid, language, depth + 1)
            out.extend(self._call_or_shrink(audio, mid, e, language, depth + 1))
            return out

    def _create_transcription(self, wav_path: str, language: str):
        last_err = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                with open(wav_path, "rb") as f:
                    return self._api().audio.transcriptions.create(
                        file=(os.path.basename(wav_path), f, "audio/wav"),
                        model=self.model,
                        language=language,
                        response_format="verbose_json",
                        timestamp_granularities=["word"],
                    )
            except RateLimitError as exc:  # 429: backoff, then raise
                last_err = exc
                if attempt < MAX_RETRIES:
                    time.sleep(3 * (2 ** attempt))
                    continue
                raise RuntimeError(f"groq rate limited: {exc}") from exc
            except Exception as exc:
                # A dropped connection / timeout used to abort the whole
                # align on the spot. On a free tier with a shared uplink
                # that is a routine blip, so retry it with backoff. A 4xx
                # (bad request, bad key) still fails fast, and the 413
                # "too large" case keeps its message so the caller can
                # bisect the slice.
                if is_transient_error(exc) and attempt < MAX_RETRIES:
                    last_err = exc
                    time.sleep(3 * (2 ** attempt))
                    continue
                raise RuntimeError(f"groq transcription failed: {exc}") from exc
        raise RuntimeError(f"groq transcription failed: {last_err}")


def _silence_cut(speech, span_start: float, hard_cut: float) -> float:
    """Latest speech-segment end in [span_start + 60% of the cap window,
    hard_cut] — cutting a request there lands on real silence. Falls back
    to hard_cut when no segment edge qualifies (always makes progress)."""
    cut = hard_cut
    if speech:
        lo = span_start + 0.6 * (hard_cut - span_start)
        for s, e in speech:
            if lo <= e <= hard_cut and e > span_start:
                cut = e
    return cut if cut > span_start else hard_cut


def _plan_bounds_request(slices, dur: float, speech=None) -> list[tuple[float, float]]:
    """Shared request-packing for hosted providers (groq, elevenlabs):
    pack VAD slices into <= MAX_REQUEST_SECONDS requests, cutting over-long
    spans on silence when raw speech segments are given."""
    return GroqProvider._plan_bounds(slices, dur, speech)


def _anchor_words_to_vad(words: list[dict], slices: list[tuple[float, float]],
                         max_shift: float = MAX_ANCHOR_SHIFT):
    """Whisper word timestamps lag/lead true speech onset by ~0.3-0.5s (they are
    snapped to the model's ~20ms token grid), which reads as 'caption appears
    before the voice'. The reliable reference is the Silero VAD slice bounds:
    first word of a slice should begin AT the slice onset, last word should end
    AT the slice end (speech actually sounds within those bounds).

    Apply a linear two-anchor drift correction per VAD slice (start->word[0],
    end->word[-1]), interpolated by position — same technique pro caption tools
    use for drift. Measured on a real recitation: |err| drops 0.45s -> 0.03s.

    Callers should pass RAW (unmerged) VAD segments: the correction is only
    valid over breath-delimited spans with no internal pauses. The per-word
    offset is capped at max_shift (default MAX_ANCHOR_SHIFT) — a bigger
    implied shift means the word was mis-associated with the slice, and
    phrase-boundary truth comes from the VAD clamp in
    segmenter._phrase_split instead. Precise providers (ElevenLabs Scribe,
    ~50ms boundaries) pass a tighter cap so good stamps are never dragged
    toward a dubious VAD bound.
    """
    if not slices or not words:
        return
    idx = 0
    for s, e in slices:
        lo = s - 0.4
        hi = e + 0.4
        while idx < len(words) and words[idx]["end"] < lo:
            idx += 1
        j = idx
        while j < len(words) and words[j]["start"] <= hi:
            j += 1
        seg = words[idx:j]
        if len(seg) < 2:
            continue
        span = e - s
        if span <= 0:
            continue
        d_start = s - seg[0]["start"]
        d_end = e - seg[-1]["end"]
        for k, w in enumerate(seg):
            frac = (w["start"] - s) / span
            off = d_start + frac * (d_end - d_start)
            off = max(-max_shift, min(max_shift, off))
            w["start"] = round(w["start"] + off, 3)
            w["end"] = round(w["end"] + off, 3)