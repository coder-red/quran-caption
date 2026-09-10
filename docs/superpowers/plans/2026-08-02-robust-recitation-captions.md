# Robust Recitation → Captions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make /align robust to background music (Silero VAD), multi-surah clips, and mid-ayah starts via a global canon word-index alignment, verified TDD.

**Architecture:** VAD segments → VAD-driven Whisper chunks → `detect_all()` global canon alignment → per-match forced alignment on VAD-sliced audio → multi-match response.

**Tech Stack:** FastAPI, torch, transformers (Whisper + Wav2Vec2ForCTC), silero-vad, numpy, rapidfuzz, FFmpeg.

**Spec:** `docs/superpowers/specs/2026-08-02-robust-recitation-captions-design.md`

## Global Constraints

- Canon Quran text is the source of truth; never put ASR words into captions.
- Forced CTC alignment (wav2vec2) is the ONLY timestamp source of truth.
- Whisper timestamp tokens are unusable — never derive timing from them.
- Lead-in Bismillah: `detect_bismillah()` prepends ayah-0 canon words; segments must skip ayah 0.
- 8GB RAM, CPU-only: only one model resident at a time (`unload()` after ASR, wav2vec2 released in `finally`). Never run wav2vec2 on unsliced long audio.
- Repo test convention: plain runnable scripts `test_*.py` (no pytest dependency).
- No code comments unless asked; follow existing style (type hints, `_normalize` from `data.quran_text`).

---

### Task 1: Install and smoke-test silero-vad

**Files:**
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: nothing
- Produces: `import silero_vad` works (used by Task 2)

- [x] **Step 1: Add dependency**

Add `silero-vad` to `requirements.txt`.

- [x] **Step 2: Install and verify import**

Run: `pip install silero-vad`
Then: `python -c "import silero_vad; print('ok')"`
Expected: prints `ok`. If the package name differs on this platform, resolve before proceeding (fallback energy gate in Task 2 protects us regardless).

- [x] **Step 3: Verify inference works offline**

Run:
```python
python -c "import torch, numpy as np; from silero_vad import load_silero_vad, get_speech_timestamps; m=load_silero_vad(); a=np.zeros(16000*2, dtype=np.float32); ts=get_speech_timestamps(torch.from_numpy(a), m, sampling_rate=16000); print('segments:', len(ts))"
```
Expected: prints `segments: 0` (silence → no speech). If it tries to download a model from the hub and fails offline, note the model path so Task 2 can preload it.

---

### Task 2: `backend/vad.py` — Silero VAD + fallback, chunking

**Files:**
- Create: `backend/vad.py`
- Test: `test_vad.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `get_speech_segments(audio: np.ndarray, sr: int = 16000, threshold: float = 0.5) -> list[tuple[float, float]]`
  - `merge_segments(segments: list[tuple[float, float]], max_gap: float = 2.0) -> list[tuple[float, float]]`
  - `chunk_for_whisper(segments: list[tuple[float, float]], target: float = 30.0, max_len: float = 30.0) -> list[tuple[float, float]]`
  - `speech_span(segments: list[tuple[float, float]], pad: float = 1.5) -> tuple[float, float] | None`
  - `_energy_segments(audio: np.ndarray, sr: int, hop_s: float = 0.05, min_sec: float = 4.0) -> list[tuple[float, float]]` (fallback, extracted from `aligner.detect_speech_span`)

- [x] **Step 1: Write the failing tests**

Create `test_vad.py`:

```python
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import soundfile as sf
from backend.vad import get_speech_segments, merge_segments, chunk_for_whisper, speech_span

sr = 16000

# 1. Silence -> no segments
sil = np.zeros(3 * sr, dtype=np.float32)
assert get_speech_segments(sil) == [], "silence must produce no segments"

# 2. Real recitation -> at least one segment
audio, _ = sf.read("data/test_ikhlas_16k.wav")
segs = get_speech_segments(audio)
assert len(segs) >= 1, f"expected speech, got {segs}"

# 3. Speech with 1s gaps merges when max_gap=2.0
segs = [(0.0, 1.0), (2.5, 3.5)]
merged = merge_segments(segs, max_gap=2.0)
assert merged == [(0.0, 3.5)], f"expected merge, got {merged}"

# 4. merge_segments keeps distant segments separate
segs = [(0.0, 1.0), (5.0, 6.0)]
assert merge_segments(segs, max_gap=2.0) == [(0.0, 1.0), (5.0, 6.0)]

# 5. chunk_for_whisper never exceeds max_len; long segments get split
chunks = chunk_for_whisper([(0.0, 20.0), (22.0, 40.0), (45.0, 60.0)])
for s, e in chunks:
    assert e - s <= 30.0 + 1e-6, f"chunk too long: {s}-{e}"
assert chunks[0][0] == 0.0 and chunks[-1][1] == 60.0
assert len(chunks) == 3, f"expected 3 chunks, got {chunks}"

# 5b. A single continuous 75s speech segment gets split into 30s pieces
chunks = chunk_for_whisper([(0.0, 75.0)])
assert chunks == [(0.0, 30.0), (30.0, 60.0), (60.0, 75.0)], chunks

# 5c. Close segments merge while total stays <= max_len
chunks = chunk_for_whisper([(0.0, 10.0), (12.0, 20.0), (22.0, 29.0)])
assert chunks == [(0.0, 29.0)], chunks

# 6. speech_span pads and returns first-to-last
sp = speech_span([(10.0, 20.0)])
assert sp == (8.5, 21.5), f"expected padded span, got {sp}"

# 7. speech_span on empty -> None
assert speech_span([]) is None

# 8. energy fallback still works
from backend.vad import _energy_segments
fallback = _energy_segments(audio)
assert fallback and fallback[0][0] < 8.0, f"energy fallback failed: {fallback}"

print("test_vad.py ALL PASSED")
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python test_vad.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.vad'`.

- [x] **Step 3: Write minimal implementation**

Create `backend/vad.py`:

```python
"""Speech activity detection for the caption pipeline.

Primary: Silero VAD (neural — robust to background music).
Fallback: RMS energy segmentation (clean-audio only).
Chunking follows the WhisperX cut-and-merge pattern: pack speech into
~30s chunks with boundaries on silence so Whisper never splits mid-ayah.
"""

import numpy as np

_MAX_CHUNK = 30.0
_TARGET_CHUNK = 30.0


def _silero_available() -> bool:
    try:
        import silero_vad  # noqa: F401
        return True
    except Exception:
        return False


def get_speech_segments(
    audio: np.ndarray, sr: int = 16000, threshold: float = 0.5
) -> list[tuple[float, float]]:
    if _silero_available():
        try:
            import torch
            from silero_vad import get_speech_timestamps, load_silero_vad

            model = load_silero_vad()
            ts = get_speech_timestamps(
                torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)),
                model,
                sampling_rate=sr,
                threshold=threshold,
            )
            return [(t["start"] / sr, t["end"] / sr) for t in ts]
        except Exception:
            pass
    return _energy_segments(audio, sr)


def _energy_segments(
    audio: np.ndarray, sr: int, hop_s: float = 0.05, min_sec: float = 4.0
) -> list[tuple[float, float]]:
    frame = max(1, int(hop_s * sr))
    n = len(audio) // frame
    if n == 0:
        return []
    rms = np.sqrt(np.mean(audio[: n * frame].reshape(-1, frame) ** 2, axis=1))
    thr = max(0.001, 0.05 * float(rms.max()))
    voiced = np.where(rms > thr)[0]
    if len(voiced) == 0:
        return []
    merged = []
    s0 = prev = voiced[0]
    for i in voiced[1:]:
        if i - prev > int(1.5 / hop_s):
            merged.append((s0 * hop_s, (prev + 1) * hop_s))
            s0 = i
        prev = i
    merged.append((s0 * hop_s, (prev + 1) * hop_s))
    return [(s, e) for s, e in merged if e - s >= min_sec]


def merge_segments(
    segments: list[tuple[float, float]], max_gap: float = 2.0
) -> list[tuple[float, float]]:
    if not segments:
        return []
    out = []
    cs, ce = segments[0]
    for s, e in segments[1:]:
        if s - ce <= max_gap:
            ce = max(ce, e)
        else:
            out.append((cs, ce))
            cs, ce = s, e
    out.append((cs, ce))
    return out


def chunk_for_whisper(
    segments: list[tuple[float, float]],
    target: float = _TARGET_CHUNK,
    max_len: float = _MAX_CHUNK,
) -> list[tuple[float, float]]:
    if not segments:
        return []
    pieces = []
    for s, e in segments:
        while e - s > max_len:
            pieces.append((s, s + max_len))
            s += max_len
        if e > s:
            pieces.append((s, e))
    chunks = []
    cs, ce = pieces[0]
    for s, e in pieces[1:]:
        if e - cs <= max_len:
            ce = e
        else:
            chunks.append((cs, ce))
            cs, ce = s, e
    chunks.append((cs, ce))
    return chunks


def speech_span(
    segments: list[tuple[float, float]], pad: float = 1.5
) -> tuple[float, float] | None:
    if not segments:
        return None
    return (max(0.0, segments[0][0] - pad), segments[-1][1] + pad)
```

- [x] **Step 4: Run tests to verify they pass**

Run: `python test_vad.py`
Expected: PASS, prints `test_vad.py ALL PASSED`. Fix the `chunk_for_whisper` packing logic if assertion 5 fails (the intent: merge segments greedily while total stays ≤ max_len; a segment may extend a chunk past `target` but never past `max_len`).

---

### Task 3: Detector global canon alignment (`detect_all`)

**Files:**
- Modify: `backend/detector.py`
- Test: `test_detector_all.py`

**Interfaces:**
- Consumes: `data.quran_text.load_ayah_texts()`, `data.quran_text._normalize`
- Produces:
  - `AyahMatch` (existing class, unchanged fields)
  - `SurahDetector.detect_all(transcript: str, min_score: float = 0.55) -> list[AyahMatch]` — ordered chronologically
  - `SurahDetector._build_index()` — lazy cache: `self._flat: list[tuple[int, int, str]]`, `self._word_pos: dict[str, list[int]]`

- [x] **Step 1: Write the failing tests**

Create `test_detector_all.py`:

```python
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from backend.detector import SurahDetector
from data.quran_text import load_ayah_texts, _normalize

texts = load_ayah_texts()
d = SurahDetector()

def txt(key):
    return _normalize(texts[key])

# 1. Clean full surah (Ikhlas) -> one match 112:1-4, score >= 0.9
t = " ".join(txt(f"112:{a}") for a in range(1, 5))
m = d.detect_all(t)
assert len(m) == 1 and m[0].surah == 112 and m[0].start_ayah == 1 and m[0].end_ayah == 4
assert m[0].score >= 0.9, m[0]

# 2. With leading Bismillah -> same match (bismillah stripped)
t = " ".join([txt("1:1")] + [txt(f"112:{a}") for a in range(1, 5)])
m = d.detect_all(t)
assert len(m) == 1 and m[0].surah == 112 and m[0].start_ayah == 1 and m[0].end_ayah == 4

# 3. Mid-ayah start (Fatihah ayah 2 onwards, no ayah 1)
t = " ".join(txt(f"1:{a}") for a in range(2, 8))
m = d.detect_all(t)
assert len(m) == 1 and m[0].surah == 1 and m[0].start_ayah == 2 and m[0].end_ayah == 7, m

# 4. Mid-ayah END (Ikhlas cut at 2.5 words -> ayah 1..3, partial 4th)
t = " ".join([txt(f"112:{a}") for a in range(1, 4)] + [txt("112:4").split()[0]])
m = d.detect_all(t)
assert len(m) == 1 and m[0].surah == 112 and m[0].start_ayah == 1, m

# 5. Multi-surah (Ikhlas then Kawthar) -> 2 matches in order
t = " ".join(txt(f"112:{a}") for a in range(1, 5)) + " " + " ".join(txt(f"108:{a}") for a in range(1, 4))
m = d.detect_all(t)
assert len(m) == 2, m
assert (m[0].surah, m[0].start_ayah, m[0].end_ayah) == (112, 1, 4), m[0]
assert (m[1].surah, m[1].start_ayah, m[1].end_ayah) == (108, 1, 3), m[1]

# 6. ASR noise: 2 substituted words still detected (score drops but >= 0.55)
words = txt("112:1").split() + txt("112:2").split() + txt("112:3").split() + txt("112:4").split()
words[3] = "xxx"
words[7] = "yyy"
t = " ".join(words)
m = d.detect_all(t)
assert len(m) == 1 and m[0].surah == 112 and m[0].score >= 0.55, m

# 7. Repeated ayah text (recitation repeats) -> single match, no crash
t = txt("112:1") + " " + txt("112:1")
m = d.detect_all(t)
assert len(m) >= 1 and all(x.surah == 112 for x in m)

# 8. Gibberish -> no matches
m = d.detect_all("zzz qqq www eee rrr ttt")
assert m == [], m

print("test_detector_all.py ALL PASSED")
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python test_detector_all.py`
Expected: FAIL with `AttributeError: 'SurahDetector' object has no attribute 'detect_all'`.

- [x] **Step 3: Write minimal implementation**

Add to `backend/detector.py`:

```python
    def _build_index(self):
        if self._flat is not None:
            return
        from data.quran_text import load_ayah_texts
        texts = load_ayah_texts()
        flat = []
        word_pos = {}
        for key, text in texts.items():
            surah, ayah = (int(p) for p in key.split(":"))
            for word in _normalize(text).split():
                idx = len(flat)
                flat.append((surah, ayah, word))
                word_pos.setdefault(word, []).append(idx)
        self._flat = flat
        self._word_pos = word_pos

    def detect_all(self, transcript: str, min_score: float = 0.55) -> list[AyahMatch]:
        """Global word-anchor alignment of the transcript against the canon.
        Returns chronologically ordered matches (one per surah range found)."""
        self._build_index()
        flat = self._flat
        word_pos = self._word_pos

        norm = _normalize(transcript)
        bismillah = _normalize("بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ")
        if norm.startswith(bismillah):
            norm = norm[len(bismillah):].strip()
        twords = norm.split()
        if not twords:
            return []

        # Greedy anchor chain: transcript word -> next canon position.
        chain: list[int] = []
        last = -1
        for w in twords:
            cands = word_pos.get(w, [])
            nxt = None
            for c in cands:
                if c > last:
                    nxt = c
                    break
            if nxt is not None:
                chain.append(nxt)
                last = nxt

        if not chain:
            return []

        # Split chain into consecutive runs (gap <= 3 tolerated deletions).
        runs: list[list[int]] = [[chain[0]]]
        for p in chain[1:]:
            if p - runs[-1][-1] <= 3:
                runs[-1].append(p)
            else:
                runs.append([p])
        runs = [r for r in runs if len(r) >= 2]

        matches = []
        for run in runs:
            s = run[0]
            e = run[-1]
            span_words = e - s + 1
            matched = len(run)
            score = min(1.0, matched / max(1, span_words + 2))
            if score < min_score:
                continue
            matches.append(AyahMatch(
                surah=flat[s][0],
                start_ayah=flat[s][1],
                end_ayah=flat[e][1],
                score=score,
                text=" ".join(w for _, _, w in flat[s:e + 1]),
            ))
        return matches
```

Add `self._flat = None` / `self._word_pos = None` to `__init__`.

- [x] **Step 4: Run tests to verify they pass**

Run: `python test_detector_all.py`
Expected: PASS. If assertion 5 (multi-surah) fails because both matches overlap into one run, split runs on surah change inside `detect_all` (add: if `flat[p][0] != flat[run[-1]][0]`, close the run).

---

### Task 4: `/align` multi-match pipeline

**Files:**
- Modify: `backend/main.py`
- Modify: `backend/models.py`
- Modify: `backend/aligner.py` (ASR chunk loop driven by VAD chunks)
- Test: `test_align_multimatch.py`

**Interfaces:**
- Consumes: `backend.vad` (Task 2), `SurahDetector.detect_all` (Task 3)
- Produces:
  - `models.MatchInfo(surah: SurahInfo, start_ayah: int, end_ayah: int, confidence: float)`
  - `models.AlignResponse.matches: list[MatchInfo]`
  - `ASREngine.transcribe_chunks(audio_path, chunks: list[tuple[float, float]]) -> dict` (same dict shape as `transcribe_word_timestamps`)
  - `/align` returns `AlignResponse` with `matches`; top-level fields = first match; `segments`/`srt`/`vtt` span all matches.

- [x] **Step 1: Write the failing tests**

Create `test_align_multimatch.py`:

```python
import sys, os, warnings, json
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

# Synthetic multi-surah clip: ikhlas + 2s silence + kawthar, 10s lead-in
a1, sr = sf.read("data/test_ikhlas_16k.wav")
a2, _ = sf.read("data/test_kawthar_16k.wav")
clip = np.concatenate([np.zeros(int(10 * sr)), a1, np.zeros(int(2 * sr)), a2, np.zeros(int(5 * sr))])
sf.write("data/test_multisurah.wav", clip, sr)

with open("data/test_multisurah.wav", "rb") as f:
    r = client.post("/align", files={"file": ("test.wav", f, "audio/wav")}, timeout=900)

print("status:", r.status_code)
data = r.json()
assert "error" not in data, data.get("error")
assert len(data["matches"]) >= 2, data["matches"]
first, second = data["matches"][0], data["matches"][1]
assert first["surah"]["id"] == 112 and second["surah"]["id"] == 108, data["matches"]
assert data["surah"]["id"] == 112
assert data["srt"], "empty srt"
ayahs = [s["ayah"] for s in data["segments"]]
assert 1 in ayahs and 4 in ayahs, ayahs
print("matches:", [(m["surah"]["name_en"], m["start_ayah"], m["end_ayah"]) for m in data["matches"]])
print("segments:", [(s["ayah"], s["start"], s["end"]) for s in data["segments"]])
print("test_align_multimatch.py ALL PASSED")
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python test_align_multimatch.py`
Expected: FAIL — response has no `matches` key (`KeyError` or assertion).

- [x] **Step 3: Implement `transcribe_chunks` in aligner.py**

```python
    def transcribe_chunks(self, audio_path: str, chunks: list[tuple[float, float]]) -> dict:
        """Whisper ASR over explicit VAD chunks (absolute timestamps)."""
        self.load()
        audio_input, sr = sf.read(audio_path)
        if sr != 16000:
            import librosa
            audio_input = librosa.resample(audio_input, orig_sr=sr, target_sr=16000)
            sr = 16000
        if len(audio_input.shape) > 1:
            audio_input = audio_input.mean(axis=1)
        all_chunks = []
        texts = []
        for (cs, ce) in chunks:
            seg = audio_input[int(cs * sr):int(ce * sr)]
            if len(seg) < sr * 0.2:
                continue
            feats = self.processor(seg, sampling_rate=sr, return_tensors="pt").input_features
            with torch.no_grad():
                generated = self.model.generate(
                    feats,
                    forced_decoder_ids=self._forced_decoder_ids,
                    return_timestamps=True,
                    output_attentions=False,
                )
            text = self.processor.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
            if text:
                texts.append(text)
            for c in self._extract_timestamps(generated, seg, sr):
                c["start"] = round(c["start"] + cs, 3)
                c["end"] = round(c["end"] + cs, 3)
                all_chunks.append(c)
        return {
            "text": " ".join(texts).strip(),
            "chunks": all_chunks,
            "words": self._words_from_chunks(all_chunks),
        }
```

Keep `transcribe_word_timestamps` (used by tests) delegating to `transcribe_chunks` with one full-range chunk.

- [x] **Step 4: Implement models.py additions**

```python
class MatchInfo(BaseModel):
    surah: SurahInfo
    start_ayah: int
    end_ayah: int
    confidence: float
```

Add `matches: list[MatchInfo] = []` to `AlignResponse`.

- [x] **Step 5: Rework /align body in main.py**

Replace the section from `result = asr_engine.transcribe_word_timestamps(audio_path)` through segment building:

```python
        import soundfile as sf
        from backend.vad import get_speech_segments, merge_segments, chunk_for_whisper, speech_span

        audio_full, sr_full = sf.read(audio_path)
        if len(audio_full.shape) > 1:
            audio_full = audio_full.mean(axis=1)

        segs = merge_segments(get_speech_segments(audio_full, sr_full))
        if not segs:
            return ErrorResponse(error="No speech detected in audio")

        wchunks = chunk_for_whisper(segs)
        result = asr_engine.transcribe_chunks(audio_path, wchunks)
        asr_engine.unload()
        transcript = result.get("text", "").strip()
        if not transcript:
            return ErrorResponse(error="No speech detected in audio")

        matches = detector.detect_all(transcript)
        if not matches:
            tight = detector.detect_tight(transcript)
            if tight and tight.score >= CONFIDENCE_THRESHOLD:
                matches = [tight]
        if not matches:
            return ErrorResponse(error="Could not identify any surah")

        span = speech_span(segs)
        sliced_path = audio_path
        offset = 0.0
        if span:
            dur = len(audio_full) / sr_full
            s, e = max(0.0, span[0]), min(dur, span[1])
            if e - s < dur - 2.0:
                sliced_path = asr_engine.slice_audio(audio_path, s, e)
                offset = s

        from data.quran_text import load_ayah_texts, _normalize, SURAH_NAMES
        texts_ar = load_ayah_texts()
        all_words = []
        all_segments = []
        match_infos = []
        for mi, m in enumerate(matches):
            start_ayah = min(m.start_ayah, m.end_ayah)
            end_ayah = max(m.start_ayah, m.end_ayah)
            canon_words = []
            if mi == 0 and forced_aligner.detect_bismillah(sliced_path):
                for word in _normalize(texts_ar.get("1:1", "")).split():
                    canon_words.append((0, word))
            for ayah in range(start_ayah, end_ayah + 1):
                for word in _normalize(texts_ar.get(f"{m.surah}:{ayah}", "")).split():
                    canon_words.append((ayah, word))
            aligned = forced_aligner.align_words(sliced_path, canon_words)
            if aligned:
                for w in aligned:
                    w["start"] = round(w["start"] + offset, 3)
                    w["end"] = round(w["end"] + offset, 3)
                all_words.extend(w for w in aligned if int(w["ayah"]) != 0)
                all_segments.extend(segments_from_word_times(
                    surah=m.surah, start_ayah=start_ayah, end_ayah=end_ayah,
                    word_times=aligned,
                ))
            match_infos.append(MatchInfo(
                surah=_surah_info(m.surah),
                start_ayah=start_ayah, end_ayah=end_ayah,
                confidence=round(m.score, 4),
            ))

        if not all_segments:
            return ErrorResponse(error="Could not map recitation to specific ayahs")

        all_segments.sort(key=lambda s: s["start"])
        srt_content = build_srt(all_segments, "ar")
        vtt_content = "WEBVTT\n\n" + srt_content.replace(",", ".")
        words = [WordTimestamp(text=w["text"], start=w["start"], end=w["end"]) for w in all_words]

        first = match_infos[0]
        return AlignResponse(
            surah=first.surah,
            start_ayah=first.start_ayah,
            end_ayah=first.end_ayah,
            confidence=first.confidence,
            words=words,
            chunks=[ChunkTimestamp(text=w.text, start=w.start, end=w.end) for w in words],
            segments=[Segment(**seg) for seg in all_segments],
            srt=srt_content,
            vtt=vtt_content,
            matches=match_infos,
        )
```

Add helper `_surah_info(surah_id)` (extract existing SURAH_NAMES lookup into it, reuse in /align). Remove the now-unused whisper-chunk fallback block (the forced-align path with `detect_tight` covers the fallback; keep `build_segments` import only if still referenced). Keep the `finally` block exactly as-is (`sliced_path` guard already exists).

- [x] **Step 6: Run tests to verify they pass**

Run: `python test_align_multimatch.py`
Expected: PASS. Note: this loads both models (~2-4 min first run). If the second match (Kawthar) times out on alignment, relax: only force-align the FIRST match and fall back to whisper-chunk segments for later matches (documented in AGENTS.md as acceptable — rough captions).

---

### Task 5: Regression + docs + user clip verification

**Files:**
- Modify: `AGENTS.md`
- Run: all tests

- [x] **Step 1: Run full regression**

Run: `python test_pipeline.py; python test_vad.py; python test_detector_all.py; python test_segmenter.py; python test_align_multimatch.py`
Expected: all PASS.

- [x] **Step 2: Live-server smoke tests**

Start backend, run:
`python test_align.py data\test_ikhlas_16k.wav`
`python test_align.py data\test_realistic.wav`
`python test_align.py data\test_multisurah.wav`
Expected: 200, surah + correct ayahs + segments, `matches` present.

- [ ] **Step 3: User's failing clip (when available)**

Run `python test_align.py <user_clip>`; if it fails, use superpowers:systematic-debugging (root cause before fix). Record the result in AGENTS.md.

- [x] **Step 4: Update AGENTS.md**

- Silero VAD replaces the energy gate; energy is fallback only.
- `detect_all` (global word-index alignment) replaces `detect_tight` as primary; `detect_tight` remains fallback.
- /align returns `matches`; multi-surah supported; mid-ayah starts supported.
- Check off the "Handle edge cases" todo partially (document what's covered).

- [x] **Step 5: Final verification**

Re-run Step 1 suite + restart both servers (backend :8000, frontend :8501) for the user.
