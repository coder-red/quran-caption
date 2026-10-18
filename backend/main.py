import os
import uuid
import shutil
import subprocess
import threading
import time
import io
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

from backend.models import (
    AlignResponse,
    SurahInfo,
    MatchInfo,
    WordTimestamp,
    ChunkTimestamp,
    Segment,
    ErrorResponse,
)
from backend.aligner import ASREngine
from backend.detector import SurahDetector
from backend.renderer import render_captions, render_preview, DEFAULT_STYLES
from backend.config import TEMP_DIR, CONFIDENCE_THRESHOLD, FFMPEG_PATH

# Caption-level match gate (lower than CONFIDENCE_THRESHOLD): disjoint rough
# matches (e.g. a slightly misheard Fatihah opening at 0.84) are real and
# must caption; overlapping spurious lanes are already resolved by coverage
# vote in the detector. Minutes-long wrong captions scored ~0.57.
MATCH_GATE = 0.80
from backend.segmenter import build_srt, segments_from_word_times, segments_from_provider_times, snap_words_to_silence
from backend.forced_aligner import ForcedAligner
from backend.providers import get_asr_provider


asr_engine = ASREngine()
detector = SurahDetector()
forced_aligner = ForcedAligner()


def _surah_info(surah_id: int) -> SurahInfo:
    from data.quran_text import SURAH_NAMES
    if 1 <= surah_id <= len(SURAH_NAMES):
        s = SURAH_NAMES[surah_id - 1]
        return SurahInfo(id=surah_id, name_ar=s["name_ar"], name_en=s["name_en"])
    return SurahInfo(id=surah_id, name_ar="", name_en="")


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(TEMP_DIR, exist_ok=True)
    yield


app = FastAPI(
    title="Quran Caption API",
    version="1.0.0",
    lifespan=lifespan,
)


def _media_path_or_404(media_id: str) -> str:
    """Resolve a stored /media upload or 404 (engine restarted / unknown id).
    Lets /align and /render reuse the server-side copy so the client uploads
    the file ONCE instead of once per endpoint."""
    path = MEDIA_FILES.get(media_id)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Media not found — re-upload the file")
    return path


@app.post("/align", response_model=AlignResponse | ErrorResponse | dict)
def align_audio(
    file: UploadFile = File(None),
    media_id: str = Form(None),
    duration: float = Form(None),
    fps: float = Form(None),
    width: int = Form(None),
    height: int = Form(None),
    asr_backend: str = Form(None),
    job: int = Form(None),
):
    """Sync mode: returns the full result (legacy / simple clients). Job mode
    (job=1): returns {"job_id"} immediately, work runs in a background thread,
    progress via GET /align/status/{id}, result via GET /align/result/{id} —
    mirrors /render so the UI can show a progress bar (long clips take
    minutes).

    Upload once: pass media_id from an earlier POST /media/{id} instead of
    re-uploading the file — the server reuses its stored copy."""
    if job:
        _purge_old_align_jobs()
        if file is not None and getattr(file, "filename", None):
            content = file.file.read()
            filename = file.filename or "audio"
            media_path = None
        elif media_id:
            media_path = _media_path_or_404(media_id)
            content, filename = None, None
        else:
            raise HTTPException(status_code=400, detail="Upload a file or pass media_id of an earlier /media upload")
        jid = uuid.uuid4().hex
        ALIGN_JOBS[jid] = {
            "status": "queued",
            "progress": 0.0,
            "stage": "queued",
            "result": None,
            "error": None,
            "created": time.time(),
        }
        threading.Thread(
            target=_align_job_worker,
            args=(jid, content, filename, media_path, duration, fps,
                  width, height, asr_backend),
            daemon=True,
        ).start()
        return {"job_id": jid, "status": "queued"}
    if file is not None and getattr(file, "filename", None):
        return _align_impl(file, duration, fps, width, height, asr_backend)
    if media_id:
        return _align_impl(None, duration, fps, width, height, asr_backend,
                           media_path=_media_path_or_404(media_id))
    raise HTTPException(status_code=400, detail="Upload a file or pass media_id of an earlier /media upload")


def _align_impl(
    file,
    duration: float | None = None,
    fps: float | None = None,
    width: int | None = None,
    height: int | None = None,
    asr_backend: str | None = None,
    p=None,
    media_path: str | None = None,
):
    """Shared align pipeline (sync path + background job worker). p: optional
    callable(stage: str, percent: float) for UI progress reporting.
    media_path: reuse a stored /media copy (owned=False — never deleted)."""
    audio_path = None
    sliced_path = None
    temp_slices = []
    owned = media_path is None

    def setp(stage: str, pct: float):
        if p:
            p(stage, pct)

    try:
        if media_path:
            audio_path = media_path
            os.makedirs(TEMP_DIR, exist_ok=True)
        elif file is not None:
            ext = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
            audio_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}{ext}")
            os.makedirs(TEMP_DIR, exist_ok=True)
            with open(audio_path, "wb") as f:
                content = file.file.read()
                f.write(content)
        else:
            return ErrorResponse(error="Upload a file or pass media_id")

        if audio_path.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
            old_path = audio_path
            audio_path, _ = asr_engine.extract_audio(old_path)
            if owned:
                try:
                    os.remove(old_path)
                except OSError:
                    pass
        elif not audio_path.lower().endswith(".wav"):
            old_path = audio_path
            audio_path, _ = asr_engine.extract_audio(old_path)
            if owned:
                try:
                    os.remove(old_path)
                except OSError:
                    pass

        import soundfile as sf
        from backend.vad import get_speech_segments, merge_segments, chunk_for_whisper, speech_span, silence_runs_from_segments
        from bisect import bisect_right

        # Low-RAM hosts (Render free tier, 512MB) die decoding hour-long
        # audio into float32. Refuse BEFORE the full read (header-only
        # probe) with a clear message instead of an OOM kill. Off by
        # default (local dev keeps unlimited); set QCAP_MAX_MINUTES on host.
        try:
            _max_min = float(os.environ.get("QCAP_MAX_MINUTES", "0") or 0)
        except ValueError:
            _max_min = 0.0
        if _max_min > 0:
            try:
                _dur_probe = sf.info(audio_path).duration
            except Exception:
                _dur_probe = 0.0
            if _dur_probe > _max_min * 60:
                return ErrorResponse(
                    error=f"Audio too long ({_dur_probe / 60:.0f} min — this host "
                          f"allows {_max_min:.0f} min; trim the clip and retry)")

        audio_full, sr_full = sf.read(audio_path, dtype="float32")
        if len(audio_full.shape) > 1:
            audio_full = audio_full.mean(axis=1)
        dur = len(audio_full) / sr_full
        setp("analyzing speech…", 8)

        # Pause ground truth for caption splitting: real audio silence, NOT
        # Whisper word-gap math (those stamps drift 200-500ms and bleed into
        # silence). Derived from the unmerged VAD segments — one pass, both
        # silence data and speech region used.
        raw_segs = get_speech_segments(audio_full, sr_full)
        sil_runs = silence_runs_from_segments(raw_segs)
        segs = merge_segments(raw_segs)
        if not segs:
            return ErrorResponse(error="No speech detected in audio")
        setp("speech segments found", 15)

        provider = get_asr_provider(asr_backend)
        if provider is not None:
            return _align_with_provider(provider, audio_path, segs, setp,
                                        sil_runs, raw_segs)

        setp("transcribing (local model)…", 20)
        wchunks = chunk_for_whisper(segs)
        result = asr_engine.transcribe_chunks(audio_path, wchunks)
        asr_engine.unload()
        transcript = result.get("text", "").strip()
        if not transcript:
            return ErrorResponse(error="No speech detected in audio")
        try:
            import json as _json
            with open(os.path.join(TEMP_DIR, "align_stage1.json"), "w", encoding="utf-8") as f:
                _json.dump({
                    "transcript": transcript,
                    "chunk_bounds": result.get("chunk_bounds", []),
                    "chunk_texts": result.get("chunk_texts", []),
                }, f, ensure_ascii=False)
        except Exception:
            pass

        matches = detector.detect_all(transcript)
        if not matches:
            tight = detector.detect_tight(transcript)
            if tight and tight.score >= CONFIDENCE_THRESHOLD:
                matches = [tight]
        if not matches:
            return ErrorResponse(error="Could not identify any surah")
        matches = _gate_matches(matches)
        setp("surah detected, aligning words…", 42)

        chunk_texts = result.get("chunk_texts", [])
        chunk_bounds = result.get("chunk_bounds", [])
        cum = []
        total = 0
        for t in chunk_texts:
            total += len(t.split())
            cum.append(total)

        def _match_bounds(m):
            if not m.word_span or not cum:
                return None
            i0, i1 = m.word_span
            c0 = min(bisect_right(cum, i0), len(chunk_bounds) - 1)
            c1 = min(bisect_right(cum, i1), len(chunk_bounds) - 1)
            return chunk_bounds[c0][0], chunk_bounds[c1][1]

        from data.quran_text import load_ayah_texts, _normalize
        texts_ar = load_ayah_texts()
        all_words = []
        all_segments = []
        match_infos = []
        # Process matches in chronological order (detect_all returns them by
        # chain strength, which can be out of audio order).
        matches_sorted = sorted(matches, key=lambda m: (m.word_span or (0, 0))[0])
        for mi, m in enumerate(matches_sorted):
            start_ayah = min(m.start_ayah, m.end_ayah)
            end_ayah = max(m.start_ayah, m.end_ayah)

            sliced_path = audio_path
            offset = 0.0
            bounds = _match_bounds(m)
            if bounds:
                if mi == 0:
                    s = chunk_bounds[0][0] if chunk_bounds else max(0.0, bounds[0] - 1.0)
                else:
                    s = bounds[0]
                s = max(0.0, s - 1.0)
                e = min(dur, bounds[1] + 1.0)
                if e - s >= 2.0 and e - s < dur - 2.0:
                    sliced_path = asr_engine.slice_audio(audio_path, s, e)
                    offset = s
                    temp_slices.append(sliced_path)

            canon_words = []
            opener_tags: dict[str, int] = {}
            if mi == 0:
                for kind in forced_aligner.detect_leading_openings(sliced_path):
                    tag = {"bismillah": 0, "takbir": -1, "istiadha": -2}[kind]
                    canon_raw = {
                        "bismillah": texts_ar.get("1:1", ""),
                        "takbir": "ٱللَّهُ أَكْبَرُ",
                        "istiadha": "أَعُوذُ بِٱللَّهِ مِنَ ٱلشَّيْطَٰنِ ٱلرَّجِيمِ",
                    }[kind]
                    for word in _normalize(canon_raw).split():
                        canon_words.append((tag, word))

            # Recite only the canon words actually present in the audio —
            # detect_all's word_off gives the exact range per ayah. Force-
            # aligning the full ayah range breaks mid-ayah clip cuts (the
            # unrecited words have no audio frames -> Viterbi fails) and
            # phantom-stretches captions at mid-ayah starts.
            word_off = getattr(m, "word_off", None) or {}
            for ayah in range(start_ayah, end_ayah + 1):
                words = _normalize(texts_ar.get(f"{m.surah}:{ayah}", "")).split()
                first, last = word_off.get(ayah, (0, len(words)))
                canon_words.extend((ayah, w) for w in words[first:last])

            aligned = forced_aligner.align_words(sliced_path, canon_words)
            if aligned:
                for w in aligned:
                    w["start"] = round(w["start"] + offset, 3)
                    w["end"] = round(w["end"] + offset, 3)
                all_words.extend(w for w in aligned if int(w["ayah"]) > 0)
                all_segments.extend(segments_from_word_times(
                    surah=m.surah, start_ayah=start_ayah, end_ayah=end_ayah,
                    word_times=aligned, word_off=word_off,
                    silence_runs=sil_runs,
                ))
            setp(
                f"aligning ayahs {start_ayah}–{end_ayah}…",
                42 + round(54 * (mi + 1) / max(1, len(matches_sorted))),
            )
            match_infos.append(MatchInfo(
                surah=_surah_info(m.surah),
                start_ayah=start_ayah, end_ayah=end_ayah,
                confidence=round(m.score, 4),
            ))

        if not all_segments:
            return ErrorResponse(error="Could not map recitation to specific ayahs")
        setp("building captions…", 97)

        all_segments.sort(key=lambda s: s["start"])
        if all_segments and all_segments[-1]["end"] < dur:
            all_segments[-1]["end"] = round(dur, 3)
        srt_content = build_srt(all_segments, "ar")
        vtt_content = "WEBVTT\n\n" + srt_content.replace(",", ".")
        words = [WordTimestamp(text=w["text"], start=w["start"], end=w["end"]) for w in all_words]

        first = max(
            match_infos,
            key=lambda mi: (mi.end_ayah - mi.start_ayah, mi.confidence),
        ) if match_infos else None
        response = AlignResponse(
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
        # Persist a copy so a client timeout / disconnect doesn't lose the
        # (expensive) result — temp/align_result.json survives the cleanup.
        try:
            import json
            with open(os.path.join(TEMP_DIR, "align_result.json"), "w", encoding="utf-8") as f:
                json.dump(response.model_dump(), f, ensure_ascii=False, default=str)
        except Exception:
            pass
        return response
    except Exception as e:
        return ErrorResponse(error=str(e))
    finally:
        forced_aligner.unload()
        for p in temp_slices:
            if p and p != audio_path and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        if owned and audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass


ALIGN_JOBS: dict[str, dict] = {}
ALIGN_JOBS_LOCK = threading.Lock()

# Only one align pipeline may run at a time. Every job is a daemon thread
# with no other coordination, and the pipeline keeps process-wide state:
# the shared Silero VAD model (vad._VAD_MODEL), plus the whisper /
# wav2vec2 models on the local path — all of which are "one resident at a
# time" by design. Two concurrent /align calls raced on that state and
# hard-crashed the process (STATUS_HEAP_CORRUPTION 0xC0000374). Jobs still
# queue and report progress; they just don't interleave.
ALIGN_PIPELINE_LOCK = threading.Lock()


def _purge_old_align_jobs():
    now = time.time()
    with ALIGN_JOBS_LOCK:
        for jid in list(ALIGN_JOBS):
            job = ALIGN_JOBS[jid]
            if now - job.get("created", 0) > 24 * 3600:
                del ALIGN_JOBS[jid]


class _JobUpload:
    """Minimal UploadFile stand-in so the worker can run _align_impl with the
    bytes already read in the request handler."""

    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self.file = io.BytesIO(content)


def _align_job_worker(
    jid: str,
    content: bytes,
    filename: str,
    media_path: str | None,
    duration: float | None,
    fps: float | None,
    width: int | None,
    height: int | None,
    asr_backend: str | None,
):
    job = ALIGN_JOBS[jid]
    job["status"] = "running"

    def p(stage: str, pct: float):
        job["stage"] = stage
        job["progress"] = round(float(pct), 1)

    try:
        # Wait our turn before touching any shared model state. Tell the
        # client it is queued rather than silently parked at 0%.
        if not ALIGN_PIPELINE_LOCK.acquire(blocking=False):
            p("waiting for the previous job…", 1.0)
            ALIGN_PIPELINE_LOCK.acquire()
        try:
            if media_path:
                result = _align_impl(
                    None, duration, fps, width, height,
                    asr_backend, p=p, media_path=media_path,
                )
            else:
                result = _align_impl(
                    _JobUpload(filename, content), duration, fps, width, height,
                    asr_backend, p=p,
                )
        finally:
            ALIGN_PIPELINE_LOCK.release()
        dumps = result.model_dump()
        job["result"] = dumps
        job["status"] = "error" if "error" in dumps else "done"
        job["progress"] = 100.0
        job["stage"] = "done"
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


@app.get("/align/status/{job_id}")
async def align_status(job_id: str):
    job = ALIGN_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "stage": job.get("stage"),
    }


@app.get("/align/result/{job_id}")
async def align_result(job_id: str):
    job = ALIGN_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in ("done", "error"):
        raise HTTPException(status_code=409, detail="Align not finished yet")
    if job["status"] == "error":
        # ErrorResponse results land here with status "error" but their
        # message in job["result"] (job["error"] is only set on crashes) —
        # prefer it so clients see e.g. "groq transcription failed: ..."
        # instead of a generic "Align failed".
        result = job.get("result") or {}
        return {"error": job.get("error") or result.get("error") or "Align failed"}
    return job["result"]


def _dump_stage1(transcript: str, words: list[dict]):
    try:
        import json
        with open(os.path.join(TEMP_DIR, "align_stage1.json"), "w", encoding="utf-8") as f:
            json.dump({
                "transcript": transcript,
                "words": words,
                "backend": getattr(asr_engine, "model_name", "provider"),
            }, f, ensure_ascii=False)
    except Exception:
        pass


def _gate_matches(matches):
    """Drop low-confidence matches when a good one exists. On repetition-heavy
    clips the detector DP wanders through wrong-ayah lanes (short repeated
    phrases resemble many canon spots, scoring 0.55-0.8) and those would
    caption WRONG text for minutes. Cross-surah duplicates are already
    resolved by coverage vote in the detector; this gate only removes the
    leftover low-score lanes. When nothing reaches the gate, keep everything
    (legacy recall fallback)."""
    if not matches:
        return matches
    if max(m.score for m in matches) < MATCH_GATE:
        return matches
    return [m for m in matches if m.score >= MATCH_GATE]


def _build_align_response(match_infos, words, segments, end_time: float | None = None) -> AlignResponse | None:
    if not segments:
        return None
    segments.sort(key=lambda s: s["start"])
    # Captions must play to the very end: hold the last caption through any
    # trailing outro audio instead of dropping it a second early.
    if end_time is not None and segments[-1]["end"] < float(end_time):
        segments[-1]["end"] = round(float(end_time), 3)
    all_segments = segments
    srt_content = build_srt(all_segments, "ar")
    vtt_content = "WEBVTT\n\n" + srt_content.replace(",", ".")
    word_ts = [
        WordTimestamp(text=w["text"], start=w["start"], end=w["end"])
        for w in words if w.get("text")
    ]
    first = max(
        match_infos,
        key=lambda mi: (mi.end_ayah - mi.start_ayah, mi.confidence),
    ) if match_infos else None
    if first is None:
        return None
    response = AlignResponse(
        surah=first.surah,
        start_ayah=first.start_ayah,
        end_ayah=first.end_ayah,
        confidence=first.confidence,
        words=word_ts,
        chunks=[ChunkTimestamp(text=w.text, start=w.start, end=w.end) for w in word_ts],
        segments=[Segment(**seg) for seg in all_segments],
        srt=srt_content,
        vtt=vtt_content,
        matches=match_infos,
    )
    try:
        import json
        with open(os.path.join(TEMP_DIR, "align_result.json"), "w", encoding="utf-8") as f:
            json.dump(response.model_dump(), f, ensure_ascii=False, default=str)
    except Exception:
        pass
    return response


def _align_with_provider(provider, audio_path: str, segs, p=None, sil_runs=None, raw_segs=None) -> AlignResponse | ErrorResponse:
    """Hosted-ASR path (groq/gemini): word timestamps come from the provider,
    so detection + ayah segmentation happen without forced alignment.
    sil_runs: audio-truth silence regions for phrase splitting. raw_segs:
    raw (unmerged) VAD speech segments — per-segment drift anchoring,
    silence-aware request packing, and VAD-clamped phrase bounds."""
    def setp(stage: str, pct: float):
        if p:
            p(stage, pct)

    # Per-slice progress + retry visibility for EVERY provider: a frozen bar
    # is indistinguishable from a hang (an 8-min Gemini clip used to sit at
    # 15% through one giant request plus minutes of silent backoff). last_pct
    # lets retry notices reuse the current bar position instead of resetting.
    last_pct = {"v": 18.0}

    def _prog(frac, label):
        last_pct["v"] = 18 + min(62, round(62 * float(frac) / 100))
        setp(label, last_pct["v"])

    def _retry(msg):
        setp(msg, last_pct["v"])

    try:
        label = f"transcribing via {provider.name.title()}…"
        if provider.name == "groq":
            result = provider.transcribe(
                audio_path, slices=segs,
                anchor_slices=raw_segs or segs,
                speech_segs=raw_segs,
                progress_cb=lambda frac: _prog(frac, label),
            )
        elif provider.name == "gemini":
            result = provider.transcribe(
                audio_path, slices=segs,
                anchor_slices=raw_segs or segs,
                speech_segs=raw_segs,
                progress_cb=lambda frac: _prog(frac, label),
                on_retry=_retry,
            )
        else:
            result = provider.transcribe(
                audio_path, slices=segs,
                anchor_slices=raw_segs or segs,
                speech_segs=raw_segs,
                progress_cb=lambda frac: _prog(frac, label),
            )
        setp("transcription complete", 80)
    except Exception as e:
        return ErrorResponse(error=str(e))
    transcript = result.get("transcript", "").strip()
    words = result.get("words", [])
    if not transcript:
        return ErrorResponse(error="No speech detected in audio")
    _dump_stage1(transcript, words)
    setp("detecting surah…", 84)
    # Snap Whisper word boundaries to the real silence edges (whisper stamps
    # drift 200-500ms and bleed into silence; stable-ts-style correction
    # against the VAD-derived runs).
    if sil_runs:
        words = snap_words_to_silence(words, sil_runs)
    matches = detector.detect_all(transcript)
    if not matches:
        tight = detector.detect_tight(transcript)
        if tight and tight.score >= CONFIDENCE_THRESHOLD:
            matches = [tight]
    if not matches:
        return ErrorResponse(error="Could not identify any surah")

    setp("building captions…", 94)
    matches = _gate_matches(matches)
    segments = segments_from_provider_times(matches, words, sil_runs, raw_segs)
    match_infos = []
    for m in sorted(matches, key=lambda m: (m.chain[0][0] if m.chain else 0)):
        match_infos.append(MatchInfo(
            surah=_surah_info(m.surah),
            start_ayah=min(m.start_ayah, m.end_ayah),
            end_ayah=max(m.start_ayah, m.end_ayah),
            confidence=round(m.score, 4),
        ))
    try:
        import soundfile as sf
        end_time = sf.info(audio_path).duration
    except Exception:
        end_time = None
    response = _build_align_response(match_infos, words, segments, end_time=end_time)
    if response is None:
        return ErrorResponse(error="Could not map recitation to specific ayahs")
    return response


MEDIA_FILES: dict[str, str] = {}


@app.post("/media/{media_id}")
async def upload_media(media_id: str, file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    path = os.path.join(TEMP_DIR, f"media_{media_id}{ext}")
    os.makedirs(TEMP_DIR, exist_ok=True)
    with open(path, "wb") as f:
        f.write(await file.read())
    MEDIA_FILES[media_id] = path
    return {"url": f"/media/{media_id}"}


@app.get("/media/{media_id}")
async def get_media(media_id: str):
    path = MEDIA_FILES.get(media_id)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Media not found")
    return FileResponse(path, media_type="video/mp4")


@app.get("/media/{media_id}/preview")
def media_preview(media_id: str):
    """Low-res transcoded preview for embedding as a data URI (browser can't
    reach the API port directly on Spaces, so bytes must be small)."""
    path = MEDIA_FILES.get(media_id)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Media not found")
    out = os.path.join(TEMP_DIR, f"preview_{media_id}.mp4")
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(path):
        # -avoid_negative_ts make_zero normalizes the ~40ms container A/V
        # start offset user MP4s carry (video start 0.04 vs audio 0.0): both
        # streams start at zero in the preview, so the audio-derived caption
        # timestamps line up with what the browser plays frame-for-frame.
        cmd = [
            FFMPEG_PATH, "-y", "-i", path,
            "-vf", "scale=320:-2", "-r", "24",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "32",
            "-c:a", "aac", "-b:a", "32k", "-ac", "1",
            "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart", out,
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=1200)
        except Exception:
            r = None
        if r is None or r.returncode != 0 or not os.path.exists(out):
            return FileResponse(path, media_type="video/mp4")
    return FileResponse(out, media_type="video/mp4")


RENDER_JOBS: dict[str, dict] = {}
RENDER_JOBS_LOCK = threading.Lock()


def _parse_style_form(
    font_size: int = 48,
    font_family: str = "Segoe UI",
    font_color: str = "#FFFFFF",
    outline_color: str = "#000000",
    outline_size: int = 2,
    shadow_size: int = 0,
    bold: bool = False,
    box: bool = False,
    position: str = "bottom",
) -> dict:
    return {
        "font_size": max(16, min(int(font_size), 160)),
        "font_family": font_family or "Segoe UI",
        "font_color": font_color or "#FFFFFF",
        "outline_color": outline_color or "#000000",
        "outline_size": max(0, min(int(outline_size), 8)),
        "shadow_size": max(0, min(int(shadow_size), 8)),
        "bold": bool(bold),
        "box": bool(box),
        "position": position if position in ("top", "middle", "bottom") else "bottom",
    }


def _render_job_worker(
    job_id: str,
    video_path: str,
    srt_content: str,
    styles: dict,
    width: int | None,
    height: int | None,
    cleanup_input: bool = True,
):
    job = RENDER_JOBS[job_id]
    try:
        job["status"] = "rendering"
        job["encoder"] = "detecting..."
        from backend.renderer import detect_encoder
        job["encoder"] = detect_encoder()

        def on_progress(p: float):
            job["progress"] = p

        output_path = render_captions(
            video_path=video_path,
            srt_content=srt_content,
            styles=styles,
            width=width,
            height=height,
            progress_cb=on_progress,
        )
        job["status"] = "done"
        job["progress"] = 100.0
        job["output_path"] = output_path
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
    finally:
        if cleanup_input:
            try:
                if os.path.exists(video_path):
                    os.remove(video_path)
            except OSError:
                pass


def _purge_old_jobs():
    now = time.time()
    with RENDER_JOBS_LOCK:
        for jid in list(RENDER_JOBS):
            job = RENDER_JOBS[jid]
            if now - job.get("created", 0) > 12 * 3600:
                if job.get("output_path") and os.path.exists(job["output_path"]):
                    try:
                        os.remove(job["output_path"])
                    except OSError:
                        pass
                del RENDER_JOBS[jid]


@app.post("/render")
async def render_video(
    file: UploadFile = File(None),
    media_id: str = Form(None),
    srt: str = Form(...),
    width: int = Form(None),
    height: int = Form(None),
    font_size: int = Form(48),
    font_family: str = Form("Segoe UI"),
    font_color: str = Form("#FFFFFF"),
    outline_color: str = Form("#000000"),
    outline_size: int = Form(2),
    shadow_size: int = Form(0),
    bold: bool = Form(False),
    box: bool = Form(False),
    position: str = Form("bottom"),
):
    _purge_old_jobs()
    video_path = None
    cleanup_input = True
    try:
        if file is not None and getattr(file, "filename", None):
            ext = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
            video_path = os.path.join(TEMP_DIR, f"input_{uuid.uuid4()}{ext}")
            os.makedirs(TEMP_DIR, exist_ok=True)
            with open(video_path, "wb") as f:
                f.write(await file.read())
        elif media_id:
            video_path = _media_path_or_404(media_id)
            cleanup_input = False
        else:
            raise HTTPException(status_code=400, detail="Upload a file or pass media_id of an earlier /media upload")

        styles = _parse_style_form(
            font_size=font_size,
            font_family=font_family,
            font_color=font_color,
            outline_color=outline_color,
            outline_size=outline_size,
            shadow_size=shadow_size,
            bold=bold,
            box=box,
            position=position,
        )

        job_id = uuid.uuid4().hex
        RENDER_JOBS[job_id] = {
            "status": "queued",
            "progress": 0.0,
            "error": None,
            "output_path": None,
            "encoder": None,
            "created": time.time(),
        }

        thread = threading.Thread(
            target=_render_job_worker,
            args=(job_id, video_path, srt, styles, width, height, cleanup_input),
            daemon=True,
        )
        thread.start()

        return {"job_id": job_id, "status": "queued"}
    except HTTPException:
        raise
    except Exception as e:
        if cleanup_input and video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except OSError:
                pass
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/render/status/{job_id}")
async def render_status(job_id: str):
    job = RENDER_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": round(job["progress"], 1),
        "error": job["error"],
        "encoder": job.get("encoder"),
    }


@app.get("/render/result/{job_id}")
async def render_result(job_id: str):
    job = RENDER_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] == "error":
        raise HTTPException(status_code=500, detail=job["error"] or "Render failed")
    if job["status"] != "done" or not job.get("output_path"):
        raise HTTPException(status_code=409, detail="Render not finished yet")
    output_path = job["output_path"]
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file missing")
    filename = f"captioned_{job_id[:8]}.mp4"
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=filename,
    )


@app.post("/render/preview")
async def render_preview_frame(
    file: UploadFile = File(...),
    srt: str = Form(...),
    frame_time: float = Form(1.0),
    font_size: int = Form(48),
    font_family: str = Form("Segoe UI"),
    font_color: str = Form("#FFFFFF"),
    outline_color: str = Form("#000000"),
    outline_size: int = Form(2),
    shadow_size: int = Form(0),
    bold: bool = Form(False),
    box: bool = Form(False),
    position: str = Form("bottom"),
):
    video_path = None
    try:
        ext = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
        video_path = os.path.join(TEMP_DIR, f"preview_{uuid.uuid4()}{ext}")
        os.makedirs(TEMP_DIR, exist_ok=True)
        with open(video_path, "wb") as f:
            f.write(await file.read())

        styles = _parse_style_form(
            font_size=font_size,
            font_family=font_family,
            font_color=font_color,
            outline_color=outline_color,
            outline_size=outline_size,
            shadow_size=shadow_size,
            bold=bold,
            box=box,
            position=position,
        )

        png_path = render_preview(video_path, srt, styles, frame_time)
        return FileResponse(
            png_path,
            media_type="image/png",
            filename="caption_preview.png",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except OSError:
                pass


@app.get("/health")
async def health():
    return {"status": "ok"}
