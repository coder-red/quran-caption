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


@app.post("/align", response_model=AlignResponse | ErrorResponse | dict)
def align_audio(
    file: UploadFile = File(...),
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
    minutes)."""
    if job:
        _purge_old_align_jobs()
        content = file.file.read()
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
            args=(jid, content, file.filename or "audio", duration, fps,
                  width, height, asr_backend),
            daemon=True,
        ).start()
        return {"job_id": jid, "status": "queued"}
    return _align_impl(file, duration, fps, width, height, asr_backend)


def _align_impl(
    file,
    duration: float | None = None,
    fps: float | None = None,
    width: int | None = None,
    height: int | None = None,
    asr_backend: str | None = None,
    p=None,
):
    """Shared align pipeline (sync path + background job worker). p: optional
    callable(stage: str, percent: float) for UI progress reporting."""
    audio_path = None
    sliced_path = None
    temp_slices = []

    def setp(stage: str, pct: float):
        if p:
            p(stage, pct)

    try:
        ext = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
        audio_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}{ext}")
        os.makedirs(TEMP_DIR, exist_ok=True)
        with open(audio_path, "wb") as f:
            content = file.file.read()
            f.write(content)

        if audio_path.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
            old_path = audio_path
            audio_path, _ = asr_engine.extract_audio(old_path)
            try:
                os.remove(old_path)
            except OSError:
                pass
        elif not audio_path.lower().endswith(".wav"):
            old_path = audio_path
            audio_path, _ = asr_engine.extract_audio(old_path)
            try:
                os.remove(old_path)
            except OSError:
                pass

        import soundfile as sf
        from backend.vad import get_speech_segments, merge_segments, chunk_for_whisper, speech_span, silence_runs_from_segments
        from bisect import bisect_right

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
            return _align_with_provider(provider, audio_path, segs, setp, sil_runs)

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
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass


ALIGN_JOBS: dict[str, dict] = {}
ALIGN_JOBS_LOCK = threading.Lock()


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
        result = _align_impl(
            _JobUpload(filename, content), duration, fps, width, height,
            asr_backend, p=p,
        )
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
        return {"error": job.get("error") or "Align failed"}
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


def _align_with_provider(provider, audio_path: str, segs, p=None, sil_runs=None) -> AlignResponse | ErrorResponse:
    """Hosted-ASR path (groq/gemini): word timestamps come from the provider,
    so detection + ayah segmentation happen without forced alignment.
    sil_runs: audio-truth silence regions for phrase splitting."""
    def setp(stage: str, pct: float):
        if p:
            p(stage, pct)

    try:
        if provider.name == "groq":
            result = provider.transcribe(
                audio_path, slices=segs,
                progress_cb=lambda frac: setp(
                    "transcribing via Groq…", 18 + min(62, round(62 * frac / 100))
                ),
            )
        else:
            result = provider.transcribe(audio_path, slices=segs)
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
    segments = segments_from_provider_times(matches, words, sil_runs)
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
        cmd = [
            FFMPEG_PATH, "-y", "-i", path,
            "-vf", "scale=320:-2", "-r", "24",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "32",
            "-c:a", "aac", "-b:a", "32k", "-ac", "1",
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
    file: UploadFile = File(...),
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
    try:
        ext = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
        video_path = os.path.join(TEMP_DIR, f"input_{uuid.uuid4()}{ext}")
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
            args=(job_id, video_path, srt, styles, width, height),
            daemon=True,
        )
        thread.start()

        return {"job_id": job_id, "status": "queued"}
    except Exception as e:
        if video_path and os.path.exists(video_path):
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
