# Quran Caption App - Progress

## Architecture
```
Browser (upload audio + metadata) → POST /align → ASR → ayah detection → forced alignment → timestamps + SRT
Browser (preview via <track>) → POST /render (optional) → FFmpeg burn-in → MP4
```

## Stack
- **Backend**: FastAPI (Python 3.12)
- **Frontend**: Streamlit
- **ASR**: tarteel-ai/whisper-base-ar-quran (HuggingFace transformers)
- **Ayah detection**: rapidfuzz fuzzy matching against Quran text
- **Alignment**: ctc-forced-aligner (jonatasgrosman/wav2vec2-large-xlsr-53-arabic) — Whisper timestamps are unusable (1 ts token per clip), forced CTC alignment is the source of truth
- **Video**: FFmpeg (burn-in)
- **Quran data**: Static JSON (embedded) + English translation (quran_en.json)

## Key facts
- Hosted ASR backends: `ASR_BACKEND=local|groq|gemini` env (default `groq` since 2026-08-14; key from `.env`), plus a per-request `asr_backend` Form override on /align. Provider factory: `backend/providers/__init__.py::get_asr_provider(name)` (returns None → local in-process path).
- `groq` provider (backend/providers/groq.py): `whisper-large-v3-turbo` verbose_json + `timestamp_granularities=["word"]` → REAL word timestamps (no forced alignment). Free tier ~20 rpm / 2k clips/day. VAD slices packed into ≤240s requests (`MAX_REQUEST_SECONDS`); per-slice offsets added; words sanitized to Arabic letters (`clean_arabic_word`). Key: `GROQ_API_KEY`, model: `GROQ_MODEL`. Leading Bismillah words are KEPT in words+transcript (detector strips them from matching; `segmenter.inject_opener_segments` re-injects them as captions).
- **Pause-aware captions** (`split_segments_at_pauses`/`_phrase_split` in segmenter.py): captions are cut at reciter breath pauses — inter-word silences ≥ `PAUSE_GAP` (0.5s) from forced-aligned word times (local path) or provider word timestamps (provider path, incl. opener segments). Each phrase becomes its own segment; canon AR text is re-sliced proportionally by word count (recitation order == text order), EN by proportional cuts snapped to sentence-break punctuation (`_snap_en_boundary`). Too-small phrases (few words / short) merge into their previous region (leading ones fold forward) and never flash. `apply_sync_padding` still enforces non-overlap so captions vanish during pauses and resume at the next phrase.
- Openers ARE captioned: leading takbir "الله أكبر", leading isti'adha "أعوذ بالله…", and every Bismillah become real segments (`inject_opener_segments`, `_leading_opener`) — takbir/isti'adha → surah 1 ayah 0, Bismillah → surah 1 ayah 1 (canon 1:1 text/en). Takbir/isti'adha caption ONLY while contiguous with the lead run (`lead_run` guard) — mid-clip they are legitimate canon (9:72, 29:45, 40:10) and captioning again would double-caption. Opener matching is SPELLING-TOLERANT (`_opener_word_matches`, `match_leading_openers`): accepts canon form OR plain-alef form (U+0670 stripped → الشيطان from ASR, الرحمان vs الرحمن) for every opener word — `detect_bismillah()`→`detect_leading_openings()` (forced_aligner, window 12s) ends-to-end detects takbir→istiadha→bismillah in order; `detector.detect_all` strips the same lead run tolerantly (its Bismillah strip already strips EVERY occurrence). Local path prepends ayah -1 (takbir), -2 (isti'adha), 0 (Bismillah) canon words; `segments_from_word_times` keys word groups by TAG (1,-1)/(1,-2)/(1,0) then maps display to 1:0/1:0/1:1 — the display (1,0) is SHARED by takbir and isti'adha so tag-keying must never be replaced by display-keying. Last segment end is extended to the audio duration in both paths (captions play to the very end; `_build_align_response(end_time=…)`, local path `dur`).
- `AyahMatch.chain` (detector.py): `[(transcript_idx, surah, ayah), ...]` per matched word — lets hosted paths cut ayah segments straight from word timestamps (`segmenter.segments_from_provider_times`). Provider path in main.py: `_align_with_provider` (no wav2vec2, no whisper model load).
- `_build_align_response` (main.py) is the shared response builder (provider path); local forced-align path keeps its own inline builder.
- Quran fine-tuned Whisper emits ~no real timestamps → real per-word times come from wav2vec2 CTC forced alignment (pure-torch Viterbi, log-space 2L+1 states)
- wav2vec2 arabic vocab (51 tokens) has no space char: word sep is `|` (id 4); `_normalize_chars` maps " "→"|", strips diacritics, ة→ه, ى→ي, ٱ→ا
- **Windowed CTC Viterbi** (`_viterbi`, forced_aligner.py): the backtrack matrix + state space only span a sliding ~60s char window (anchored on the previous chunk's best end position, 80-char margin), NOT the whole clip. The full-surah backtrack was `(frames × 2·chars)` uint8 → ~24GB for full Al-Baqarah (30k chars, 2h+) → swap death ("cmd looks stuck"). Windowed: ~300MB max, single-chunk degenerate case = byte-identical to old; verified T=400k/L=30k in 91s. `align_words`/`detect_leading_openings` unchanged (model forward still chunked at 60s).
- ASR on raw MP3 hallucinates → /align always ffmpeg-converts to 16k mono WAV first
- Alafasy clips open with takbir/Bismillah (~0.3–6.2s) not in target ayahs → `detect_bismillah()` greedy-decodes first 8s, prepends ayah 0 canon words; those words are now captioned (local) or re-injected from word times (provider via `inject_opener_segments`)
- `detect_tight` windows slice by WORD count (prefix sums + bisect), not ayah count
- VAD: Silero VAD is PRIMARY speech detection (`backend/vad.py`, neural — robust to background music); RMS energy gate is fallback only. `get_speech_segments` → `merge_segments(max_gap=2.0)` → `chunk_for_whisper` (pack into ≤30s chunks on silence)
- `detect_all` (global word-anchor alignment over canon `(surah, ayah, word)` flat index + `word_pos` dict) is PRIMARY detection: multi-surah + mid-ayah starts, returns ordered `AyahMatch` list each with `word_span` (transcript word indices); `detect_tight` is fallback, `detect` (rapidfuzz) unused by /align
- **Pause truth = audio, NOT Whisper word-gap math** (2026-08-15): Whisper's timestamp head drifts 200–500ms and misplaces silence onto words (buzz#1339 "subtitles linger over silence", stable-ts, whisperX#1247, arXiv 2607.05364 — all say don't trust native word stamps for pauses). `backend/vad.py::silence_runs_from_segments` derives true inter-speech gaps from the UNMERGED VAD segments (one pass — `_align_impl` computes `raw_segs` → `sil_runs`, then merges for chunking; zero extra VAD cost). `segmenter.snap_words_to_silence` pulls Whisper word boundaries in to the silence-run edges (stable-ts-style; runs in `_align_with_provider` for groq+gemini). `_phrase_split` splits when MEASURED silence ≥ `SILENCE_SPLIT` (0.6s) with no silence data falling back to whisper-gap ≥ `PAUSE_GAP` (1.0s — higher bar, gaps are weak evidence). Verified on 2h Baqarah: whisper-gap rule made 48 mid-ayah cuts, only 7 had ≥0.6s real silence (43 phantom splits killed, 2 real ones gained). `test_pause_split.py` (14 tests incl. snapping/silence-evidence) → 18 total pass.
- **Vocative compound fix** (root cause of "caption delayed at ayah start" on Al-Hujurat 49:1): Uthmani writes يَا أَيُّهَا as ONE word `يَـٰٓأَيُّهَا` (superscript alef U+0670; normalized `يايها`, 142 occurrences) but ASR emits two tokens `يا أيها` → `word_pos["يا"]` was empty and `ايها` had no near-49:1 occurrence, so the backward walk stopped at `الذين` and the 49:1 caption started ~4.9s late (at الذين@18.69 vs يا@14.42). Fix in `_build_index`: detect U+0670 vocative compounds (NOT plain-hamza verbs like يَأۡكُلُ), register both tokens (`يا`, `ا`+rest — the alef merges into the compound) at the compound's slot, track `_split_pos`; backward walk uses `bisect_right` + allows `prev == last` only at split positions. Re-verified: 49:1 segment now 13.769–39.893 (was 18.643), all detector/segmenter/multimatch/api tests pass.
- /align force-aligns PER MATCH on its own VAD-bounded slice (`_match_bounds` maps word_span→whisper chunk times); running all matches on one slice stretches words across surahs
- Canon Uthmani text contains U+06E1 (ۡ Quranic sukun) — `_normalize` strips it (must match hardcoded Bismillah strings)
- 8GB RAM, CPU-only: only one model resident at a time; `asr_engine.unload()` after ASR, `forced_aligner.unload()` in /align finally; per-match slices keep wav2vec2 bounded
- UI automation: headless Edge CDP `DOM.setFileInputFiles` wedges the renderer on Streamlit → use in-page DataTransfer: fetch file from local static server (http://127.0.0.1:8899, CORS *) → File → DataTransfer → input.files → dispatch change (see test_e2e_ui.py)
- Streamlit 1.58: data editor renders as `[data-testid="stDataFrame"]` (not stDataEditor); segmented control buttons are `[data-testid^=stBaseButton-segmented_control]`
- Caption player uses `st.html(..., unsafe_allow_javascript=True)` (NOT `st.components.v1.html` — iframe blocks requestFullscreen): fullscreen button on `#capwrap` (video+caption div), playback position persisted to `sessionStorage` key `qcap_pos_<media_id>` every 250ms and restored on rerun → style changes (font/size/color/position…) never restart the video, no ffmpeg involved (all styles are CSS in `applyStyle()`; ffmpeg burn-in happens once, in the "Burn & download" section right below the player — no separate render step). **CRITICAL: `st.html` re-renders the block on Streamlit rerun but DOES NOT re-execute the embedded `<script>`** — the player therefore binds ONCE via doc-level event delegation (`window.__qcap` guard) and reads segments/lang/style from `data-segs`/`data-st`/`data-lang` base64 data-attrs on #capwrap; rerans just swap the attributes. Position-race guard: the 250ms save skips t<=0 / near-end so a fresh element can't clobber a valid saved position. CDP `Runtime.evaluate` needs `userGesture: true` for `requestFullscreen`. Live caption overlay verified in headless Edge (Bismillah text at t=1s, `#cap` visible) in BOTH local (http src) and deployed-emulation (SPACE_ID=1 → data-URI preview → JS converts to Blob URL: `vid.src` becomes `blob:http://…`) modes.
- **Verified slim boot** (clean venv, `deploy/space/requirements.txt`, NO torch installed): `backend.main` imports fine (aligner's ASREngine import is torch-safe at module level); silero missing → energy VAD fallback still detects speech; /align with `asr_backend=groq` + mocked Groq client (FakeClient.audio.transcriptions.create returning `words=` dicts) returns Al-Ikhlas 1–3, confidence 1.0, 4 segments, SRT ok. `get_asr_provider("gemini")` raises clean RuntimeError when gemini deps absent (guard in backend/providers/__init__.py, synced to deploy). `deploy/space` mirrors `backend|frontend|data` exactly (test fixtures intentionally excluded).

## Verified (real Al-Hashr clips, 2026-08-10)
- Mid-ayah START confirmed correct end-to-end: `temp\hashr_mid45.mp4` (59:2-4, clip starts mid-ayah 2 at "مِنَ اللَّهِ") → 7 segments, first caption اللَّهُ 0.29–1.27. Word times verified by cross-check against the 90s clip: mid45 time + 44.5 == 90s time (من 4.2==48.66, وقذف 7.46==51.92, ذلك 34.48==78.92, …). No phantom/stretched lead-in; captions start at the first recited word (clipped text, not full ayah).
- `temp\hashr_90s.mp4` (59:1-4, 90.04s) via live API now returns 200, surah Al-Hashr 1–4, conf 0.8493, 72 words, 10 segments — the "Could not map" failure is gone (word_off fix). Detected: سَبَّحَ 17.49 → وَمَن يُشَآقِّ 85.55–89.98.
- Known edge artifact (acceptable): a clip cut mid-word can make whisper mishear that word (e.g. فَأَتَاهُمُ → "وَأَتَاهُمُ"), dropping it from the detection chain → short uncaptioned gap where the reciter says the clipped word (hashr_mid45: ~1.27–4.15 during "فَأَتَاهُمُ اللَّهُ"). Not a timing error.
- `deploy/space/backend/{detector,main,forced_aligner,segmenter}.py` hash-matched to local (2026-08-10). Tests: pytest test_pause_split + test_groq_provider = 12 passed; test_openers.py ALL PASSED; py_compile clean.
- **Full Al-Baqarah captioned in 9.7 min via groq** (2026-08-14): `temp\baqarah_full_002.mp3` (7576s, 2h06m) → HTTP 200, surah 2 ayahs 2-286, conf 0.8901, 6198 words, 854 segments, last caption ends 7575.93/7575.94. Free tier OK despite clip > 7200s/hr ASH cap (retries recovered). `ASR_BACKEND` default flipped `local` → `groq` (config.py:41, key from .env). Local path (whisper+wav2vec2) is now the fallback only; windowed Viterbi keeps it non-OOM offline.

## Todo

### Phase 1 - Core Engine
- [x] Create project structure
- [x] Install dependencies
- [x] Load Quran text data (114 surahs, 6236 ayahs)
- [x] Build /align endpoint (ASR + detection + alignment)
  - [x] ASR with tarteel-ai model + word timestamps
  - [x] Ayah detection via fuzzy matching
  - [x] Confidence scoring
  - [x] SRT/VTT generation
- [x] Build /render endpoint (FFmpeg burn-in)

### Phase 2 - Frontend
- [x] Streamlit UI with upload, preview, download
- [x] Video preview with subtitle overlay
- [x] SRT download
- [x] MP4 download (via /render)

### Phase 3 - Testing & Polish
- [x] Install FFmpeg (C:\ffmpeg\ffmpeg-master-latest-win64-gpl\bin) + auto-detect in backend/config.py
- [x] Fix /render for Windows (relative subtitle paths, Segoe UI font, UTF-8 BOM SRT)
- [x] Fix starlette/streamlit version conflict (starlette==0.47.3, fastapi>=0.138)
- [x] Hardware-accelerated rendering (Intel QSV auto-detect; ~4x faster than CPU)
- [x] Caption styles (color, outline, shadow, font, size, bold, box bg, top/middle/bottom position) + presets
- [x] Job-based /render with progress polling + single-frame style preview endpoint
- [x] Burned-caption position fix: this ffmpeg build's libass IGNORES `force_style Alignment` (5→top-left, 8→middle) — caption position MUST come from inline `{\an2/5/8}` tags injected into each SRT cue line (`renderer.inject_position`); also `/render/preview` input-seek silently drops subtitles unless `-copyts` is passed; `_parse_style_form` must whitelist "middle"
- [x] UI redesign (dark theme, step cards, style presets, live preview, progress bar)
- [x] E2E verified in real browser (upload → detect → style → preview → render → download)
- [x] New editor flow E2E: forced-align → detection card (Al-Ikhlas 1–4, 94%) → live caption overlay at correct times → language switch (ar/en/both) → style presets → editable ayah table → SRT/VTT download → render MP4 (h264_qsv)
- [ ] Test with multiple reciters and clips
- [x] Handle edge cases (mid-ayah starts, multi-surah clips, lead-in Bismillah) — TDD'd: test_vad.py, test_detector_all.py, test_align_multimatch.py, test_openers.py (opener tolerant spelling, lead-only takbir/isti'adha, local-path opener tags, SRT langs); repeated ayahs still rough
- [x] Pause-aware captions (breath-pause phrase splitting) — test_pause_split.py (5 scenarios + padding non-overlap + local/provider wiring)
- [ ] Performance optimization

## Deployment (HF Spaces: mhmdxch/quran-captions)
- Free tier **blocks Docker/Gradio spaces** (needs PRO); Streamlit SDK spaces stay free → deployed as `sdk: streamlit` with the FastAPI engine running as a **child process** inside the Space.
- `frontend/app.py` DEPLOYED mode (`SPACE_ID` env set): `API_URL=http://127.0.0.1:8000` (server-side calls), `PUBLIC_URL=""` (browser URLs are same-origin), `_ensure_backend()` (st.cache_resource) pre-downloads both HF repos + Silero, spawns `uvicorn backend.main:app` child, polls /health.
- No nginx/apt possible → `backend/config.py` falls back to `imageio_ffmpeg.get_ffmpeg_exe()` (static ffmpeg pip binary) when system ffmpeg missing.
- Browser can't reach the API port → `GET /media/{id}/preview` transcodes a 320px low-res MP4 (crf 32, cached in temp/); frontend embeds it as a base64 **data URI** in the custom caption player; render results are fetched server-side and delivered via `st.video(bytes)` + `st.download_button`.
- `/align` timeout raised 900 → 7200 (runs take ~1h on 2 vCPU). **Job mode** (2026-08-15): `POST /align` with `job=1` returns `{"job_id"}` immediately and runs in a daemon thread (`_align_job_worker`, `ALIGN_JOBS` dict + 24h purge) → `GET /align/status/{id}` (status/progress/stage, groq path advances 18→80 by slice via `progress_cb`), `GET /align/result/{id}` (409 if not done). Progress hooks (`p(stage,pct)`/`setp`) threaded through `_align_impl`, `_align_with_provider`, and the local forced-align loop (8 load, 15 VAD, 42+54·match% align, 97 build, 100). UI polls every 0.8s → linear `st.progress` with stage text (replaces 7200s sync POST). Sync mode preserved when `job` omitted.
- Space README (repo root) frontmatter: `sdk: streamlit, sdk_version: 1.58.0, app_file: frontend/app.py` — the API doesn't surface app_file/sdk_version, that's normal.
- requirements.txt pins `streamlit==1.58.0`, `fastapi==0.141.1` (AGENTS starlette==0.47.3 combo); torch unpinned (PyPI CUDA wheel, runs CPU — big build).
- **Quota**: free cpu-basic slots are exhausted by user's other spaces (credit-hf, fraud-dashboard are MAIN — never pause/delete; idp deleted with permission; hr-workflow-demo untouched). `restart` → 403 until a slot frees or PRO. Space is fully staged; one restart click (or PRO) starts it.
- Local emulation of deployed mode: `$env:SPACE_ID="x"` + `streamlit run frontend/app.py` + spawned uvicorn reproduces the Space behavior exactly (data-URI previews etc.).
